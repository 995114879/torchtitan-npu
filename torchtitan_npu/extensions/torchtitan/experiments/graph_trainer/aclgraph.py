# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare, partition and capture the AOT joint graph on NPU."""

import os
from collections import defaultdict
from functools import partial

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols, statically_known_true
from torch.fx.passes.split_module import split_module
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_leaves, tree_map
from torchtitan.experiments.graph_trainer.passes import _get_pass_name
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    register_pass_pipeline,
    register_post_init_hook,
)
from torchtitan.tools.logging import logger

from .aclgraph_runtime import ACLGraphWrapper, _CaptureCoordinator, _PartitionInterpreter


def _storages(node):
    return {
        StorageWeakRef(value.untyped_storage())
        for value in tree_leaves(node.meta.get("val"))
        if isinstance(value, torch.Tensor)
    }


def _known_output_metadata(value):
    scalar_types = (type(None), bool, int, float, complex, str, bytes, torch.SymInt, torch.SymFloat, torch.SymBool)
    if type(value) in (torch.Tensor, FakeTensor, *scalar_types):
        return True
    if type(value) in (tuple, list):
        return all(_known_output_metadata(item) for item in value)
    if type(value) is dict:
        # Pytree leaves omit keys and custom container context, which may hide aliases.
        return all(type(key) in scalar_types and _known_output_metadata(item) for key, item in value.items())
    return False


def _written_node_storages(node):
    written = set()
    if isinstance(node.target, torch._ops.OpOverload):
        for i, argument in enumerate(node.target._schema.arguments):
            if argument.alias_info and argument.alias_info.is_write:
                value = node.args[i] if i < len(node.args) else node.kwargs.get(argument.name)
                for leaf in tree_leaves(value):
                    if isinstance(leaf, torch.fx.Node):
                        written.update(_storages(leaf))
    return written


def _written_storages(gm):
    return set().union(*(_written_node_storages(node) for node in gm.graph.nodes))


def _storage_sensitive_inputs(gm):
    storages = set()
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload) and any(
            argument.name == "storage_offset" for argument in node.target._schema.arguments
        ):
            storages.update(set().union(*(_storages(parent) for parent in node.all_input_nodes)))
    return storages


def _preserve_storage(node, sensitive_storages):
    return bool(_storages(node) & sensitive_storages) or any(
        isinstance(value, torch.Tensor)
        and (
            not statically_known_true(value.storage_offset() == 0)
            or not value.is_contiguous()
            or value.is_conj()
            or value.is_neg()
        )
        for value in tree_leaves(node.meta.get("val"))
    )


def _partition_storage_safe_regions(gm, eager_nodes, static_storages):
    # Repartition only when an escaping view or a write would target a private
    # copy of a dynamic eager allocation. Ordinary reads remain capturable.
    while True:
        partitions, eager_partitions = {}, set()
        partition_id, previous_eager = -1, None
        for node in gm.graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            eager = node in eager_nodes
            if eager != previous_eager:
                partition_id += 1
                previous_eager = eager
            partitions[node] = partition_id
            if eager:
                eager_partitions.add(partition_id)
        dynamic_outputs, unsafe = {}, set()
        for node in gm.graph.nodes:
            storages = _storages(node)
            if node.op == "placeholder":
                dynamic_outputs[node] = storages - static_storages
                continue
            inputs = set().union(*(_storages(parent) for parent in node.all_input_nodes))
            dynamic = set().union(*(dynamic_outputs[parent] for parent in node.all_input_nodes))
            dynamic_outputs[node] = storages & dynamic
            if node in eager_nodes:
                dynamic_outputs[node].update(storages - inputs)
            elif node in partitions:
                escapes = any(user.op == "output" or partitions.get(user) != partitions[node] for user in node.users)
                # Container leaves are borrowed, not staged individually. Unpack
                # a dynamic eager container before passing tensors to capture.
                dynamic_container = any(
                    dynamic_outputs[parent] and not isinstance(parent.meta.get("val"), torch.Tensor)
                    for parent in node.all_input_nodes
                )
                if dynamic_container or _written_node_storages(node) & dynamic or (escapes and storages & dynamic):
                    unsafe.add(node)
        if not unsafe:
            return partitions, eager_partitions
        eager_nodes.update(unsafe)


def _mutated_input_indices(gm):
    written = _written_storages(gm)
    return {i for i, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder") if _storages(node) & written}


def _eager_boundary(node):
    # Dispatch, grouped expert GEMMs and combine share data-dependent EP
    # layouts. Keep the whole subtree eager, including recompute/backward.
    module_fqn = node.meta.get("custom", {}).get("module_fqn", "")
    return (
        "routed_experts" in module_fqn.split(".")
        or node.target == torch.ops.aten._local_scalar_dense.default
        or any(
            (isinstance(value, torch.Tensor) and (value.device.type == "cpu" or has_free_unbacked_symbols(value)))
            or (isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)) and has_free_unbacked_symbols(value))
            for parent in (node, *node.all_input_nodes)
            for value in tree_leaves(parent.meta.get("val"))
        )
    )


def reinplace_loss_accumulation_pass(gm, _example_inputs):
    """Restore chunked LM-head's private FP32 gradient accumulator after SAR.

    Functionalization makes the original add_ out of place. Only reuse fresh
    copies (or their exclusively consumed accumulation chain), never graph
    inputs, state, or tensors with another reader/view.
    """
    owned = set()
    changed = 0
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.add.Tensor or node.meta.get("custom", {}).get("module_fqn") != "loss":
            continue
        lhs, rhs = node.args[:2]
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            continue
        left, right, output = (value.meta.get("val") for value in (lhs, rhs, node))
        if not (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and isinstance(output, torch.Tensor)
            and left.dtype == right.dtype == output.dtype == torch.float32
            and left.ndim == 2
            and left.shape == right.shape == output.shape
            and left.is_contiguous()
            and (lhs.target == torch.ops.aten._to_copy.default or lhs in owned)
            and set(lhs.users) == {node}
        ):
            continue
        old_storage = StorageWeakRef(output.untyped_storage())
        aliases = [
            value
            for candidate in gm.graph.nodes
            for value in tree_leaves(candidate.meta.get("val"))
            if isinstance(value, torch.Tensor) and StorageWeakRef(value.untyped_storage()) == old_storage
        ]
        if any(value.dtype != left.dtype for value in aliases):
            continue

        def rebase(value, *, old_storage=old_storage, left=left, output=output):
            if isinstance(value, torch.Tensor) and StorageWeakRef(value.untyped_storage()) == old_storage:
                return left.as_strided(
                    value.shape,
                    value.stride(),
                    left.storage_offset() + value.storage_offset() - output.storage_offset(),
                )
            return value

        node.target = torch.ops.aten.add_.Tensor
        for candidate in gm.graph.nodes:
            if "val" in candidate.meta:
                candidate.meta["val"] = tree_map(rebase, candidate.meta["val"])
        owned.add(node)
        changed += 1
    gm.graph.lint()
    gm.recompile()
    logger.info("ACLGraph reused %d exclusively owned FP32 loss accumulators", changed)
    return gm


def aclgraph_pass(
    gm,
    example_inputs,
    *,
    static_input_indices,
):
    """Capture stable device work; retain host and routed-expert work in eager."""
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    input_storages = [_storages(node) for node in placeholders]
    static_storages = set().union(*(input_storages[i] for i in static_input_indices))
    mutated_storages = set().union(*(input_storages[i] for i in _mutated_input_indices(gm)))
    returned = next(node for node in gm.graph.nodes if node.op == "output")
    returned_nodes = [node for node in tree_leaves(returned.args) if isinstance(node, torch.fx.Node)]
    # make_fx can omit val for lifted None/scalar inputs returned as gradient
    # placeholders. Their example values contain no tensor aliases.
    for node, example in zip(placeholders, example_inputs, strict=False):
        if (
            node in returned_nodes
            and "val" not in node.meta
            and type(example) in (type(None), bool, int, float, complex, str, bytes)
        ):
            node.meta["val"] = example
    escaping_storages = set().union(*(_storages(node) for node in returned_nodes))
    # Caller-owned context such as DeviceMesh passes through unchanged; only
    # computed outputs need a known pytree layout for per-step copies.
    unknown_nodes = [
        node.name
        for node in returned_nodes
        if "val" not in node.meta
        or (node.op not in ("placeholder", "get_attr") and not _known_output_metadata(node.meta["val"]))
    ]
    unknown_metadata = bool(unknown_nodes)
    if unknown_metadata:
        examples = dict(zip(placeholders, example_inputs, strict=False))
        logger.warning(
            "ACLGraph keeps region output copies because alias metadata is incomplete at %s",
            [
                (node.name, "val" in node.meta, type(node.meta.get("val")).__name__, type(examples.get(node)).__name__)
                for node in dict.fromkeys(returned_nodes)
                if node.name in unknown_nodes
            ][:10],
        )
    output_alias_counts = defaultdict(int)
    for node in returned_nodes:
        for value in tree_leaves(node.meta.get("val")):
            if isinstance(value, torch.Tensor):
                output_alias_counts[StorageWeakRef(value.untyped_storage())] += 1
    # Caller-owned aliases must keep their original addresses. Capture-owned
    # aliases can stay in graphs: copy them together after the step's last write.
    eager_alias_storages = escaping_storages & set().union(*input_storages)
    eager_alias_storages.update(
        StorageWeakRef(value.untyped_storage())
        for node in returned_nodes
        for value in tree_leaves(node.meta.get("val"))
        if isinstance(value, torch.Tensor)
        and (
            value.layout != torch.strided
            or value.is_quantized
            or value.is_conj()
            or value.is_neg()
            # Empty allocations have no address for native-handle alias checks.
            or statically_known_true(value.untyped_storage().nbytes() == 0)
        )
    )
    if unknown_metadata:
        eager_alias_storages.update(storage for storage, count in output_alias_counts.items() if count > 1)
        eager_alias_storages.update(escaping_storages & _written_storages(gm))
    staged_input_indices = {
        i
        for i, node in enumerate(placeholders)
        if i not in static_input_indices
        and isinstance(value := node.meta.get("val"), torch.Tensor)
        and value.device.type == "npu"
        and value.is_contiguous()
        and len(input_storages[i]) == 1
        and not input_storages[i] & (mutated_storages | static_storages | escaping_storages)
    }
    storage_inputs = defaultdict(list)
    for i, storages in enumerate(input_storages):
        for storage in storages:
            storage_inputs[storage].append(i)
    input_aliases = {}
    for indices in storage_inputs.values():
        if len(indices) < 2:
            continue
        first = placeholders[indices[0]].meta["val"]
        first_layout = (*first.shape, *first.stride(), first.storage_offset())
        if all(
            i in staged_input_indices
            and (value := placeholders[i].meta["val"]).dtype == first.dtype
            and len(value.shape) == len(first.shape)
            and all(
                statically_known_true(a == b)
                for a, b in zip((*value.shape, *value.stride(), value.storage_offset()), first_layout, strict=True)
            )
            for i in indices
        ):
            input_aliases.update((i, indices[0]) for i in indices[1:])
        else:
            staged_input_indices.difference_update(indices)
    static_storages.update(set().union(*(input_storages[i] for i in staged_input_indices)))
    eager_nodes = {
        node
        for node in gm.graph.nodes
        if node.op not in ("placeholder", "get_attr", "output")
        and (
            _eager_boundary(node)
            or any(_storages(value) & eager_alias_storages for value in (node, *node.all_input_nodes))
        )
    }
    partitions, eager_partitions = _partition_storage_safe_regions(gm, eager_nodes, static_storages)
    sensitive_storages = _storage_sensitive_inputs(gm)
    result = split_module(gm, gm, partitions.__getitem__, keep_original_order=True, keep_original_node_name=True)
    # Borrow producer buffers instead of retaining a second copy of every
    # activation/unsharded weight. During recording FX owns their lifetime.
    captured_storages = set()
    cloned_nodes = set()
    region_clone_indices: dict[str, set[int] | None] = {}
    for name, region in result.named_children():
        assert isinstance(region, torch.fx.GraphModule)
        if int(name.rsplit("_", 1)[1]) not in eager_partitions:
            output = next(node for node in region.graph.nodes if node.op == "output")
            output_nodes = [node for node in tree_leaves(output.args) if isinstance(node, torch.fx.Node)]
            if unknown_metadata or any("val" not in node.meta for node in output_nodes):
                region_clone_indices[name] = None
                cloned_nodes.update(node.name for node in output_nodes)
            else:
                # Keep the original storage through every internal reader and
                # writer. The root output owns the final per-step copies.
                region_clone_indices[name] = set()
            region_input_storages = set().union(
                *(_storages(node) for node in region.graph.nodes if node.op == "placeholder")
            )
            for node in tree_leaves(output.args):
                if isinstance(node, torch.fx.Node):
                    # Views of eager inputs inherit their producer's changing
                    # addresses; only allocations made here are capture-owned.
                    captured_storages.update(_storages(node) - region_input_storages)
    for node in placeholders:
        captured_storages.difference_update(_storages(node))
    captured_storages.difference_update(eager_alias_storages)
    clone_output_indices = set()
    if not unknown_metadata:
        offset = 0
        for leaf in tree_leaves(returned.args):
            values = tree_leaves(leaf.meta["val"]) if isinstance(leaf, torch.fx.Node) else [leaf]
            clone_output_indices.update(
                offset + i
                for i, value in enumerate(values)
                if isinstance(value, torch.Tensor) and StorageWeakRef(value.untyped_storage()) in captured_storages
            )
            offset += len(values)
    producers = {node.name: partitions.get(node) for node in gm.graph.nodes}
    coordinator = _CaptureCoordinator()
    summaries = []
    for name, region in result.named_children():
        assert isinstance(region, torch.fx.GraphModule)
        eager = int(name.rsplit("_", 1)[1]) in eager_partitions
        summaries.append(
            {
                "name": name,
                "eager": eager,
                "capture": not eager,
            }
        )
        if eager:
            continue
        clone_indices = region_clone_indices[name]
        region_inputs = [node for node in region.graph.nodes if node.op == "placeholder"]
        # SAR can reuse forward FakeTensor storage metadata on eager
        # recomputations. Trust the actual FX producer, not storage alone.
        captured_inputs = {
            i
            for i, node in enumerate(region_inputs)
            if (producer := producers.get(node.name)) is not None
            and producer not in eager_partitions
            and node.name not in cloned_nodes
        }
        mutated = _mutated_input_indices(region)
        written_storages = set().union(*(_storages(region_inputs[i]) for i in mutated))
        static_indices = (
            mutated
            | captured_inputs
            | {i for i, node in enumerate(region_inputs) if _storages(node) & (static_storages | written_storages)}
        )
        # A mixed container may also own eager/state buffers. Borrow it only
        # when every tensor storage comes from a captured producer.
        borrowed_indices = {
            i
            for i, node in enumerate(region_inputs)
            if i in captured_inputs and (storages := _storages(node)) and storages <= captured_storages
        }
        output = next(node for node in region.graph.nodes if node.op == "output")
        output_nodes = [node for node in tree_leaves(output.args) if isinstance(node, torch.fx.Node)]
        reusable_indices = set()
        if all(node.meta.get("val") is not None and _known_output_metadata(node.meta["val"]) for node in output_nodes):
            output_storages = set().union(*(_storages(node) for node in output_nodes))
            reusable_indices = {
                i
                for i, node in enumerate(region_inputs)
                if i not in static_indices
                and isinstance(value := node.meta.get("val"), torch.Tensor)
                and value.layout == torch.strided
                and value.is_contiguous()
                and (storages := _storages(node))
                and not storages & output_storages
            }
        wrapper = ACLGraphWrapper(
            region.forward,
            static_input_indices=static_indices,
            capture_runnable=torch.fx.Interpreter(region).boxed_run,
            label=name,
            clone_outputs=clone_indices is None or bool(clone_indices),
            clone_output_indices=clone_indices,
            capture_coordinator=coordinator,
            borrowed_input_indices=borrowed_indices,
            capture_boxed=True,
            reusable_input_indices=reusable_indices,
            preserve_storage_indices={
                i for i, node in enumerate(region_inputs) if _preserve_storage(node, sensitive_storages)
            },
        )
        object.__setattr__(region, "_aclgraph_wrapper", wrapper)
        region.forward = wrapper
        coordinator.wrappers.append(wrapper)
    result.meta.update(gm.meta)
    result.meta["aclgraph_partitions"] = summaries
    object.__setattr__(result, "_aclgraph_wrappers", coordinator.wrappers)
    object.__setattr__(result, "_aclgraph_capture_coordinator", coordinator)
    result.forward = _PartitionInterpreter(
        result,
        staged_input_indices=staged_input_indices,
        input_aliases=input_aliases,
        clone_output_indices=clone_output_indices,
        preserve_storage_indices={
            i for i, node in enumerate(placeholders) if _preserve_storage(node, sensitive_storages)
        },
    ).run
    logger.info(
        "Applied ACLGraph pass: %d capture regions, %d eager regions; %d reusable region inputs",
        len(coordinator.wrappers),
        len(eager_partitions),
        sum(len(wrapper.reusable_input_indices) for wrapper in coordinator.wrappers),
    )
    logger.info("ACLGraph copies %d tensor output leaves at the joint-step exit", len(clone_output_indices))
    return result


def _close_aclgraph_trainer(trainer, original_close):
    gm = getattr(getattr(trainer, "_traced_step", None), "gm", None)
    for wrapper in getattr(gm, "_aclgraph_wrappers", []):
        # Release captured HCCL references before process-group teardown.
        wrapper.teardown()
    coordinator = getattr(gm, "_aclgraph_capture_coordinator", None)
    if coordinator is not None:
        coordinator.teardown()
    original_close()


def _aclgraph_enabled(config):
    compile_config = config.compile
    return getattr(compile_config, "enable_passes", True) and "aclgraph_pass" not in getattr(
        compile_config, "disable_passes", ()
    )


def _validate_aclgraph_config(config, parallel_dims):
    if config.compile.mode != "aot_fx_trace" or config.compile.inductor_compilation != "regional":
        raise ValueError("ACLGraph pipelines require aot_fx_trace with regional compilation")
    if config.compile.ep_overlap.enabled:
        raise ValueError("ACLGraph requires EP overlap disabled")
    if getattr(parallel_dims, "pp_enabled", False):
        raise ValueError("ACLGraph does not support pipeline parallelism; set PP=1 or disable ACLGraph")
    if getattr(parallel_dims, "cp_enabled", False):
        raise ValueError("ACLGraph fixed document capacity requires context parallel disabled")


@register_post_init_hook("mutation-functionalization+aclgraph")
def prepare_aclgraph_trainer(trainer):
    if not _aclgraph_enabled(trainer.config):
        # A CLI disable list replaces the recipe's default list. Restore its
        # NPU-safe CUDA graph exclusion when only aclgraph_pass was supplied.
        disabled = trainer.config.compile.disable_passes
        if "aclgraph_pass" in disabled and "cudagraph_pass" not in disabled:
            disabled.append("cudagraph_pass")
        return
    _validate_aclgraph_config(trainer.config, trainer.parallel_dims)
    import torch_npu

    get_queue_mode = getattr(torch_npu.npu, "get_task_queue_enable", None)
    queue_mode = get_queue_mode() if get_queue_mode is not None else int(os.environ.get("TASK_QUEUE_ENABLE", "1"))
    if queue_mode == 2:
        set_queue_mode = getattr(torch_npu.npu, "set_task_queue_enable", None)
        if set_queue_mode is None:
            raise ValueError("ACLGraph requires TASK_QUEUE_ENABLE=1/0 or torch_npu's dynamic task-queue API")
        # Switch once before training; changing modes drains queued submissions.
        set_queue_mode(1)
        logger.warning("ACLGraph switched task queue mode 2 -> 1; NPU graph capture does not support mode 2")
    trainer.close = partial(_close_aclgraph_trainer, trainer, trainer.close)
    for model in trainer.model_parts:
        prepare = getattr(model, "prepare_aclgraph", None)
        if prepare is not None:
            seq_len = trainer.config.training.seq_len
            local_batch_size = trainer.config.training.local_batch_size
            document_alignment = getattr(trainer.config.dataloader, "per_doc_alignment", 1)
            if document_alignment < 1 or seq_len % document_alignment:
                raise ValueError("ACLGraph requires a positive dataloader.per_doc_alignment that divides seq_len")
            # Packed segments occupy at least one alignment unit. Unknown loaders
            # use the one-token bound; the metadata builder still checks each batch.
            document_capacity = local_batch_size * (seq_len // document_alignment)
            prepare(document_capacity=document_capacity)
            logger.info(
                "ACLGraph model prepared: model=%s document_capacity=%d; routed experts remain eager",
                type(model).__name__,
                document_capacity,
            )


@register_pass_pipeline("mutation-functionalization+aclgraph")
def construct_aclgraph_passes(traced_result, config, *, parallel_dims=None):
    aclgraph_enabled = _aclgraph_enabled(config)
    if aclgraph_enabled:
        _validate_aclgraph_config(config, parallel_dims)
    passes = PASS_PIPELINE_REGISTRY["mutation-functionalization"](traced_result, config, parallel_dims=parallel_dims)
    if not aclgraph_enabled:
        return passes
    passes = [p for p in passes if _get_pass_name(p) not in {"cudagraph_pass", "insert_kernel_annotations_pass"}]
    return [
        *passes,
        reinplace_loss_accumulation_pass,
        partial(
            aclgraph_pass,
            static_input_indices=list(range(traced_result.num_static_inputs)),
        ),
    ]
