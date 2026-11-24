# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ACLGraph capture/replay, input guards and buffer ownership."""

from datetime import timedelta
from operator import itemgetter

import torch
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import keystr, tree_flatten, tree_flatten_with_path, tree_leaves, tree_map, tree_unflatten
from torchtitan.tools.logging import logger


def _storage_key(value):
    storage = value.untyped_storage()
    # Native non-owning handles have separate StorageImpls pointing at the
    # same allocation. StorageWeakRef alone would split aliases on replay.
    address = storage.data_ptr()
    return (value.device, address) if address else StorageWeakRef(storage)


def _staging_source(arg, *, preserve_storage=True):
    if not preserve_storage:
        return arg
    # Byte views preserve even a storage whose size is not divisible by dtype.
    return torch.empty(0, dtype=torch.uint8, device=arg.device).set_(
        arg.untyped_storage(), 0, (arg.untyped_storage().nbytes(),), (1,)
    )


def _storage_view(storage, template):
    num_bytes = storage.numel() // template.element_size() * template.element_size()
    base = storage if num_bytes == storage.numel() else storage[:num_bytes]
    value = base.view(template.dtype).as_strided(template.shape, template.stride(), template.storage_offset())
    if template.is_conj():
        value = value.conj()
    if template.is_neg():
        value = value._neg_view()
    return value


def _clone_staging_input(arg, *, preserve_storage=True):
    if not preserve_storage or (
        arg.storage_offset() == 0
        and arg.is_contiguous()
        and arg.nbytes == arg.untyped_storage().nbytes()
        and not arg.is_conj()
        and not arg.is_neg()
    ):
        return arg.detach().clone()
    return _storage_view(_staging_source(arg).clone(), arg)


def _copy_staging_input(saved, arg, *, preserve_storage=True):
    if arg.storage_offset() != saved.storage_offset() or (
        preserve_storage and arg.untyped_storage().nbytes() != saved.untyped_storage().nbytes()
    ):
        raise ValueError("ACLGraph staging storage_offset or storage size changed")
    _staging_source(saved, preserve_storage=preserve_storage).copy_(
        _staging_source(arg, preserve_storage=preserve_storage)
    )


def _tensor_input_spec(value):
    return (
        value.shape,
        value.stride(),
        value.dtype,
        value.device,
        value.data_ptr(),
        value.storage_offset(),
        value.untyped_storage().nbytes(),
        type(value),
        torch._C._dispatch_keys(value).raw_repr(),
    )


@torch.no_grad()
def _clone_graph_outputs(output, clone_indices):
    leaves, spec = tree_flatten(output)
    copied_storages = {_storage_key(leaves[i]) for i in clone_indices}
    aliases = {}
    for i, value in enumerate(leaves):
        if isinstance(value, torch.Tensor):
            storage = _storage_key(value)
            if storage in copied_storages:
                aliases.setdefault(storage, []).append(i)
    for indices in aliases.values():
        first = max((leaves[i] for i in indices), key=lambda value: value.untyped_storage().nbytes())
        if (
            len(indices) == 1
            and first.storage_offset() == 0
            and first.is_contiguous()
            and first.nbytes == first.untyped_storage().nbytes()
        ):
            leaves[indices[0]] = first.clone()
            continue
        # Copy raw storage once, then rebuild every returned view, including
        # dtype views and absolute offsets. Separate clones would break aliases.
        source = torch.empty(0, dtype=torch.uint8, device=first.device).set_(
            first.untyped_storage(), 0, (first.untyped_storage().nbytes(),), (1,)
        )
        copied = source.clone()
        views = {}
        for i in indices:
            value = leaves[i]
            if id(value) not in views:
                views[id(value)] = _storage_view(copied, value)
            leaves[i] = views[id(value)]
    return tree_unflatten(leaves, spec)


class _CaptureCoordinator:
    """Align initial warmup/capture across ranks; stable replay stays local."""

    def __init__(self):
        self.wrappers = []
        self.group: torch.distributed.ProcessGroup | None = None
        self.initialized = False
        self.graph_pool = None
        self.shared_input_buffers = {}
        self._input_buffer_streams = {}

    def check_input_buffer_streams(self):
        if not self._input_buffer_streams:
            return
        import torch_npu

        for device, stream in self._input_buffer_streams.items():
            if torch_npu.npu.current_stream(device) != stream:
                raise ValueError("ACLGraph shared input buffers require the same replay stream across calls")

    def stage_input(self, arg, used_slots, *, preserve_storage=True):
        # Each region gets distinct slots for simultaneous inputs. Regions can
        # reuse them only when none of their outputs aliases the staged input.
        storage_size = arg.untyped_storage().nbytes() if preserve_storage else None
        key = (arg.shape, arg.stride(), arg.dtype, arg.device, arg.storage_offset(), storage_size)
        slot = used_slots.get(key, 0)
        used_slots[key] = slot + 1
        if arg.device.type == "npu":
            import torch_npu

            stream = torch_npu.npu.current_stream(arg.device)
            previous = self._input_buffer_streams.setdefault(arg.device, stream)
            if stream != previous:
                raise ValueError("ACLGraph shared input buffers require the same replay stream across regions")
        buffers = self.shared_input_buffers.setdefault(key, [])
        if slot == len(buffers):
            buffers.append(_clone_staging_input(arg, preserve_storage=preserve_storage))
        else:
            # Initial captures of later regions also consume the current input.
            _copy_staging_input(buffers[slot], arg, preserve_storage=preserve_storage)
        return buffers[slot]

    def before_capture(self):
        if not self.initialized:
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                group = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=600))
                assert isinstance(group, torch.distributed.ProcessGroup)
                self.group = group
                names = [value.label for value in self.wrappers]
                gathered: list[list[str] | None] = [None] * torch.distributed.get_world_size(group)
                torch.distributed.all_gather_object(gathered, names, group=group)
                if any(value != names for value in gathered):
                    raise ValueError("ACLGraph requires identical region order across ranks")
            self.initialized = True
        if self.group is not None:
            with torch.profiler.record_function("ACLGraph/capture_coordination"):
                torch.distributed.barrier(group=self.group)

    def teardown(self):
        if self.group is not None:
            torch.distributed.destroy_process_group(self.group)
            self.group = None
        self.graph_pool = None
        self.shared_input_buffers.clear()
        self._input_buffer_streams.clear()


class ACLGraphWrapper:
    """Warm up once, then capture/replay using stable input buffers.

    Lifted state and mutated inputs retain their addresses. Escaping gradients
    are copied: GraphTrainer may assign a returned gradient to ``param.grad``.
    Intermediates use non-owning replay handles; owning capture-time results
    follow normal FX liveness. Regions share one pool in a fixed capture order.
    """

    def __init__(
        self,
        runnable,
        *,
        static_input_indices,
        capture_runnable=None,
        label="joint step",
        clone_outputs=True,
        clone_output_indices=None,
        capture_coordinator=None,
        borrowed_input_indices=(),
        capture_boxed=False,
        reusable_input_indices=(),
        preserve_storage_indices=None,
    ):
        self.runnable = runnable
        self.capture_runnable = capture_runnable or runnable
        self.label = label
        self.static_input_indices = set(static_input_indices)
        self.clone_outputs = clone_outputs
        self.clone_output_indices = None if clone_output_indices is None else set(clone_output_indices)
        self.capture_coordinator = capture_coordinator
        self.borrowed_input_indices = set(borrowed_input_indices)
        if not self.borrowed_input_indices <= self.static_input_indices:
            raise ValueError("ACLGraph borrowed inputs must also be static inputs")
        self.capture_boxed = capture_boxed
        self.reusable_input_indices = set(reusable_input_indices)
        # Standalone callables lack FX access information: keep their storage.
        self.preserve_storage_indices = None if preserve_storage_indices is None else set(preserve_storage_indices)
        self.warmed_up = False
        self.graph = None
        self.args = None
        self.input_specs = None
        self.output = None
        self._flat_outputs = None
        self._output_spec = None
        self._clone_output_indices: tuple[int, ...] = ()
        self._input_guard = None
        self._static_pointers = ()
        self._dynamic_layouts = ()
        self._scalar_inputs = ()
        self._container_inputs = {}

    def _prepare_input_guard(self, args):
        guard = RootGuardManager()
        guard.add_length_check_guard(len(args), [], None)
        pointers, layouts, scalars = [], [], []
        self._container_inputs = {}
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                leaves, structure = tree_flatten_with_path(arg)
                if structure.is_leaf():
                    scalars.append((i, arg))
                else:
                    # Containers are borrowed as-is during capture. Their tensor
                    # leaves need address guards, never Python value equality.
                    self._container_inputs[i] = (
                        structure,
                        [
                            (
                                keystr(path),
                                None if isinstance(value, torch.Tensor) else value,
                                _tensor_input_spec(value) if isinstance(value, torch.Tensor) else None,
                            )
                            for path, value in leaves
                        ],
                    )
                continue
            manager = guard.list_getitem_manager(i, f"input[{i}]", arg, GuardManagerType.GUARD_MANAGER)
            manager.add_tensor_match_guard(
                arg,
                list(arg.shape),
                list(arg.stride()),
                f"input[{i}]",
                [],
                None,
                type(arg),
                torch._C._dispatch_keys(arg),
            )
            if i in self.static_input_indices:
                pointers.append((i, arg.data_ptr(), arg.storage_offset()))
            else:
                preserve = self.preserve_storage_indices is None or i in self.preserve_storage_indices
                layouts.append((i, arg.storage_offset(), arg.untyped_storage().nbytes() if preserve else None))
        self._input_guard = guard
        self._static_pointers = tuple(pointers)
        self._dynamic_layouts = tuple(layouts)
        self._scalar_inputs = tuple(scalars)

    @staticmethod
    def _tensor_change_reason(arg, spec, *, static, preserve_storage=True):
        if not isinstance(arg, torch.Tensor):
            return f"Tensor -> {type(arg).__name__}"
        if type(arg) is not spec[7]:
            return f"Tensor type {spec[7].__name__} -> {type(arg).__name__}"
        if torch._C._dispatch_keys(arg).raw_repr() != spec[8]:
            return "dispatch keys changed"
        for field, old, new in zip(
            ("shape", "stride", "dtype", "device"),
            spec[:4],
            (arg.shape, arg.stride(), arg.dtype, arg.device),
            strict=True,
        ):
            if old != new:
                return f"{field} {old} -> {new}"
        if arg.storage_offset() != spec[5]:
            return f"storage_offset {spec[5]} -> {arg.storage_offset()}"
        if preserve_storage and arg.untyped_storage().nbytes() != spec[6]:
            return f"storage size {spec[6]} -> {arg.untyped_storage().nbytes()}"
        if static and arg.data_ptr() != spec[4]:
            return f"static address {spec[4]:#x} -> {arg.data_ptr():#x}"
        return None

    def _input_change_reason(self, args):
        assert self.args is not None and self.input_specs is not None
        if len(args) != len(self.args):
            return f"argument count {len(self.args)} -> {len(args)}"
        for i, (structure, saved_leaves) in self._container_inputs.items():
            leaves, current_structure = tree_flatten_with_path(args[i])
            if current_structure != structure:
                return f"input[{i}] container structure changed"
            for (_, value), (path, saved, spec) in zip(leaves, saved_leaves, strict=True):
                if spec is not None:
                    reason = self._tensor_change_reason(value, spec, static=True)
                    if reason is not None:
                        return f"input[{i}]{path} {reason}"
                elif value is not saved and (type(value) is not type(saved) or value != saved):
                    return f"input[{i}]{path} scalar {saved!r} -> {value!r}"
        # C++ guards check all tensor layouts without a Python loop per input.
        # Address checks remain explicit: tensor identity does not guard .data.
        if self._input_guard is not None and self._input_guard.check(args):
            for i, address, offset in self._static_pointers:
                if args[i].storage_offset() != offset:
                    return f"input[{i}] storage_offset {offset} -> {args[i].storage_offset()}"
                if args[i].data_ptr() != address:
                    return f"input[{i}] static address {address:#x} -> {args[i].data_ptr():#x}"
            for i, offset, storage_size in self._dynamic_layouts:
                if args[i].storage_offset() != offset:
                    return f"input[{i}] storage_offset {offset} -> {args[i].storage_offset()}"
                if storage_size is not None and args[i].untyped_storage().nbytes() != storage_size:
                    return f"input[{i}] storage size {storage_size} -> {args[i].untyped_storage().nbytes()}"
            for i, saved in self._scalar_inputs:
                arg = args[i]
                if arg is not saved and (type(arg) is not type(saved) or arg != saved):
                    return f"input[{i}] scalar {saved!r} -> {arg!r}"
            return None
        for i, (arg, saved, spec) in enumerate(zip(args, self.args, self.input_specs, strict=True)):
            if isinstance(saved, torch.Tensor):
                assert spec is not None
                preserve = self.preserve_storage_indices is None or i in self.preserve_storage_indices
                reason = self._tensor_change_reason(
                    arg,
                    spec,
                    static=i in self.static_input_indices,
                    preserve_storage=preserve,
                )
                if reason is not None:
                    return f"input[{i}] {reason}"
            elif (
                i not in self._container_inputs and arg is not saved and (type(arg) is not type(saved) or arg != saved)
            ):
                return f"input[{i}] scalar {saved!r} -> {arg!r}"
        return "C++ tensor guard rejected an input" if self._input_guard is not None else None

    def __call__(self, *args):
        return self.call_boxed(list(args))

    @torch.no_grad()
    def call_boxed(self, args):
        import torch_npu

        if self.graph is not None:
            reason = self._input_change_reason(args)
            if reason is not None:
                raise ValueError(f"ACLGraph static replay inputs changed at {self.label}: {reason}")
        elif self.capture_coordinator is not None:
            self.capture_coordinator.before_capture()
        if not self.warmed_up:
            output = self.runnable(*args)
            self.warmed_up = True
            logger.info("ACLGraph %s: warmup completed", self.label)
            return output
        if self.graph is None:
            saved_args, used_slots, shared_storages = [], {}, set()
            for i, arg in enumerate(args):
                preserve = self.preserve_storage_indices is None or i in self.preserve_storage_indices
                if i in self.static_input_indices or not isinstance(arg, torch.Tensor):
                    saved = arg
                elif (
                    self.capture_coordinator is not None
                    and i in self.reusable_input_indices
                    and type(arg) is torch.Tensor
                    and arg.layout == torch.strided
                    and not arg.is_quantized
                    and arg.is_contiguous()
                ):
                    saved = self.capture_coordinator.stage_input(arg, used_slots, preserve_storage=preserve)
                    shared_storages.add(_storage_key(saved))
                else:
                    saved = _clone_staging_input(arg, preserve_storage=preserve)
                saved_args.append(saved)
            self.args = tuple(saved_args)
            del saved_args
            self.input_specs = [_tensor_input_spec(arg) if isinstance(arg, torch.Tensor) else None for arg in args]
            self._prepare_input_guard(args)
            for i, (arg, saved) in enumerate(zip(args, self.args, strict=True)):
                if isinstance(arg, torch.Tensor) and arg.stride() != saved.stride():
                    raise ValueError(
                        f"ACLGraph {self.label}: dynamic input[{i}] copy must preserve strides; "
                        f"shape={tuple(arg.shape)}, source={arg.stride()}, copied={saved.stride()}"
                    )
            graph = torch_npu.npu.NPUGraph()
            coordinator = self.capture_coordinator
            if coordinator is not None:
                if coordinator.graph_pool is None:
                    coordinator.graph_pool = torch_npu.npu.graph_pool_handle()
                pool = coordinator.graph_pool
            else:
                pool = torch_npu.npu.graph_pool_handle()
            capture_inputs = list(self.args)
            protected_storages = {
                _storage_key(value)
                for i, arg in enumerate(self.args)
                if i not in self.borrowed_input_indices
                for value in tree_leaves(arg)
                if isinstance(value, torch.Tensor)
            }

            def replay_handle(value):
                if (
                    coordinator is not None
                    and isinstance(value, torch.Tensor)
                    and value.device.type == "npu"
                    and _storage_key(value) not in protected_storages
                ):
                    return torch_npu._C._weak_ref_tensor(value)
                return value

            self.args = tuple(
                tree_map(replay_handle, arg) if i in self.borrowed_input_indices else arg
                for i, arg in enumerate(self.args)
            )
            args.clear()
            # Loop locals must not keep borrowed inputs alive during capture.
            arg = saved = None
            # All regions use the same default capture stream and execute once,
            # in joint-FX order. Capture-time outputs follow the caller's normal
            # liveness; replay only needs non-owning handles to their addresses.
            with torch_npu.npu.graph(graph, pool=pool):
                self.output = (
                    self.capture_runnable(capture_inputs)
                    if self.capture_boxed
                    else self.capture_runnable(*capture_inputs)
                )
                # Join pending HCCL streams before ending this region's capture.
                for value in tree_leaves(self.output):
                    if isinstance(value, torch.Tensor) and value.device.type == "npu":
                        torch.ops._c10d_functional.wait_tensor.default(value)
            self.graph = graph
            captured_output = self.output
            if any(
                isinstance(value, torch.Tensor) and _storage_key(value) in shared_storages
                for value in tree_leaves(captured_output)
            ):
                raise ValueError(f"ACLGraph reusable input unexpectedly aliases a captured output at {self.label}")
            if coordinator is not None:
                self.output = tree_map(replay_handle, captured_output)
                owner = getattr(self.capture_runnable, "__self__", None)
                if isinstance(owner, torch.fx.Interpreter):
                    owner.env.clear()
            if self.clone_outputs:
                static_storages = {
                    _storage_key(value)
                    for i, arg in enumerate(self.args)
                    if i in self.static_input_indices and i not in self.borrowed_input_indices
                    for value in tree_leaves(arg)
                    if isinstance(value, torch.Tensor)
                }
                self._flat_outputs, self._output_spec = tree_flatten(self.output)
                assert self.clone_output_indices is None or self.clone_output_indices <= set(
                    range(len(self._flat_outputs))
                )
                self._clone_output_indices = tuple(
                    i
                    for i, value in enumerate(self._flat_outputs)
                    if isinstance(value, torch.Tensor)
                    and (self.clone_output_indices is None or i in self.clone_output_indices)
                    and _storage_key(value) not in static_storages
                )
            logger.info("ACLGraph %s: capture SUCCEEDED", self.label)
        else:
            captured_output = None
            assert self.args is not None
            for i, (arg, saved) in enumerate(zip(args, self.args, strict=True)):
                if isinstance(saved, torch.Tensor) and i not in self.static_input_indices:
                    preserve = self.preserve_storage_indices is None or i in self.preserve_storage_indices
                    _copy_staging_input(saved, arg, preserve_storage=preserve)
        assert self.graph is not None
        self.graph.replay()
        if not self.clone_outputs:
            # During recording, owning tensors keep inputs live for subsequent
            # captures until their last FX consumer. Replay allocates no tensors
            # in this pool, so non-owning handles suffice on subsequent steps.
            return captured_output if captured_output is not None else self.output
        assert self._flat_outputs is not None and self._output_spec is not None
        # Uncloned intermediates must keep their owning capture-time tensors
        # until the last FX consumer, even when another output is copied.
        outputs = tree_leaves(captured_output) if captured_output is not None else list(self._flat_outputs)
        for i in self._clone_output_indices:
            outputs[i] = outputs[i].clone()
        return tree_unflatten(outputs, self._output_spec)

    def teardown(self):
        self.graph = None
        self.args = None
        self.input_specs = None
        self.output = None
        self._flat_outputs = None
        self._output_spec = None
        self._clone_output_indices = ()
        self._input_guard = None
        self._static_pointers = ()
        self._dynamic_layouts = ()
        self._scalar_inputs = ()
        self._container_inputs = {}
        owner = getattr(self.capture_runnable, "__self__", None)
        if isinstance(owner, torch.fx.Interpreter):
            owner.env.clear()


class _PartitionInterpreter(torch.fx.Interpreter):
    """Release the caller's last references before entering a boxed region."""

    def __init__(
        self,
        module,
        *,
        staged_input_indices=(),
        input_aliases=None,
        clone_output_indices=(),
        preserve_storage_indices=None,
    ):
        super().__init__(module)
        self._capture_coordinator = getattr(module, "_aclgraph_capture_coordinator", None)
        self._staged_input_indices = tuple(sorted(staged_input_indices))
        self._producer_input_indices = None
        self._input_buffers = {}
        self._input_aliases = dict(input_aliases or {})
        self._clone_output_indices = tuple(sorted(clone_output_indices))
        self._preserve_storage_indices = None if preserve_storage_indices is None else set(preserve_storage_indices)
        # The pass uses each storage group's first input as its canonical source.
        staged_indices = set(self._staged_input_indices)
        for i, source in self._input_aliases.items():
            if i not in staged_indices or source not in staged_indices or source >= i or source in self._input_aliases:
                raise ValueError(
                    "ACLGraph input aliases must point to earlier non-aliased staged inputs: "
                    f"input[{i}] -> input[{source}]"
                )
        # Keep the last-use map, but perform collection before boxed calls.
        self.garbage_collect_values = False
        self._inputs = [node for node in self.graph.nodes if node.op == "placeholder"]
        self._body = [node for node in self.graph.nodes if node.op != "placeholder"]
        self._boxed_args = {
            node: itemgetter(*node.args)
            for node in self._body
            if node.op == "call_module"
            and node.args
            and not node.kwargs
            and all(isinstance(arg, torch.fx.Node) for arg in node.args)
        }

    def run(self, *args, initial_env=None, enable_io_processing=True):
        if self._capture_coordinator is not None:
            self._capture_coordinator.check_input_buffer_streams()
        if initial_env is not None or len(args) != len(self._inputs):
            result = super().run(*args, initial_env=initial_env, enable_io_processing=enable_io_processing)
            self.env.clear()
            return result
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        if self._staged_input_indices or self._input_aliases:
            args = self._stage_inputs(args)
        # Lifted parameters need bindings, not a Python interpreter call each.
        self.env = dict(zip(self._inputs, args, strict=True))
        for node in self._body:
            result = self.env[node] = self.run_node(node)
            if node.op == "output":
                # Original graph outputs stayed live until the copy above, so
                # the shared pool could not reuse their storage during capture.
                self.env.clear()
                return self.graph.process_outputs(result) if enable_io_processing else result

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)
        return _clone_graph_outputs(result, self._clone_output_indices) if self._clone_output_indices else result

    def _stage_inputs(self, args):
        # Share one persistent buffer across every consumer, including aliases
        # returned by captured regions. Update it once before this step's work.
        for i, source in self._input_aliases.items():
            if _tensor_input_spec(args[i]) != _tensor_input_spec(args[source]):
                raise ValueError(f"ACLGraph shared input alias changed: input[{i}] and input[{source}]")
        if self._producer_input_indices is None:
            # Producer-owned buffers pass straight through. Region guards check
            # their addresses/layouts; no root staging or duplicate guard loop.
            producer_sources = {
                self._input_aliases.get(i, i)
                for i in self._staged_input_indices
                if getattr(args[i], "_dynamo_static_input_type", None) == "guarded"
            }
            owned_indices = {i for i in self._staged_input_indices if self._input_aliases.get(i, i) in producer_sources}
            self._producer_input_indices = owned_indices
            self._staged_input_indices = tuple(
                i for i in self._staged_input_indices if i not in self._producer_input_indices
            )
        values = list(args)
        destinations, sources = [], []
        for i in self._staged_input_indices:
            arg = args[i]
            preserve = self._preserve_storage_indices is None or i in self._preserve_storage_indices
            if i in self._input_aliases:
                source = self._input_aliases[i]
                values[i] = values[source]
                continue
            saved = self._input_buffers.get(i)
            if saved is None or (
                saved.shape,
                saved.stride(),
                saved.dtype,
                saved.device,
                saved.storage_offset(),
                saved.untyped_storage().nbytes() if preserve else None,
            ) != (
                arg.shape,
                arg.stride(),
                arg.dtype,
                arg.device,
                arg.storage_offset(),
                arg.untyped_storage().nbytes() if preserve else None,
            ):
                saved = self._input_buffers[i] = _clone_staging_input(arg, preserve_storage=preserve)
            else:
                destinations.append(_staging_source(saved, preserve_storage=preserve))
                sources.append(_staging_source(arg, preserve_storage=preserve))
            values[i] = saved
        if sources:
            torch._foreach_copy_(destinations, sources)
        return values

    def run_node(self, n):
        region = self.fetch_attr(n.target) if n.op == "call_module" else None
        wrapper = getattr(region, "_aclgraph_wrapper", None)
        if wrapper is not None:
            fetch = self._boxed_args.get(n)
            if fetch is not None:
                values = fetch(self.env)
                boxed = [values] if len(n.args) == 1 else list(values)
                del values
            else:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                assert not kwargs
                boxed = list(args)
                del args
            for value in self.user_to_last_uses.get(n, []):
                del self.env[value]
            return wrapper.call_boxed(boxed)
        result = super().run_node(n)
        for value in self.user_to_last_uses.get(n, []):
            del self.env[value]
        return result
