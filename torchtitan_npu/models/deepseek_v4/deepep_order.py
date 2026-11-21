# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass, get_custom_graph_passes, get_hash_for_files
from torch._inductor.fx_passes.control_dependencies import preserve_node_ordering
from torch.utils._ordered_set import OrderedSet


class _CPBeforeDeepEP(CustomGraphPass):
    def __init__(self, group_name: str):
        self.group_name = group_name

    def uuid(self):
        return get_hash_for_files((__file__,), extra=self.group_name)

    def __call__(self, graph: torch.fx.Graph) -> None:
        dispatch = getattr(torch.ops.deepep, "dispatch", None)
        if dispatch is None:
            return
        pending = OrderedSet()
        dependencies = {}
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops._c10d_functional.all_to_all_single.default:
                group = node.args[3] if len(node.args) > 3 else node.kwargs.get("group_name")
                if group == self.group_name:
                    pending.add(node)
            elif node.target == dispatch.default and pending:
                # Depend on launches, so send-only ranks can still overlap their waits.
                dependencies[node] = pending
                pending = OrderedSet()
        preserve_node_ordering(graph, dependencies)


def install_cp_deepep_order(group_name: str) -> None:
    installed = get_custom_graph_passes(config.post_grad_custom_pre_pass)
    if not any(isinstance(p, _CPBeforeDeepEP) and p.group_name == group_name for p in installed):
        config.post_grad_custom_pre_pass = (*installed, _CPBeforeDeepEP(group_name))
