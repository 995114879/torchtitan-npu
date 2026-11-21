# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the launch dependency on a send-only CP rank without requiring DeepEP."""

import torch
from torch._inductor.fx_passes.control_dependencies import control_deps

from torchtitan_npu.models.deepseek_v4.deepep_order import _CPBeforeDeepEP


def test_dispatch_depends_on_cp_launch_even_without_a_received_tensor(monkeypatch):
    # Stand in for the A5-only dispatch overload while exercising the real Inductor graph rewrite.
    monkeypatch.setattr(torch.ops.deepep, "dispatch", torch.ops.aten.neg, raising=False)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    exchange = torch.ops._c10d_functional.all_to_all_single.default
    cp = graph.call_function(exchange, (x, [0, 0], [0, 3], "cp"))
    other = graph.call_function(exchange, (x, [3, 0], [0, 3], "other"))
    dispatch = graph.call_function(torch.ops.deepep.dispatch.default, (x,))
    wait = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (cp,))
    graph.output((dispatch, other, wait))
    module = torch.fx.GraphModule({}, graph)

    _CPBeforeDeepEP("cp")(module.graph)
    module.graph.lint()

    (ordered_dispatch,) = module.graph.find_nodes(op="call_function", target=control_deps)
    assert ordered_dispatch.args[0] == (cp,)
    assert next(iter(ordered_dispatch.users)).op == "output"
