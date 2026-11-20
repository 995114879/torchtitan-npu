# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter contracts for the V4.1 mHC fused modules.

Selection/isolation via the real debug model; arithmetic contracts run on
CPU with the fused kernels replaced by reference implementations.  The
kernel mocks prove the adapter feeds the kernels the documented operands
and keeps the autograd chain connected; NPU kernel execution is validated
on the target hardware (see the verification records).
"""

import pytest
import torch
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre

MHC_POST_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post"
MHC_SINKHORN_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn"
MHC_PREMIX_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_premix"


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


class TestMHcAdapters:
    def test_override_selects_and_preserves_config(self):
        cfg = model_registry("deepseek_v4_1_debugmodel").model
        reference = model_registry("deepseek_v4_1_debugmodel").model
        apply_overrides(
            OverrideConfig(imports=[MHC_POST_OVERRIDE, MHC_SINKHORN_OVERRIDE]),
            cfg,
        )
        from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPost, AscV41HcPre

        for layer, ref_layer in zip(cfg.layers[:6], reference.layers[:6], strict=True):
            assert isinstance(layer.hc_post, AscV41HcPost.Config)
            assert isinstance(layer.hc_attn_pre, AscV41HcPre.Config)
            # the fused configs are pure replacements: every declared field
            # keeps the reference value
            for field in ("hc_mult", "dim", "sinkhorn_iters", "hc_eps", "norm_eps"):
                assert getattr(layer.hc_attn_pre, field) == getattr(ref_layer.hc_attn_pre, field)

    def test_hc_post_matches_reference_arithmetic(self, monkeypatch):
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPost.Config().build()
        fused_module = fused.AscV41HcPost.Config().build()
        # Block-level contract: ``y`` is the collapsed single stream [B, T, D];
        # ``residual`` is the multi-stream [B, T, n, D].
        B, T, D, n = 2, 5, 8, 4
        y = torch.randn(B, T, D, dtype=torch.bfloat16)
        residual = torch.randn(B, T, n, D, dtype=torch.bfloat16)
        post = torch.rand(B, T, n, dtype=torch.float32)
        comb = torch.rand(B, T, n, n, dtype=torch.float32)

        def mhc_post(residual, comb, y, post):
            out = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.sum(
                comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
            )
            return out.type_as(y)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_post", mhc_post)

        # Forward and backward both flow through the mocked kernel unchanged:
        # same values, same gradients on every operand.
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in (y, residual, post, comb)]
        fused_inputs = [t.detach().clone().requires_grad_(True) for t in (y, residual, post, comb)]
        expected = reference(*ref_inputs)
        actual = fused_module(*fused_inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.randn_like(expected)
        expected.backward(grad)
        actual.backward(grad)
        for ref_t, fused_t, name in zip(ref_inputs, fused_inputs, ("y", "residual", "post", "comb"), strict=True):
            assert ref_t.grad is not None and fused_t.grad is not None, name
            torch.testing.assert_close(fused_t.grad, ref_t.grad, rtol=1e-5, atol=1e-6)

    def test_hc_post_unfolds_the_batch_folded_stream(self, monkeypatch):
        """Sequence-parallel linears return the collapsed stream as ``[T, D]``;
        the reference broadcasts that silently, the kernel needs ``[B, T, D]``."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        fused_module = fused.AscV41HcPost.Config().build()
        B, T, D, n = 1, 5, 8, 4
        y_folded = torch.randn(T, D, dtype=torch.bfloat16)
        residual = torch.randn(B, T, n, D, dtype=torch.bfloat16)
        post = torch.rand(B, T, n, dtype=torch.float32)
        comb = torch.rand(B, T, n, n, dtype=torch.float32)
        expected = HcPost.Config().build()(y_folded.unsqueeze(0), residual, post, comb)

        received = []

        def mhc_post(x, h_res, h_out, h_post):
            received.append(tuple(t.dim() for t in (x, h_res, h_out, h_post)))
            out = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + torch.sum(
                h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=2
            )
            return out.type_as(h_out)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_post", mhc_post)
        actual = fused_module(y_folded, residual, post, comb)
        assert received[0] == (4, 4, 3, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_split_sinkhorn_matches_reference_and_feeds_kernel(self, monkeypatch):
        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        module = fused.AscV41HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module.load_state_dict(reference.state_dict())
        n = module.hc_mult

        calls = []

        def sinkhorn(comb, *, eps, num_iters, out_flag):
            # Differentiable CPU stand-in mirroring the reference's balancing:
            # proving the adapter connects the graph, not the native kernel.
            calls.append((eps, num_iters, out_flag))
            row_max = comb.max(dim=-1, keepdim=True).values
            out = torch.exp(comb - row_max)
            out = out / out.sum(dim=-1, keepdim=True) + eps
            out = out / (out.sum(dim=-2, keepdim=True) + eps)
            for _ in range(num_iters - 1):
                out = out / (out.sum(dim=-1, keepdim=True) + eps)
                out = out / (out.sum(dim=-2, keepdim=True) + eps)
            return out, None, None

        monkeypatch.setattr(fused.torch_npu, "npu_mhc_sinkhorn", sinkhorn)

        B, T = 2, 3
        mixes = torch.randn(B, T, 2 * n + n * n, requires_grad=True)
        ref_mixes = mixes.detach().clone().requires_grad_(True)
        ref_pre, ref_post, ref_comb = reference._split_sinkhorn(ref_mixes)
        pre, post, comb = module._split_sinkhorn(mixes)

        # pre/post keep the reference arithmetic exactly; comb matches the
        # stand-in's balancing (same math as the reference), and the kernel
        # received the documented iteration knobs.
        torch.testing.assert_close(pre, ref_pre, rtol=0, atol=0)
        torch.testing.assert_close(post, ref_post, rtol=0, atol=0)
        torch.testing.assert_close(comb, ref_comb, rtol=1e-6, atol=1e-7)
        assert calls[0] == (module.hc_eps, module.sinkhorn_iters, 1)

        # the gradient connection: identical backward through both paths
        (pre.sum() + post.sum() + comb.sum()).backward()
        (ref_pre.sum() + ref_post.sum() + ref_comb.sum()).backward()
        torch.testing.assert_close(mixes.grad, ref_mixes.grad, rtol=1e-5, atol=1e-7)

    def test_premix_override_selects_premix_adapter(self):
        cfg = model_registry("deepseek_v4_1_debugmodel").model
        apply_overrides(OverrideConfig(imports=[MHC_PREMIX_OVERRIDE]), cfg)
        from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPrePremix

        for layer in cfg.layers[:6]:
            assert isinstance(layer.hc_attn_pre, AscV41HcPrePremix.Config)
            assert isinstance(layer.hc_ffn_pre, AscV41HcPrePremix.Config)

    def test_premix_adapter_uses_the_kernel_in_inference(self, monkeypatch):
        """Inference has no gradient to preserve, so the kernel collapses."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module = fused.AscV41HcPrePremix.Config(dim=8).build()
        module.load_state_dict(reference.state_dict())

        batch, seq, dim, n = 2, 3, 8, module.hc_mult
        x = torch.randn(batch, seq, n, dim, dtype=torch.bfloat16)
        pre_mix = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
        seen = {}

        def premix_op(x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag):
            seen["premix"] = premix
            seen["out_flag"] = out_flag
            pre = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
            post = torch.ones(batch, seq, n, dtype=torch.float32)
            comb = torch.full((batch, seq, n, n), 1.0 / n, dtype=torch.float32)
            hin = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
            empty = torch.empty(0)
            return (hin, post, comb.reshape(batch, seq, n * n), pre, empty, empty, empty, empty)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_pre_sinkhorn_premix", premix_op)

        with torch.no_grad():
            y, pre, post, comb = module(x, pre_mix)
        assert seen["premix"] is pre_mix
        assert seen["out_flag"] is True
        torch.testing.assert_close(y, module.collapse(x, pre_mix), rtol=0, atol=0)
        assert comb.shape == (batch, seq, n, n)

    def test_premix_adapter_drives_the_kernel_in_training(self, monkeypatch):
        """The kernel runs under autograd, keeping ``hPre`` valid for the next layer."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module = fused.AscV41HcPrePremix.Config(dim=8).build()
        module.load_state_dict(reference.state_dict())

        batch, seq, dim, n = 2, 3, 8, module.hc_mult
        x = torch.randn(batch, seq, n, dim, dtype=torch.bfloat16)
        pre_mix = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
        seen = {}

        def premix_op(x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, need_backward):
            seen["premix"] = premix
            seen["need_backward"] = need_backward
            pre = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
            post = torch.ones(batch, seq, n, dtype=torch.float32)
            comb = torch.full((batch, seq, n, n), 1.0 / n, dtype=torch.float32)
            hin = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
            empty = torch.empty(0)
            return (hin, post, comb.reshape(batch, seq, n * n), pre, empty, empty, empty, empty)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_pre_sinkhorn_premix", premix_op)

        y, pre, post, comb = module(x, pre_mix)

        assert seen["premix"] is pre_mix
        assert seen["need_backward"] is True
        torch.testing.assert_close(y, module.collapse(x, pre_mix), rtol=0, atol=0)
        assert comb.shape == (batch, seq, n, n)

    # --- mhc_pre_sinkhorn_premix backward ---------------------------------
    #
    # The forward tests above replace the op with a plain Python callable, so
    # ``register_autograd`` never runs and ``_mhc_pre_sinkhorn_premix_backward``
    # had no coverage at all.  These call it directly with a fake ctx and a
    # recording kernel: they pin the operand order, the gradient-slot
    # alignment, and every "missing gradient" branch.

    @staticmethod
    def _premix_backward_fixture(*, with_premix: bool):
        from types import SimpleNamespace

        batch, seq, n, dim = 2, 3, 4, 8
        mix = (2 + n) * n
        premix = torch.randn(batch, seq, n) if with_premix else None
        saved = (
            torch.randn(batch, seq, n, dim),  # x
            torch.randn(mix, n * dim),  # phi
            torch.randn(3),  # alpha
            torch.randn(mix),  # bias
            premix,
            torch.randn(batch, seq, n),  # h_pre
            torch.randn(batch, seq, n),  # hc_before_norm
            torch.randn(batch, seq),  # inv_rms
            torch.randn(batch, seq),  # sum_out
            torch.randn(batch, seq),  # norm_out
        )
        return SimpleNamespace(saved_tensors=saved, hc_eps=1e-6), saved

    @staticmethod
    def _install_recording_backward(monkeypatch, *, empty_grad_premix: bool = False):
        import torch as torch_mod

        recorded = {}

        def premix_backward(*args):
            recorded["args"] = args
            premix_arg = args[12]
            if premix_arg is not None and not empty_grad_premix:
                grad_premix = torch.full_like(premix_arg, 0.5)
            else:
                grad_premix = torch.empty(0)
            return (
                torch.zeros_like(args[3]),  # grad_x
                torch.zeros_like(args[4]),  # grad_phi
                torch.zeros_like(args[5]),  # grad_alpha
                torch.zeros_like(args[6]),  # grad_bias
                grad_premix,
            )

        # ``raising=False``: the backward op only exists once the CANN package
        # ships the experimental kernel, and this stub replaces it either way.
        monkeypatch.setattr(
            torch_mod.ops.cann_ops_transformer,
            "mhc_pre_sinkhorn_premix_backward",
            premix_backward,
            raising=False,
        )
        return recorded

    def test_premix_backward_aligns_operands_and_gradient_slots(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, saved = self._premix_backward_fixture(with_premix=True)
        recorded = self._install_recording_backward(monkeypatch)
        grad_h_pre = torch.randn(2, 3, 4)

        grads = mhc_ops._mhc_pre_sinkhorn_premix_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), grad_h_pre, None
        )

        args = recorded["args"]
        # 14 tensor operands + hc_eps, in the order the kernel documents.
        assert len(args) == 15
        assert args[12] is saved[4]  # premix
        assert args[13] is grad_h_pre  # gradPre
        assert args[14] == ctx.hc_eps  # hcEps
        # 10 autograd slots: 4 parameter grads, premix, then 5 attribute slots.
        assert len(grads) == 10
        assert grads[5:] == (None,) * 5
        torch.testing.assert_close(grads[4], torch.full_like(saved[4], 0.5))

    def test_premix_backward_zero_fills_missing_output_gradients(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, saved = self._premix_backward_fixture(with_premix=True)
        recorded = self._install_recording_backward(monkeypatch)

        mhc_ops._mhc_pre_sinkhorn_premix_backward(ctx, None, torch.randn(2, 3, 4), torch.randn(2, 3, 16), None, None)

        args = recorded["args"]
        x = saved[0]
        # hin took no upstream gradient: it is zero-filled, not skipped.
        assert args[0].shape == (x.shape[0], x.shape[1], x.shape[-1])
        assert args[0].dtype == x.dtype
        assert torch.count_nonzero(args[0]).item() == 0
        # pre was unused downstream: gradPre is zero-filled so the kernel still
        # receives a tensor and the parameters keep their hIn-path gradient.
        torch.testing.assert_close(args[13], torch.zeros_like(saved[4]))

    def test_premix_backward_drops_absent_premix(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, _ = self._premix_backward_fixture(with_premix=False)
        recorded = self._install_recording_backward(monkeypatch)

        grads = mhc_ops._mhc_pre_sinkhorn_premix_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), None, None
        )

        assert recorded["args"][12] is None
        assert recorded["args"][13] is None
        assert grads[4] is None

    def test_premix_backward_drops_empty_kernel_gradient(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, _ = self._premix_backward_fixture(with_premix=True)
        self._install_recording_backward(monkeypatch, empty_grad_premix=True)

        grads = mhc_ops._mhc_pre_sinkhorn_premix_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), torch.randn(2, 3, 4), None
        )

        assert grads[4] is None
