# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 MHC overrides backed by TileLang implementations.

V4.1 keeps a cross-layer pre-mix graph, so ``TilelangV41HcPre`` adapts the
model-side ``forward(x, pre_mix) -> (y, pre, post, comb)`` contract to the
TileKernels split/apply primitives (the freshly computed ``pre`` is returned
for the next sub-layer and the branches collapse with the incoming
``pre_mix``).  ``TilelangV41HcPost`` adapts ``forward(y, residual, post, comb)``
to the external TileKernels ``mhc_post`` operator.
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre
from torchtitan_npu.ops.tilelang import tilelang_mhc_post, tilelang_mhc_pre_v41


class TilelangV41HcPre(HcPre):
    """V4.1 HcPre backed by TileKernels Split/Apply and torch_npu Sinkhorn."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def _to_local(self, tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def forward(self, x_BLHcD: torch.Tensor, pre_mix_BLHc: torch.Tensor):
        if isinstance(x_BLHcD, DTensor):
            raise ValueError(
                "TileLang V4.1 HcPre does not support DTensor activations; use the validated spmd_types backend"
            )
        if x_BLHcD.device.type != "npu":
            raise ValueError(f"TileLang V4.1 HcPre requires NPU input, got {x_BLHcD.device}")
        if x_BLHcD.ndim != 4:
            raise ValueError(f"TileLang V4.1 HcPre expects x with shape [B,S,N,D], got {tuple(x_BLHcD.shape)}")
        if self.hc_mult != 4:
            raise ValueError(f"TileLang V4.1 HcPre currently supports only hc_mult=4, got {self.hc_mult}")
        if x_BLHcD.shape[-2] != self.hc_mult:
            raise ValueError(f"TileLang V4.1 HcPre expects N={self.hc_mult}, got x shape {tuple(x_BLHcD.shape)}")
        if x_BLHcD.shape[-1] % 64 != 0:
            raise ValueError(
                f"TileLang V4.1 HcPre requires a hidden size divisible by 64, got x shape {tuple(x_BLHcD.shape)}"
            )
        if x_BLHcD.dtype != torch.bfloat16:
            raise ValueError(f"TileLang V4.1 HcPre requires torch.bfloat16 input, got {x_BLHcD.dtype}")
        if not x_BLHcD.is_contiguous():
            raise ValueError("TileLang V4.1 HcPre requires contiguous input")
        batch, seq = x_BLHcD.shape[:2]
        if pre_mix_BLHc.ndim != 3:
            raise ValueError(f"TileLang V4.1 HcPre expects pre_mix with shape [B,S,N], got {tuple(pre_mix_BLHc.shape)}")
        if pre_mix_BLHc.shape != (batch, seq, self.hc_mult):
            raise ValueError(
                "TileLang V4.1 HcPre requires pre_mix shape [B,S,N] to match x [B,S,N,D], "
                f"got pre_mix={tuple(pre_mix_BLHc.shape)} for x={tuple(x_BLHcD.shape)}"
            )
        if pre_mix_BLHc.dtype != torch.float32:
            raise ValueError(f"TileLang V4.1 HcPre requires float32 pre_mix, got {pre_mix_BLHc.dtype}")
        if not pre_mix_BLHc.is_contiguous():
            raise ValueError("TileLang V4.1 HcPre requires contiguous pre_mix")

        return tilelang_mhc_pre_v41(
            x_BLHcD,
            pre_mix_BLHc,
            self._to_local(self.hc_fn).float(),
            self._to_local(self.hc_scale).float(),
            self._to_local(self.hc_base).float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )


class TilelangV41HcPost(HcPost):
    """Run TileKernels ``mhc_post`` while preserving the V4.1 model contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def _prepare_inputs(self, y_bld, residual_bl_hcd, post_bl_hc, comb_bl_hc_hc):
        y_was_2d = y_bld.ndim == 2
        residual_was_3d = residual_bl_hcd.ndim == 3
        post_was_2d = post_bl_hc.ndim == 2
        comb_was_3d = comb_bl_hc_hc.ndim == 3
        if residual_was_3d:
            residual_bl_hcd = residual_bl_hcd.unsqueeze(0)
        if post_was_2d:
            post_bl_hc = post_bl_hc.unsqueeze(0)
        if comb_was_3d:
            comb_bl_hc_hc = comb_bl_hc_hc.unsqueeze(0)
        if y_was_2d:
            y_bld = y_bld.reshape(residual_bl_hcd.shape[0], residual_bl_hcd.shape[1], -1)
        return y_bld, residual_bl_hcd, post_bl_hc, comb_bl_hc_hc, residual_was_3d

    def _validate_inputs(self, y_bld, residual_bl_hcd, post_bl_hc, comb_bl_hc_hc) -> None:
        if y_bld.device.type != "npu":
            raise ValueError(f"TileLang V4.1 HcPost requires NPU inputs, got {y_bld.device}")
        if y_bld.ndim != 3:
            raise ValueError(f"TileLang V4.1 HcPost expects y with shape [B,S,D], got {tuple(y_bld.shape)}")
        if residual_bl_hcd.ndim != 4:
            raise ValueError(
                f"TileLang V4.1 HcPost expects residual with shape [B,S,N,D], got {tuple(residual_bl_hcd.shape)}"
            )
        batch, seq, hidden = y_bld.shape
        if residual_bl_hcd.shape[:2] != (batch, seq) or residual_bl_hcd.shape[-1] != hidden:
            raise ValueError(
                "TileLang V4.1 HcPost requires residual shape [B,S,N,D] to match y [B,S,D], "
                f"got y={tuple(y_bld.shape)}, residual={tuple(residual_bl_hcd.shape)}"
            )
        num_streams = residual_bl_hcd.shape[2]
        if post_bl_hc.shape != (batch, seq, num_streams):
            raise ValueError(
                "TileLang V4.1 HcPost requires post shape [B,S,N], "
                f"got post={tuple(post_bl_hc.shape)} for y={tuple(y_bld.shape)} and N={num_streams}"
            )
        if comb_bl_hc_hc.shape != (batch, seq, num_streams, num_streams):
            raise ValueError(
                "TileLang V4.1 HcPost requires comb shape [B,S,N,N], "
                f"got comb={tuple(comb_bl_hc_hc.shape)} for N={num_streams}"
            )
        if y_bld.dtype != torch.bfloat16 or residual_bl_hcd.dtype != torch.bfloat16:
            raise ValueError(
                "TileLang V4.1 HcPost requires y and residual to use torch.bfloat16, "
                f"got y={y_bld.dtype}, residual={residual_bl_hcd.dtype}"
            )
        if post_bl_hc.dtype != torch.float32 or comb_bl_hc_hc.dtype != torch.float32:
            raise ValueError(
                "TileLang V4.1 HcPost requires post and comb to use torch.float32, "
                f"got post={post_bl_hc.dtype}, comb={comb_bl_hc_hc.dtype}"
            )
        if any(not tensor.is_contiguous() for tensor in (y_bld, residual_bl_hcd, post_bl_hc, comb_bl_hc_hc)):
            raise ValueError("TileLang V4.1 HcPost requires contiguous y, residual, post and comb inputs")

    def forward(
        self,
        y_BLD: torch.Tensor,
        residual_BLHcD: torch.Tensor,
        post_BLHc: torch.Tensor,
        comb_BLHcHc: torch.Tensor,
    ) -> torch.Tensor:
        y_BLD, residual_BLHcD, post_BLHc, comb_BLHcHc, residual_was_3d = self._prepare_inputs(
            y_BLD, residual_BLHcD, post_BLHc, comb_BLHcHc
        )
        self._validate_inputs(y_BLD, residual_BLHcD, post_BLHc, comb_BLHcHc)
        out = tilelang_mhc_post(y_BLD, residual_BLHcD, post_BLHc.unsqueeze(-1), comb_BLHcHc)
        if residual_was_3d:
            out = out.squeeze(0)
        return out


@override(
    target=HcPre.Config,
    exact=True,
    description="V4.1 single-pass mHC pre backed by TileKernels Split/Apply and torch_npu Sinkhorn",
)
def tilelang_hc_pre(cfg: HcPre.Config) -> TilelangV41HcPre.Config:
    return derive(cfg, TilelangV41HcPre.Config)


@override(
    target=HcPost.Config,
    exact=True,
    description="V4.1 mHC post backed by the external TileKernels mhc_post operator",
)
def tilelang_hc_post(cfg: HcPost.Config) -> TilelangV41HcPost.Config:
    return derive(cfg, TilelangV41HcPost.Config)
