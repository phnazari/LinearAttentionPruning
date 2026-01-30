# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

class IndependentNorm(nn.Module):
    """
    Applies separate normalization layers to each head.
    Used when each head prunes different dimension indices.
    """
    def __init__(self, head_v_dim, eps=1e-6, num_heads=1, use_gate=False):
        super().__init__()
        self.head_v_dim = head_v_dim
        self.num_heads = num_heads
        self.use_gate = use_gate
        
        norms = []
        for _ in range(num_heads):
            if use_gate:
                norms.append(FusedRMSNormGated(head_v_dim, eps=eps))
            else:
                norms.append(RMSNorm(head_v_dim, eps=eps))
        self.norms = nn.ModuleList(norms)
        
    def forward(self, x, g=None):
        # x: [b, t, h, d]
        # g: [b, t, h, d] (optional gate)
        outputs = []
        for h in range(len(self.norms)):
            if g is not None:
                outputs.append(self.norms[h](x[:, :, h], g[:, :, h]))
            else:
                outputs.append(self.norms[h](x[:, :, h]))
        return torch.stack(outputs, dim=2)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class SharedKernelConv1d(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        num_heads: int = 1,
        per_head: bool = False,
        shift_only_conv: bool = False,
        conv_scaled_shift: bool | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.per_head = per_head
        self.shift_only_conv = shift_only_conv
        self.conv_scaled_shift = conv_scaled_shift

        n_kernels = num_heads if per_head else 1

        if shift_only_conv:
            if conv_scaled_shift:
                # Learnable scalar r
                self.r = nn.Parameter(torch.empty(n_kernels, 1, 1))
            else:
                # Hard-coded constant 1.0 (not a parameter)
                self.register_buffer("r", torch.ones(n_kernels, 1, 1))
        else:
            # Standard full kernel
            self.shared_kernel = nn.Parameter(torch.empty(n_kernels, 1, kernel_size))

        self.conv = ShortConvolution(
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
        )

        if hasattr(self.conv, "weight"):
            del self.conv.weight

        self.reset_parameters()

    def reset_parameters(self):
        if self.shift_only_conv:
            if self.conv_scaled_shift:
                nn.init.ones_(self.r)
        else:
            nn.init.kaiming_uniform_(self.shared_kernel, a=math.sqrt(5))

    def resize(self, new_hidden_size: int, new_num_heads: int = None, device=None, dtype=None):
        self.hidden_size = new_hidden_size
        if new_num_heads is not None:
            self.num_heads = new_num_heads
        
        # Update head_dim for the forward pass expansion
        if self.per_head:
            assert new_hidden_size % self.num_heads == 0
            self.head_dim = new_hidden_size // self.num_heads
        else:
            self.head_dim = new_hidden_size

        self.conv = ShortConvolution(
            hidden_size=new_hidden_size,
            kernel_size=self.kernel_size,
            bias=self.conv.bias is not None,
            activation=self.conv.activation,
        )
        if device is not None or dtype is not None:
            self.conv = self.conv.to(device=device, dtype=dtype)

        if hasattr(self.conv, "weight"):
            del self.conv.weight

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.shift_only_conv:
            weight = F.pad(self.r, (self.kernel_size - 1, 0), value=0.0)
        else:
            weight = self.shared_kernel

        if self.per_head:
            head_dim = self.hidden_size // self.num_heads
            expanded = weight.expand(self.num_heads, head_dim, -1)
            self.conv.weight = expanded.reshape(self.hidden_size, 1, self.kernel_size)
        else:
            self.conv.weight = weight.expand(self.hidden_size, -1, -1)

        return self.conv(x, **kwargs)


class DeltaNet2(nn.Module):
    r"""
    The layer implementaion for [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484).  # noqa:
    DeltaNet was originally proposed in [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174). # noqa

    Args:
        mode (str, Optional):
            Which DeltaNet kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 1.0.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `False`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        qk_activation (str, Optional):
            The activation function for the query and key. Default: `silu`.
        qk_norm (str, Optional):
            The normalization method for the query and key. Default: `l2`.
        per_head_conv (bool, Optional):
            Whether to use one convolution kernel per head instead of one kernel shared across
            all heads. If True, each head learns its own kernel. If False, all heads share
            a single kernel. Only applies when use_short_conv=True. Default: `False`.
    """

    def __init__(
        self,
        mode: str = "chunk",
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        use_output_norm: bool = True,
        per_head_norm: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        per_head_conv: bool = False,
        share_kv_conv: bool = False,
        shift_only_conv: bool = False,
        **kwargs,
    ) -> DeltaNet2:
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.per_head_conv = per_head_conv
        self.share_kv_conv = share_kv_conv

        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.per_head_norm = per_head_norm
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        if mode == "fused_chunk":
            raise NotImplementedError(
                "fused_chunk_delta_rule is now deprecated. Please use `chunk_delta_rule` instead."
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        assert (
            self.key_dim % num_heads == 0
        ), f"key dim must be divisible by num_heads of {num_heads}"
        assert (
            self.value_dim % num_heads == 0
        ), f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            # Use shared kernel convolutions - kernel weights shared across channels
            # If per_head_conv=True, each head gets its own kernel
            # If per_head_conv=False, all heads share a single kernel
            self.q_conv1d = SharedKernelConv1d(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if qk_activation == "silu" else None,
                num_heads=num_heads,
                per_head=per_head_conv,
                shift_only_conv=shift_only_conv
            )
            self.k_conv1d = SharedKernelConv1d(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if qk_activation == "silu" else None,
                num_heads=num_heads,
                per_head=per_head_conv,
                shift_only_conv=shift_only_conv
            )

            if not self.share_kv_conv:
                self.v_conv1d = SharedKernelConv1d(
                    hidden_size=self.value_dim,
                    kernel_size=conv_size,
                    bias=conv_bias,
                    activation="silu",
                    num_heads=num_heads,
                    per_head=per_head_conv,
                    shift_only_conv=shift_only_conv
                )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if self.per_head_norm:
            self.o_norm = IndependentNorm(self.head_v_dim, eps=norm_eps, num_heads=num_heads, use_gate=use_gate)
        elif use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = "fused_recurrent" if q_len <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            if self.share_kv_conv:
                v, conv_state_v = self.k_conv1d(
                    x=self.v_proj(hidden_states),
                    cache=conv_state_v,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
            else:
                v, conv_state_v = self.v_conv1d(
                    x=self.v_proj(hidden_states),
                    cache=conv_state_v,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError

        if self.qk_norm == "sum":
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=(self.qk_norm == "l2"),
            )
        elif mode == "chunk":
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=(self.qk_norm == "l2"),
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(
                    (conv_state_q, conv_state_k, conv_state_v)
                    if self.use_short_conv
                    else None
                ),
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
