import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =============================================================================
# Helpers
# =============================================================================


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


def build_norm_layer(dim, norm_layer, in_format='channels_last',
                     out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class StemLayer(nn.Module):
    """Stem layer of InternImage."""

    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# WaveOperator: damped-wave propagation on the rfft2 grid.
#
# Each block projects its input to an initial field u0, an initial velocity
# v0 and a gate z, moves u0 / v0 to the frequency domain with rfft2 and lets
# every Fourier mode evolve for a learnable time t:
#
#   damping (decay) term:  R_damp = exp(-alpha * lam^p * t)
#   oscillation term:      U_wave = e^{-alpha lam^p t / 2} [ U0 cos(w t)
#                                   + (V0 + alpha lam^p / 2 * U0) sin(w t) / w ]
#                          with w = sqrt(lam^p)
#   output:                U(t) = R_damp * U0 + wave_weight * (U_wave - R_damp * U0)
#
#   * alpha (H, Wr, K, D): non-negative damping = softplus(stage-shared
#     per-position field alpha_embed + per-block per-channel bias alpha_ch)
#   * lam (H, Wr): rfft2 Laplacian eigenvalues (2 pi f)^2, non-negative
#   * p (K, D): learnable spectral exponent, so lam^p is a fractional power
#   * t (K, D): learnable propagation time per head and channel
#   * wave_weight (K,): learnable mixing weight of the oscillation term,
#     initialised to 0 so training starts from the damping (decay) term
#
# U(t) is brought back with irfft2, normalised, gated by SiLU(z) and projected
# by a linear layer.
#
# Input / output contract:
#     forward(x, alpha_raw) -> tensor, both (B, C, H, W)
# alpha_raw is owned by the WaveFormer trunk (self.alpha_embed[stage]) and
# passed in via AdditionalInputSequential.
# =============================================================================


def _fft2d(inp):
    """2D real FFT over (H, W): the paper's F, returning (B, C, H, W//2+1) complex."""
    work_dtype = inp.dtype if inp.dtype in (torch.float32, torch.float64) else torch.float32
    return torch.fft.rfft2(inp.to(work_dtype), dim=(-2, -1), norm='ortho')


def _ifft2d(spec, shape):
    """Inverse of :func:`_fft2d`; ``shape`` is the (H, W) to reconstruct."""
    return torch.fft.irfft2(spec, s=shape, dim=(-2, -1), norm='ortho')


def _fourier_lambda_map(H, W, device=torch.device("cpu"), dtype=torch.float):
    """Periodic (FFT) Laplacian eigenvalues on the rfft2 grid: (2*pi*f)^2, shape (H, W//2+1)."""
    freq_h = 2 * math.pi * torch.fft.fftfreq(H, device=device, dtype=dtype).abs()
    freq_w = 2 * math.pi * torch.fft.rfftfreq(W, device=device, dtype=dtype)
    return freq_h.view(-1, 1) ** 2 + freq_w.view(1, -1) ** 2


def _inv_softplus(y):
    """x such that softplus(x) == y, for y > 0."""
    return math.log(math.expm1(y))


class WaveOperator(nn.Module):
    r"""Wave propagation operator.

    For every rfft2 mode the operator evolves the projected field for a
    learnable time ``t``. The response combines the damping (decay) term

        R_damp = exp(-alpha * lam^p * t)

    with the damped oscillation term

        U_wave = e^{-alpha lam^p t / 2} [ U0 cos(w t) + (V0 + alpha lam^p / 2 U0) sin(w t) / w ],
        w = sqrt(lam^p),

    mixed by the per-head weight ``wave_weight`` (init 0):

        U(t) = R_damp U0 + wave_weight (U_wave - R_damp U0).

    Parameterisation:
    * ``alpha_raw`` (H, Wr, K) is passed in from the outside
      (self.alpha_embed[stage] on the WaveFormer trunk). Combined with the
      per-block per-channel bias ``alpha_ch`` (C,) and made non-negative via
      softplus + clamp at ALPHA_MIN.
    * ``t_raw`` (K,) + ``t_ch`` (C,): propagation time, non-negative via
      softplus. ``t_raw`` is initialised on a log-uniform grid so heads start
      at different time scales.
    * ``p_raw`` (K,) + ``p_ch`` (C,): spectral exponent p of lam^p, positive
      via softplus, initialised to p = 1.
    * ``wave_weight`` (K,): mixing weight of the oscillation term, init 0.
    * Linear head emits 3*C outputs (u0, v0, z).
    """

    # Lower bound on the effective damping alpha. alpha -> 0 makes the
    # damping term exp(-alpha * lam * t) an identity, which is numerically
    # harmless, so the bound is kept small: at 1e-4 the raw space stays
    # usable down to softplus^{-1}(1e-4) = -9.2 with gradients alive, so
    # channels that want a near-identity response never sit on the clamp
    # (where clamp() kills the gradient), and deep blocks with large t can
    # still keep high frequencies.
    ALPHA_MIN = 1e-4

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96,
                 propagation_time=1.0, num_heads=4, **kwargs):
        super().__init__()
        self.res = res
        self.hidden_dim = hidden_dim
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        # 3*C: u0 (initial field), v0 (initial velocity for the oscillation
        # term), z (gate).
        self.linear = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # kept only as a bookkeeping flag; forward does not depend on it
        self.infer_mode = infer_mode
        if propagation_time <= 0:
            raise ValueError(f"propagation_time must be > 0, got {propagation_time}")

        # t: (K,), one propagation time per head. Log-uniform init around
        # ``propagation_time`` so heads start at different time scales
        # ((0.71, 1.41) at K=2, (0.71, 0.89, 1.12, 1.41) at K=4, etc.).
        # Identical init would leave heads gradient-symmetric.
        if num_heads == 1:
            t_targets = torch.tensor([propagation_time], dtype=torch.float32)
        else:
            SPREAD = math.sqrt(2.0)
            log_lo = math.log(propagation_time / SPREAD)
            log_hi = math.log(propagation_time * SPREAD)
            t_targets = torch.exp(torch.linspace(log_lo, log_hi, num_heads))
        t_raw_init = torch.tensor(
            [_inv_softplus(float(t)) for t in t_targets], dtype=torch.float32)
        self.t_raw = nn.Parameter(t_raw_init)

        # t_ch: per-block, per-channel bias on the propagation time,
        #     t(k, d) = softplus(t_raw(k) + t_ch(k, d))
        # alpha_ch shifts alpha additively (a constant in alpha space, scaled
        # only by lam), so it cannot rescale a channel's learned spatial
        # damping pattern. t_ch gives each channel a multiplicative gain on
        # the whole exponent, i.e. an affine reparameterisation (shift via
        # alpha_ch, scale via t_ch) of the shared spectral field.
        # 1-D shape so ``set_weight_decay`` excludes it from weight decay
        # automatically. Init: trunc_normal_(std=0.02); near t_raw = 0.54
        # (t = 1) softplus' ~= 0.63, so the initial t perturbation is O(0.01).
        self.t_ch = nn.Parameter(torch.empty(hidden_dim))
        trunc_normal_(self.t_ch, mean=0.0, std=0.02,
                      a=-0.08, b=0.08)  # 4-sigma bounds

        # alpha_raw is stage-shared: owned by WaveFormer.alpha_embed and
        # passed through forward(x, alpha_raw). See WaveFormer.__init__ for
        # the init.
        #
        # alpha_ch: per-block, per-channel bias added to alpha_raw before the
        # softplus. It lets each channel within a stage learn its own damping
        # profile (a global offset from the stage-shared per-position map):
        #     alpha(x, y, d) = softplus(alpha_raw(x, y, 0) + alpha_ch(d))
        # 1-D shape so ``set_weight_decay`` excludes it from weight decay
        # automatically (no need to touch no_weight_decay_keywords).
        # Init: trunc_normal_(std=0.02) so channel symmetry is broken from
        # step 0 while |alpha_ch| ~ 0.02 keeps the effective damping close to
        # the stage-shared init (softplus is locally linear near
        # alpha_raw = -2.97, so 0.02 in raw ~= 0.02 * softplus'(-2.97)
        # ~= 0.001 in the effective alpha space -- essentially a nudge).
        self.alpha_ch = nn.Parameter(torch.empty(hidden_dim))
        trunc_normal_(self.alpha_ch, mean=0.0, std=0.02,
                      a=-0.08, b=0.08)  # 4-sigma bounds

        # p_raw: per-block learnable spectral exponent,
        #     R_damp = exp(-alpha * lam^p * t)
        # A hard-wired p = 1 forces the stage-shared alpha to spend its
        # per-position budget approximating a fractional law alpha(lam) ~
        # lam^q; a learnable p hands this radial shape to a dedicated scalar
        # and frees alpha to model per-position + per-channel deviations.
        # p = softplus(p_raw + p_ch) > 0, init p = 1 (p_raw =
        # softplus^{-1}(1) = 0.5413). 1-D shape (K,) -- laid out like t_raw,
        # and excluded from weight decay via the shape==1 rule.
        self.p_raw = nn.Parameter(
            torch.full((num_heads,), _inv_softplus(1.0), dtype=torch.float32))

        # p_ch: per-block, per-channel bias on the spectral exponent,
        # completing the per-channel shift/scale/shape trio:
        #     p(k, d)  = softplus(p_raw(k) + p_ch(k, d))
        #     R_damp   = exp(-softplus(alpha_raw + alpha_ch) * lam^p * t)
        # alpha_ch shifts alpha additively, t_ch scales the whole exponent per
        # channel; p_ch lets each channel choose its own decay shape:
        # p < 1 = heavy-tailed / long-range (slower decay growth with
        # frequency), p > 1 = sharper low-pass.
        # 1-D shape so ``set_weight_decay`` excludes it automatically.
        # Init: zeros, NOT trunc_normal. Channel symmetry is already broken by
        # alpha_ch and t_ch, so d(loss)/d(p_ch(d)) differs per channel even at
        # p_ch == 0; a 0.02 jitter would instead perturb the largest
        # eigenvalue's response by ~15% (lam^p ~ exp(p ln lam), ln(19.7) = 3
        # amplifies raw-space jitter ~50x at the top bin).
        self.p_ch = nn.Parameter(torch.zeros(hidden_dim))

        # wave_weight: per-head learnable mixing weight of the damped
        # oscillation term. Init 0 -> forward at init == damping (decay) term.
        # 1-D shape so set_weight_decay excludes it from weight decay.
        self.wave_weight = nn.Parameter(torch.zeros(num_heads))

        # lambda cache (H, Wr) -> tensor. Attached from WaveFormer.make_layer
        # so every operator in a stage reuses the same cache.
        self._lambda_cache = {}
        self._shared_lambda_cache = None

    def attach_lambda_cache(self, cache: dict):
        """Point the operator at a cache owned by an outer scope (e.g. a stage).

        Any number of WaveOperators can share the same dict; the first
        forward at a given (H, W, device) fills it, and every subsequent
        forward at that resolution hits without rebuilding the grid.
        """
        self._shared_lambda_cache = cache

    @property
    def propagation_time(self):
        """Non-negative propagation time per head and channel, shape (K, D).

        t(k, d) = softplus(t_raw(k) + t_ch(k, d)) where the (C,) = (K*D,)
        t_ch is viewed as (K, D) so each head's D channels get their own
        time. K=1 (current base config) reduces this to a per-channel (C,)
        time around the scalar t_raw.
        """
        K, D = self.num_heads, self.head_dim
        return F.softplus(self.t_raw.view(K, 1) + self.t_ch.view(K, D))

    @property
    def spectral_exponent(self):
        """Positive spectral exponent p = softplus(p_raw + p_ch), shape (K, D).

        Init 1.0 (lam^1). The per-channel bias p_ch, viewed as (K, D), lets
        each channel pick its own decay shape.
        """
        K, D = self.num_heads, self.head_dim
        return F.softplus(self.p_raw.view(K, 1) + self.p_ch.view(K, D))

    def _get_lambda(self, H, W, device):
        cache = self._shared_lambda_cache if self._shared_lambda_cache is not None else self._lambda_cache
        key = (H, W, device)
        lam = cache.get(key)
        if lam is None:
            lam = _fourier_lambda_map(H, W, device=device, dtype=torch.float32)
            cache[key] = lam
        return lam

    def _project(self, x):
        """dwconv + linear head -> (u0, v0, z), each (B, C, H, W)."""
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                  # (B, H, W, C)
        x = self.linear(x)                          # (B, H, W, 3C)
        u0, v0, z = x.chunk(chunks=3, dim=-1)
        u0 = u0.permute(0, 3, 1, 2).contiguous()
        v0 = v0.permute(0, 3, 1, 2).contiguous()
        z = z.permute(0, 3, 1, 2).contiguous()
        return u0, v0, z

    def _gate_and_project_out(self, u_t, z, shape):
        x = _ifft2d(u_t, shape).to(z.dtype)         # (B, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()      # (B, H, W, C)
        x = self.out_norm(x)
        x = x * F.silu(z.permute(0, 2, 3, 1))
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, C, H, W)
        return x

    def forward(self, x: torch.Tensor, alpha_raw=None):
        if alpha_raw is None:
            raise ValueError(
                "WaveOperator.forward requires alpha_raw (stage-shared "
                "parameter of shape (res, res//2+1, K)). It is supplied by "
                "WaveFormer.forward_features via AdditionalInputSequential."
            )
        H, W = x.shape[-2:]
        B, C = x.shape[0], self.hidden_dim
        K, D = self.num_heads, self.head_dim  # C == K * D

        u0, v0, z = self._project(x)
        x_u0 = _fft2d(u0)                # (B, C, H, Wr)  complex
        x_v0 = _fft2d(v0)                # (B, C, H, Wr)  complex
        Wr = x_u0.shape[-1]

        # ---- lam^p, t, alpha on the (K, D, H, Wr) grid ----
        lam = self._get_lambda(H, W, u0.device)                 # (H, Wr)
        # Fractional power lam^p, per channel. The clamp only touches the DC
        # bin (lam = 0 -> 1e-12): lam^p = 1e-12^p ~ 0 there, so the DC
        # response stays 1, and d(lam^p)/dp = lam^p * ln(lam) is finite
        # everywhere (ln(1e-12) = -27.6 bounds it).
        ln_lam = lam.clamp(min=1e-12).log()                     # (H, Wr)
        # p: (K, D) -> (K, D, 1, 1)
        p = self.spectral_exponent.view(K, D, 1, 1)
        lam_p = torch.exp(p * ln_lam.view(1, 1, H, Wr))         # (K, D, H, Wr)
        # t: (K, D) -> (K, D, 1, 1)
        t = self.propagation_time.view(K, D, 1, 1)
        # alpha composition:
        #   alpha_raw: (H, Wr, K)     stage-shared per-position damping
        #   alpha_ch : (C,) = (K*D,)  per-block per-channel bias
        # Channels are laid out as K groups of D consecutive entries so
        # the (B, K, D, ...) view of the spectrum aligns with alpha_ch
        # reshaped to (K, D). We combine before softplus so both terms
        # share one saturation.
        alpha_ch_kd = self.alpha_ch.view(K, D)                    # (K, D)
        alpha = F.softplus(
            alpha_raw.unsqueeze(-1)                                # (H, Wr, K, 1)
            + alpha_ch_kd.view(1, 1, K, D)                         # (1, 1, K, D)
        ).clamp(min=WaveOperator.ALPHA_MIN)                        # (H, Wr, K, D)
        alpha = alpha.permute(2, 3, 0, 1)                          # (K, D, H, Wr)

        # ---- damping (decay) term ----
        # Real, non-negative, bounded by 1: alpha * lam^p * t >= 0, so
        # exp(-.) is in (0, 1].
        resp_damp = torch.exp(-alpha * lam_p * t)                  # (K, D, H, Wr)

        # ---- damped oscillation term, mixed in by wave_weight ----
        # U(t) = e^{-alpha lam^p t/2} [ U0 cos(wt) + (V0 + alpha lam^p/2 U0) sin(wt)/w ]
        # with w = sqrt(lam^p). wave_weight init 0 -> output == damping term.
        omega = torch.sqrt(lam_p.clamp(min=1e-12))                 # (K, D, H, Wr)
        omega_safe = omega.clamp(min=1e-6)                         # avoid /0 at DC
        phase = omega * t                                          # (K, D, H, Wr)
        envelope = torch.exp(-alpha * lam_p * t / 2.0)             # (K, D, H, Wr)
        resp_cos = envelope * torch.cos(phase)                     # (K, D, H, Wr)
        resp_sin = envelope * torch.sin(phase) / omega_safe        # (K, D, H, Wr)

        # apply to (B, K, D, H, Wr) spectrum
        spec_dtype = x_u0.dtype
        resp_damp = resp_damp.to(spec_dtype)
        resp_cos = resp_cos.to(spec_dtype)
        resp_sin = resp_sin.to(spec_dtype)
        x_u0 = x_u0.view(B, K, D, H, Wr)                           # (B, K, D, H, Wr)
        x_v0 = x_v0.view(B, K, D, H, Wr)                           # (B, K, D, H, Wr)

        u_t_damp = resp_damp * x_u0                                # 阻尼衰减项
        u_t_wave = (resp_cos * x_u0
                    + resp_sin * (x_v0 + (alpha * lam_p / 2.0) * x_u0))
        w = self.wave_weight.view(K, 1, 1, 1)                      # (K, 1, 1, 1)
        u_t = u_t_damp + w * (u_t_wave - u_t_damp)                 # wave_weight 初始为 0
        u_t = u_t.reshape(B, C, H, Wr)

        return self._gate_and_project_out(u_t, z, (H, W))


# =============================================================================
# WaveBlock and WaveFormer trunk.
#   * self.op in every block is a WaveOperator (see above).
#   * The trunk owns the stage-shared damping field `alpha_embed` and passes
#     it to every block of the stage.
#   * `infer_init` is a no-op (nothing to precompute).
# Stem, downsample, MLP, LN2d, residual layout, layer_scale, drop_path
# scheduling, classifier head and weight init follow a standard hierarchical
# 4-stage design.
# =============================================================================


class WaveBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode=False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm=True,
        layer_scale=None,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = WaveOperator(
            res=res, dim=hidden_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, infer_mode=infer_mode,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        self.infer_mode = infer_mode
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, alpha_raw):
        # alpha_raw is the stage-shared damping parameter, shape
        # (res, res//2+1, K). Passed straight through to self.op.
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, alpha_raw)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), alpha_raw))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, alpha_raw)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), alpha_raw))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor, alpha_raw=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, alpha_raw)
        else:
            return self._forward(input, alpha_raw)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class WaveFormer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, num_heads=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.depths = depths
        self.num_heads = num_heads

        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer='GELU',
                                     norm_layer='LN')

        res0 = img_size / patch_size
        self.res = [int(res0), int(res0 // 2), int(res0 // 4), int(res0 // 8)]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.infer_mode = infer_mode

        # alpha_embed: per-stage learnable damping field on the rfft2 grid, of
        # shape (res_i, res_i // 2 + 1, K). The damping term is
        # exp(-alpha * lam * t) with lam in [0, ~19.7] and t ~ 1 at init, so
        # alpha directly controls how strongly each block damps high
        # frequencies at initialisation. A small init keeps every block close
        # to identity at the start, with damping learned from data:
        # softplus^{-1}(0.05) = -2.9444 (max-freq damping
        # exp(-0.05 * 19.74 * 1) ~ 0.37, mid-freq ~ 0.78). trunc_normal_
        # jitter breaks position/head symmetry from step 0.
        num_heads_alpha = self.num_heads
        alpha_init_mean = -2.9706281  # softplus^{-1}(0.05)
        alpha_init_std = 0.02
        self.alpha_embed = nn.ParameterList()
        for i in range(self.num_layers):
            res_h = self.res[i]
            res_w = self.res[i] // 2 + 1
            a = torch.empty(res_h, res_w, num_heads_alpha)
            trunc_normal_(
                a,
                mean=alpha_init_mean,
                std=alpha_init_std,
                a=alpha_init_mean - 4 * alpha_init_std,
                b=alpha_init_mean + 4 * alpha_init_std,
            )
            self.alpha_embed.append(nn.Parameter(a))

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(self.make_layer(
                res=self.res[i_layer],
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=self.make_downsample(
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=LayerNorm2d,
                ) if (i_layer < self.num_layers - 1) else nn.Identity(),
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
                num_heads=self.num_heads,
            ))

        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Names containing any of these substrings are excluded from weight
        decay by the optimiser builder in utils/optimizer.py.

        alpha_embed is a stage-shared per-position damping field of shape
        (H, W//2+1, K). It is multi-dimensional so it would otherwise slip
        past the shape==1 rule in ``set_weight_decay``. It carries a
        meaningful init (softplus^{-1}(0.05) = -2.9706) and decaying it toward
        0 pulls alpha up (softplus(0)=0.693, ~14x stronger damping than init),
        which was empirically harmful early in training.

        t_raw, t_ch, p_raw, p_ch and wave_weight are 1-D and already excluded
        by shape==1.
        """
        return {"alpha_embed"}

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )

    @staticmethod
    def make_layer(
        res=14, dim=96, depth=2, drop_path=[0.1, 0.1], use_checkpoint=False,
        norm_layer=LayerNorm2d, post_norm=True, layer_scale=None,
        downsample=nn.Identity(), mlp_ratio=4.0, infer_mode=False,
        num_heads=4, **kwargs,
    ):
        assert depth == len(drop_path)
        # All WaveOperators in this stage see the same (H, W), so they share a
        # single Laplacian eigenvalue cache. In the deepest stage this saves
        # rebuilding lam 18 or 24 times per epoch's first batch; more
        # importantly it means the cache miss path only runs once per stage.
        stage_lambda_cache: dict = {}
        blocks = []
        for d in range(depth):
            block = WaveBlock(
                res=res,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
                num_heads=num_heads,
            )
            # Wire the operator inside this block into the stage-shared cache.
            block.op.attach_lambda_cache(stage_lambda_cache)
            blocks.append(block)
        return AdditionalInputSequential(*blocks, downsample)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        # No-op: the WaveOperator has nothing to precompute. Kept so main.py's
        # `if config.THROUGHPUT_MODE: model.infer_init()` still runs without
        # special-casing this build.
        pass

    def forward_features(self, x):
        x = self.patch_embed(x)
        for i, layer in enumerate(self.layers):
            # Pass stage-shared alpha_raw. AdditionalInputSequential forwards
            # it to every WaveBlock, which forwards it to WaveOperator.
            x = layer(x, self.alpha_embed[i])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    model = WaveFormer()
    inp = torch.randn((1, 3, 224, 224))
    analyze = FlopCountAnalysis(model, (inp,))
    print(flop_count_str(analyze))
