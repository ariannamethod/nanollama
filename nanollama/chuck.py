"""
Chuck Optimizer — PyTorch Edition

θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η

Faithful port of Chuck v7/v8 from lee.c.
Drop-in replacement for AdamW with 9 levels of self-awareness.
Adam is blind. Chuck sees. Chuck remembers.

Binary-compatible with C chuck.mem format (4 floats × 16 bytes per entry).

In memory of Carlos Ray "Chuck" Norris (1940–2026).
"""

import math
import os
import re
import struct
import random
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import torch
from torch.optim import Optimizer


# ═══════════════════════════════════════════════════════════════════════
# Chuck Memory — persistent across training runs
# Binary-compatible with C version (chuck.mem from lee.c)
# ═══════════════════════════════════════════════════════════════════════

class ChuckMemory:
    """Persistent training memory with reservoir sampling.

    Each entry: (loss, grad_norm, lambda, delta_loss) = 16 bytes.
    Binary layout matches C struct ChuckMem from lee.c — files are
    interchangeable between C and Python.
    """

    ENTRY_FMT = 'ffff'
    ENTRY_SIZE = struct.calcsize(ENTRY_FMT)  # 16 bytes

    def __init__(self, capacity: int = 500, path: str = 'chuck.mem'):
        self.capacity = capacity
        self.path = path
        self.entries: List[Tuple[float, float, float, float]] = []
        self.total: int = 0

    def load(self) -> int:
        """Load memories from binary file. Returns count loaded."""
        if not os.path.exists(self.path):
            return 0
        with open(self.path, 'rb') as f:
            data = f.read()
        n = len(data) // self.ENTRY_SIZE
        self.entries = []
        for i in range(min(n, self.capacity)):
            e = struct.unpack_from(self.ENTRY_FMT, data, i * self.ENTRY_SIZE)
            self.entries.append(e)
        self.total = len(self.entries)
        return len(self.entries)

    def save_entry(self, loss: float, grad_norm: float, lam: float,
                   delta_loss: float):
        """Record a memory snapshot. Reservoir sampling when full."""
        self.total += 1
        entry = (float(loss), float(grad_norm), float(lam), float(delta_loss))
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
            with open(self.path, 'ab') as f:
                f.write(struct.pack(self.ENTRY_FMT, *entry))
        else:
            slot = random.randint(0, self.total - 1)
            if slot < self.capacity:
                self.entries[slot] = entry
                with open(self.path, 'wb') as f:
                    for e in self.entries:
                        f.write(struct.pack(self.ENTRY_FMT, *e))

    def recall(self, loss: float, grad_norm: float) -> float:
        """Nearest-neighbor lookup → λ_prior. Returns -1 if no memory."""
        if not self.entries:
            return -1.0
        best_dist = float('inf')
        best_lambda = -1.0
        for e_loss, e_gnorm, e_lam, e_dloss in self.entries:
            dl = (loss - e_loss) / (abs(loss) + 1e-8)
            dg = (grad_norm - e_gnorm) / (abs(grad_norm) + 1e-8)
            dist = dl * dl + dg * dg
            if e_dloss < 0:        # prefer memories where loss improved
                dist *= 0.5
            if dist < best_dist:
                best_dist = dist
                best_lambda = e_lam
        return best_lambda

    def __len__(self) -> int:
        return len(self.entries)


# ═══════════════════════════════════════════════════════════════════════
# Chuck Monitor — forward hooks for σ (activation health)
# ═══════════════════════════════════════════════════════════════════════

class ChuckMonitor:
    """Forward hooks for σ — activation health signal.

    Auto-attaches to SiLU/GELU and LayerNorm/RMSNorm modules.
    Call feed_attention_entropy() with raw attention weights for Level 8.

    Usage::

        monitor = ChuckMonitor(model)
        # ... forward pass ...
        optimizer.step(loss=loss_val)   # ChuckOptimizer reads monitor.sigma
    """

    def __init__(self, model: torch.nn.Module):
        self.hooks: list = []
        self._silu_alive: List[float] = []
        self.silu_health: float = 1.0
        self.norm_scale_ema: float = 1.0
        self._norm_init: bool = False
        self.attn_entropy_ema: List[float] = []
        self._attn_init: bool = False
        self._h_max: float = 1.0
        self.act_magnitudes: List[float] = []
        self._attach(model)

    # ------------------------------------------------------------------
    def _attach(self, model: torch.nn.Module):
        norm_types: list = [torch.nn.LayerNorm]
        if hasattr(torch.nn, 'RMSNorm'):
            norm_types.append(torch.nn.RMSNorm)
        norm_tuple = tuple(norm_types)

        layer_idx = 0
        for _name, module in model.named_modules():
            # Level 3: activation health
            if isinstance(module, (torch.nn.SiLU, torch.nn.GELU)):
                self.hooks.append(
                    module.register_forward_hook(self._silu_hook))
            # Level 3: norm stability
            elif isinstance(module, norm_tuple):
                self.hooks.append(
                    module.register_forward_hook(self._norm_hook))
            # Custom RMSNorm by class name
            if ('rmsnorm' in type(module).__name__.lower()
                    and not isinstance(module, norm_tuple)):
                self.hooks.append(
                    module.register_forward_hook(self._norm_hook))
            # Level 5: signal flow — detect transformer blocks
            cls = type(module).__name__.lower()
            if any(k in cls for k in
                   ['block', 'decoderlayer', 'encoderlayer']):
                idx = layer_idx
                self.hooks.append(module.register_forward_hook(
                    lambda _m, _i, o, i=idx: self._signal_hook(i, o)))
                layer_idx += 1

    def _silu_hook(self, _module, _input, output):
        with torch.no_grad():
            self._silu_alive.append(
                (output.abs() > 1e-6).float().mean().item())

    def _norm_hook(self, _module, _input, output):
        with torch.no_grad():
            scale = output.norm(dim=-1).mean().item()
            if not self._norm_init:
                self.norm_scale_ema = scale
                self._norm_init = True
            else:
                self.norm_scale_ema = (0.99 * self.norm_scale_ema
                                       + 0.01 * scale)

    def _signal_hook(self, layer_idx: int, output):
        with torch.no_grad():
            out = output[0] if isinstance(output, tuple) else output
            mag = out.abs().mean().item()
            while len(self.act_magnitudes) <= layer_idx:
                self.act_magnitudes.append(0.0)
            self.act_magnitudes[layer_idx] = mag

    # ------------------------------------------------------------------
    def feed_attention_entropy(self, attn_weights: torch.Tensor):
        """Feed raw attention weights ``[B, H, S, S]`` for Level 8.

        Call this after each forward pass if your model exposes
        attention weights.
        """
        with torch.no_grad():
            self._h_max = math.log(attn_weights.shape[-1])
            eps = 1e-8
            # entropy per head, averaged over batch & query positions
            ent = -(attn_weights * (attn_weights + eps).log()) \
                .sum(-1).mean(dim=(0, 2))  # → [H]
            if not self._attn_init:
                self.attn_entropy_ema = ent.tolist()
                self._attn_init = True
            else:
                for i in range(len(self.attn_entropy_ema)):
                    self.attn_entropy_ema[i] = (
                        0.99 * self.attn_entropy_ema[i]
                        + 0.01 * ent[i].item())

    # ------------------------------------------------------------------
    @property
    def sigma(self) -> float:
        """Compute σ ∈ (0, 1] — composite health signal."""
        s = 1.0
        # SiLU / GELU health
        if self._silu_alive:
            health = sum(self._silu_alive) / len(self._silu_alive)
            self.silu_health = health
            if health < 0.7:
                s *= health / 0.7
        # Norm stability
        if self._norm_init:
            if self.norm_scale_ema > 5.0 or self.norm_scale_ema < 0.2:
                s *= 0.9
        # Attention entropy
        if self._attn_init and self.attn_entropy_ema:
            for h_ent in self.attn_entropy_ema:
                ratio = h_ent / (self._h_max + 1e-8)
                if ratio < 0.1:
                    s *= 0.95      # collapsed head
                elif ratio > 0.95:
                    s *= 0.98      # fully diffuse head
        return s

    @property
    def signal_flow_ratio(self) -> Optional[float]:
        """act_mag[last] / act_mag[first].  None if not tracked."""
        if (len(self.act_magnitudes) >= 2
                and self.act_magnitudes[0] > 1e-8):
            return (self.act_magnitudes[-1]
                    / (self.act_magnitudes[0] + 1e-8))
        return None

    def reset(self):
        """Clear per-step state.  Called automatically by ChuckOptimizer."""
        self._silu_alive = []

    def detach(self):
        """Remove all hooks from the model."""
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ═══════════════════════════════════════════════════════════════════════
# Chuck Optimizer
# ═══════════════════════════════════════════════════════════════════════

class ChuckOptimizer(Optimizer):
    r"""Self-aware optimizer — 9 levels of awareness.

    .. math::
        \theta \mathrel{-}= (\alpha \times S \times \lambda_\Psi
        \times \lambda_l \times \sigma)\;
        \frac{\hat m}{\sqrt{\hat v}+\varepsilon} + \eta

    Drop-in replacement for ``torch.optim.AdamW``.
    Pass ``loss`` to :meth:`step` for full Chuck awareness.
    Without ``loss`` it degrades gracefully to vanilla Adam.

    Args:
        params: iterable of parameters or param groups.
            Each group may contain a ``'layer'`` key (int) for
            per-layer awareness.  Use :func:`chuck_params` to
            auto-detect layers.
        lr: peak learning rate.
        betas: Adam momentum coefficients (β₁, β₂).
        eps: Adam ε.
        weight_decay: decoupled L2 (AdamW-style).
        window: loss / grad trend window (steps).
        damp_range: (λ_min, λ_max) clamp for dampen.
        psi_cap: maximum Ψ trust weight.
        psi_half: memories for Ψ_w = 50 % of cap.
        mem_cap: max entries in chuck.mem (reservoir).
        mem_path: path to persistent memory file.
        rec_thr: λ-shift threshold for recording a memory.
        rec_cd: cooldown steps between recordings.
        macro_int: macro patience check interval (steps).
        macro_pat: checks without improvement → LR drop.
        macro_decay: LR scale factor per macro drop.
        monitor: optional :class:`ChuckMonitor` (enables σ).
        freeze_thr: grad norm below which a layer may freeze.
        freeze_pat: low-norm steps before freeze fires.
        verbose: print status every N steps (0 = silent).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        window: int = 16,
        damp_range: Tuple[float, float] = (0.3, 2.0),
        psi_cap: float = 0.3,
        psi_half: float = 100.0,
        mem_cap: int = 500,
        mem_path: str = 'chuck.mem',
        rec_thr: float = 0.25,
        rec_cd: int = 50,
        macro_int: int = 1000,
        macro_pat: int = 3,
        macro_decay: float = 0.5,
        monitor: Optional['ChuckMonitor'] = None,
        freeze_thr: float = 0.01,
        freeze_pat: int = 8,
        verbose: int = 0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.window = window
        self.damp_lo, self.damp_hi = damp_range
        self.psi_cap = psi_cap
        self.psi_half = psi_half
        self.rec_thr = rec_thr
        self.rec_cd = rec_cd
        self.macro_int = macro_int
        self.macro_pat = macro_pat
        self.macro_decay = macro_decay
        self.monitor = monitor
        self.freeze_thr = freeze_thr
        self.freeze_pat = freeze_pat
        self.verbose = verbose

        # Persistent memory
        self.memory = ChuckMemory(capacity=mem_cap, path=mem_path)
        loaded = self.memory.load()

        # ── Global state (Chuck's soul) ──────────────────────────────
        self.dampen: float = 1.0
        self.noise: float = 0.0
        self.sigma: float = 1.0
        self.loss_ema: float = 0.0
        self.gnorm_ema: float = 0.0
        self.psi: float = 0.0
        self.psi_w: float = (
            min(self.psi_cap, loaded / (loaded + self.psi_half))
            if loaded > 0 else 0.0)
        self.macro_ema: float = 0.0
        self.best_macro: float = 1e9
        self.lr_scale: float = 1.0
        self.macro_stag: int = 0
        self.macro_drops: int = 0
        self.rec_lambda: float = 1.0
        self.rec_loss: float = 999.0
        self._rec_cd_ctr: int = 0
        self.global_step: int = 0

        # Loss ring buffer
        self._hist = [0.0] * window
        self._hpos: int = 0
        self._hfull: bool = False
        self._stag: int = 0

        # Per-layer state
        self._layers: Dict[int, dict] = {}
        self._rec_frozen: Dict[int, bool] = {}
        self._lmap: Dict[int, int] = {}        # group_idx → layer_id
        for gi, group in enumerate(self.param_groups):
            lid = group.get('layer', gi)
            self._lmap[gi] = lid
            if lid not in self._layers:
                self._layers[lid] = dict(
                    ghist=[0.0] * window, dampen=1.0, frozen=False,
                    pos=0, full=False, stag=0)
                self._rec_frozen[lid] = False

        if loaded > 0:
            print(f'  chuck: loaded {loaded} memories from {mem_path} '
                  f'(\u03a8_w={self.psi_w:.2f})')

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _clamp(self, d: float) -> float:
        return max(self.damp_lo, min(self.damp_hi, d))

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None, *, loss: Optional[float] = None):
        """Perform one optimisation step.

        Args:
            closure: optional closure that re-evaluates the model and
                returns the loss (as a tensor).
            loss: current scalar loss value (float).  Required for full
                Chuck awareness; without it Chuck runs as vanilla Adam.
        """
        if closure is not None:
            with torch.enable_grad():
                lv = closure()
                if loss is None:
                    loss = lv.item()

        if loss is None:
            return self._adam_fallback()

        self.global_step += 1
        W = self.window

        # ═══ Level 1: Global λ (loss trend) ══════════════════════════
        if self.loss_ema == 0.0:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss

        self._hist[self._hpos % W] = self.loss_ema
        self._hpos += 1
        if self._hpos >= W:
            self._hfull = True

        if self._hfull:
            q = W // 4
            recent = sum(
                self._hist[(self._hpos - 1 - i) % W]
                for i in range(q)) / q
            old = sum(
                self._hist[(self._hpos - W + i) % W]
                for i in range(q)) / q
            trend = (recent - old) / (old + 1e-8)
            if trend > 0.02:
                self.dampen *= 0.97        # loss rising → brake (symmetric)
            elif trend < -0.02:
                self.dampen *= 1.03        # loss falling → push (symmetric)
            if abs(trend) < 0.001:
                self._stag += 1
                if self._stag > 8:
                    self.noise = 0.001
                    self._stag = 0
            else:
                self._stag = 0
                self.noise *= 0.9
            # Mean reversion: prevent dampen from getting stuck
            self.dampen = 0.999 * self.dampen + 0.001 * 1.0
            self.dampen = self._clamp(self.dampen)

        # ═══ Level 9: Macro patience ═════════════════════════════════
        if self.macro_ema == 0.0:
            self.macro_ema = loss
        else:
            self.macro_ema = 0.999 * self.macro_ema + 0.001 * loss
        if (self.global_step % self.macro_int == 0
                and self.global_step > W):
            if self.macro_ema > self.best_macro * 0.999:
                self.macro_stag += 1
                if self.macro_stag >= self.macro_pat:
                    self.lr_scale *= self.macro_decay
                    if self.lr_scale < 0.05:
                        self.lr_scale = 0.05
                    self.macro_stag = 0
                    self.macro_drops += 1
            else:
                self.best_macro = self.macro_ema
                self.macro_stag = 0
                # Macro recovery: if improving, let lr_scale grow back
                if self.lr_scale < 1.0:
                    self.lr_scale = min(1.0, self.lr_scale * 1.2)

        # ═══ Level 4: σ (activation health from monitor) ═════════════
        self.sigma = self.monitor.sigma if self.monitor else 1.0

        # ═══ Grad norms (shared by Levels 2, 5, 6) ═══════════════════
        lgnorms: Dict[int, float] = defaultdict(float)
        total_gsq = 0.0
        for gi, group in enumerate(self.param_groups):
            lid = self._lmap[gi]
            for p in group['params']:
                if p.grad is not None:
                    gsq = p.grad.norm().item() ** 2
                    lgnorms[lid] += gsq
                    total_gsq += gsq
        for lid in lgnorms:
            lgnorms[lid] = math.sqrt(lgnorms[lid])
        gnorm = math.sqrt(total_gsq + 1e-8)

        # ═══ Level 2: Per-layer λ_l (grad norm trend) ════════════════
        for lid, ls in self._layers.items():
            if ls['frozen']:
                continue
            gn = lgnorms.get(lid, 0.0)
            ls['ghist'][ls['pos'] % W] = gn
            ls['pos'] += 1
            if ls['pos'] >= W:
                ls['full'] = True
            if ls['full']:
                q = W // 4
                recent = sum(
                    ls['ghist'][(ls['pos'] - 1 - i) % W]
                    for i in range(q)) / q
                old = sum(
                    ls['ghist'][(ls['pos'] - W + i) % W]
                    for i in range(q)) / q
                trend = (recent - old) / (old + 1e-8)
                if trend > 0.05:
                    ls['dampen'] *= 1.05   # grad rising → boost
                elif trend < -0.05:
                    ls['dampen'] *= 0.95   # grad settling → ease
                # freeze check
                if gn < self.freeze_thr:
                    ls['stag'] += 1
                    if ls['stag'] > self.freeze_pat:
                        ls['frozen'] = True
                else:
                    ls['stag'] = 0
                ls['dampen'] = self._clamp(ls['dampen'])

        # ═══ Level 5: Cross-layer signal flow ═════════════════════════
        if self.monitor is not None:
            fr = self.monitor.signal_flow_ratio
            if fr is not None:
                skeys = sorted(self._layers.keys())
                nl = len(skeys)
                if nl > 1:
                    for idx, lid in enumerate(skeys):
                        ls = self._layers[lid]
                        if ls['frozen']:
                            continue
                        depth = idx / (nl - 1)
                        if fr < 0.3:
                            ls['dampen'] *= (1.0 + 0.02 * depth)
                        elif fr > 3.0:
                            ls['dampen'] *= (1.0 - 0.02 * depth)
                        ls['dampen'] = self._clamp(ls['dampen'])

        # ═══ Level 6: Ψ — subjectivity ═══════════════════════════════
        mem_n = len(self.memory)
        self.psi_w = (
            min(self.psi_cap, mem_n / (mem_n + self.psi_half))
            if mem_n > 0 else 0.0)
        lambda_psi = self.dampen
        if mem_n > 0:
            lp = self.memory.recall(loss, gnorm)
            if lp > 0:
                self.psi = lp - self.dampen
                lambda_psi = self.dampen + self.psi_w * self.psi
                lambda_psi = self._clamp(lambda_psi)

        # Memory recording on regime change
        self._rec_cd_ctr += 1
        if self._hfull and self._rec_cd_ctr >= self.rec_cd:
            dl = loss - self.rec_loss
            shift = abs(self.dampen - self.rec_lambda) / (
                self.rec_lambda + 1e-8)
            regime = shift > self.rec_thr
            if not regime:
                for lid, ls in self._layers.items():
                    if ls['frozen'] != self._rec_frozen.get(lid, False):
                        regime = True
                        break
            if regime:
                self.memory.save_entry(loss, gnorm, self.dampen, dl)
                self.rec_lambda = self.dampen
                self.rec_loss = loss
                self._rec_cd_ctr = 0
                for lid, ls in self._layers.items():
                    self._rec_frozen[lid] = ls['frozen']

        # ═══ Adaptive gradient clipping ═══════════════════════════════
        if self.gnorm_ema == 0.0:
            self.gnorm_ema = gnorm
        else:
            self.gnorm_ema = 0.97 * self.gnorm_ema + 0.03 * gnorm
        aclip = 1.0
        if self.gnorm_ema > 1e-8:
            aclip = max(0.5, min(2.0, 1.5 * self.gnorm_ema))
            if gnorm > 3.0 * self.gnorm_ema:
                aclip *= 0.5
        clip = aclip / gnorm if gnorm > aclip else 1.0

        # ═══ Parameter updates ════════════════════════════════════════
        for gi, group in enumerate(self.param_groups):
            lid = self._lmap[gi]
            ls = self._layers[lid]
            if ls['frozen']:
                continue

            b1, b2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group.get('weight_decay', 0.0)
            eff_lr = lr * lambda_psi * ls['dampen'] * self.sigma \
                * self.lr_scale

            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st['step'] = 0
                    st['m'] = torch.zeros_like(p.data)
                    st['v'] = torch.zeros_like(p.data)
                st['step'] += 1
                m, v = st['m'], st['v']

                # Decoupled weight decay (AdamW) — scaled by lr_scale
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd * self.lr_scale)

                g = p.grad * clip
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                bc1 = 1.0 - b1 ** st['step']
                bc2 = 1.0 - b2 ** st['step']
                denom = (v / bc2).sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-eff_lr / bc1)

                if self.noise > 0:
                    p.data.add_(torch.randn_like(p.data),
                                alpha=self.noise)

        # Post-step cleanup
        if self.monitor is not None:
            self.monitor.reset()

        # Verbose logging (matches C output format)
        if self.verbose > 0 and self.global_step % self.verbose == 0:
            frz = ' '.join(
                f'L{l}:frz'
                for l, s in sorted(self._layers.items())
                if s['frozen'])
            print(
                f'step {self.global_step:>6d} | loss {loss:.4f} | '
                f'chuck: \u03bb={self.dampen:.2f} '
                f'\u03a8={self.psi:+.2f} ({len(self.memory)} mem) '
                f'\u03c3={self.sigma:.2f} macro={self.lr_scale:.2f}'
                f'{" | " + frz if frz else ""}')

    # ------------------------------------------------------------------
    def _adam_fallback(self):
        """Vanilla Adam — used when step() is called without loss."""
        for group in self.param_groups:
            b1, b2 = group['betas']
            lr, eps = group['lr'], group['eps']
            wd = group.get('weight_decay', 0.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st['step'] = 0
                    st['m'] = torch.zeros_like(p.data)
                    st['v'] = torch.zeros_like(p.data)
                st['step'] += 1
                m, v = st['m'], st['v']
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                g = p.grad
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1 = 1.0 - b1 ** st['step']
                bc2 = 1.0 - b2 ** st['step']
                denom = (v / bc2).sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-lr / bc1)

    # ------------------------------------------------------------------
    # state save / load (preserves Chuck's soul across checkpoints)
    # ------------------------------------------------------------------
    def state_dict(self):
        sd = super().state_dict()
        sd['chuck'] = dict(
            dampen=self.dampen, noise=self.noise, sigma=self.sigma,
            loss_ema=self.loss_ema, gnorm_ema=self.gnorm_ema,
            psi=self.psi, psi_w=self.psi_w,
            macro_ema=self.macro_ema, best_macro=self.best_macro,
            lr_scale=self.lr_scale, macro_stag=self.macro_stag,
            macro_drops=self.macro_drops,
            rec_lambda=self.rec_lambda, rec_loss=self.rec_loss,
            rec_cd_ctr=self._rec_cd_ctr, global_step=self.global_step,
            hist=list(self._hist), hpos=self._hpos,
            hfull=self._hfull, stag=self._stag,
            layers={str(k): dict(v) for k, v in self._layers.items()},
            rec_frozen={str(k): v
                        for k, v in self._rec_frozen.items()},
        )
        return sd

    def load_state_dict(self, state_dict):
        chuck = state_dict.pop('chuck', None)
        super().load_state_dict(state_dict)
        if chuck is not None:
            for k in ('dampen', 'noise', 'sigma', 'loss_ema',
                       'gnorm_ema', 'psi', 'psi_w', 'macro_ema',
                       'best_macro', 'lr_scale', 'macro_stag',
                       'macro_drops', 'rec_lambda', 'rec_loss',
                       'global_step'):
                setattr(self, k, chuck[k])
            self._rec_cd_ctr = chuck['rec_cd_ctr']
            self._hist = chuck['hist']
            self._hpos = chuck['hpos']
            self._hfull = chuck['hfull']
            self._stag = chuck['stag']
            self._layers = {int(k): v
                            for k, v in chuck['layers'].items()}
            self._rec_frozen = {int(k): v
                                for k, v in chuck['rec_frozen'].items()}
            self.memory.load()

    # ------------------------------------------------------------------
    @property
    def frozen_layers(self) -> List[int]:
        """Layer IDs that Chuck has frozen."""
        return [lid for lid, ls in self._layers.items()
                if ls['frozen']]

    def unfreeze_all(self):
        """Unfreeze every layer.  Useful before fine-tuning."""
        for ls in self._layers.values():
            ls['frozen'] = False
            ls['stag'] = 0


# ═══════════════════════════════════════════════════════════════════════
# Utility: auto-detect transformer layers
# ═══════════════════════════════════════════════════════════════════════

def chuck_params(model: torch.nn.Module, lr: float = 3e-3,
                 **kw) -> List[dict]:
    """Auto-detect transformer layers and build param groups.

    Scans ``model.named_parameters()`` for common patterns::

        .layers.N.   .blocks.N.   .h.N.
        .encoder.layer.N.   .decoder.layer.N.

    Parameters that don't match get ``layer=-1`` (global — typically
    embeddings / output heads; eligible for update but not for
    per-layer freezing).

    Returns a list of param-group dicts ready for
    ``ChuckOptimizer(chuck_params(model), ...)``.
    """
    patterns = [
        r'(?:^|\.)layers\.(\d+)\.', r'(?:^|\.)blocks\.(\d+)\.',
        r'(?:^|\.)h\.(\d+)\.', r'(?:^|\.)encoder\.layer\.(\d+)\.',
        r'(?:^|\.)decoder\.layer\.(\d+)\.',
    ]
    buckets: Dict[int, list] = defaultdict(list)
    glob: list = []
    for name, param in model.named_parameters():
        matched = False
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                buckets[int(m.group(1))].append(param)
                matched = True
                break
        if not matched:
            glob.append(param)
    groups = []
    if glob:
        groups.append(dict(params=glob, layer=-1, lr=lr, **kw))
    for idx in sorted(buckets):
        groups.append(dict(params=buckets[idx], layer=idx, lr=lr, **kw))
    return groups
