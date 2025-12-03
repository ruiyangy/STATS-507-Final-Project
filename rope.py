import torch
from typing import Iterable, Tuple

def _dtype_tag(dt: torch.dtype) -> str:
    if dt is torch.float32:  return "fp32"
    if dt is torch.bfloat16: return "bf16"
    if dt is torch.float16:  return "fp16"
    raise ValueError(f"Unsupported dtype for cache: {dt}")

class Rotary(torch.nn.Module):
    """
    CUDA-graph safe RoPE with per-dtype caches.
    - Precompute on init (or via prepare_dtypes) up to max_seq_len
    - forward only selects & slices; no assignments/mutations
    """
    def __init__(
        self,
        head_dim: int,
        *,
        max_seq_len: int,
        base: float = 10000.0,
        cache_dtypes: Iterable[torch.dtype] = (torch.float32,)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)

        # ---- Build fp32 base once (on CPU; will move with .to(...)) ----
        half = head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(self.max_seq_len, dtype=torch.float32)          # [S]
        freqs = t[:, None] * inv_freq[None, :]                            # [S, half]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)                   # [S, D]
        cos_base = emb.cos()                                              # [S, D] fp32
        sin_base = emb.sin()

        # Keep the fp32 base (useful fallback & for making other dtypes later)
        self.register_buffer("cos_fp32", cos_base, persistent=False)
        self.register_buffer("sin_fp32", sin_base, persistent=False)

        # ---- Optionally pre-materialize dtype caches ----
        want = tuple(dict.fromkeys(cache_dtypes))  # de-dup while preserving order
        self._materialize_dtype_caches(want)

    @torch.no_grad()
    def _materialize_dtype_caches(self, dtypes: Iterable[torch.dtype]) -> None:
        """
        Create dtype-specific buffers (on current module device) from fp32 base.
        Safe to call BEFORE torch.compile capture. Do NOT call inside forward.
        """
        for dt in dtypes:
            tag = _dtype_tag(dt)
            cos_name, sin_name = f"cos_{tag}", f"sin_{tag}"
            if hasattr(self, cos_name):  # already present -> skip
                continue
            # Make on the module's current device so they move together if you later .to(...)
            dev = self.cos_fp32.device
            cos_cast = self.cos_fp32.to(device=dev, dtype=dt)
            sin_cast = self.sin_fp32.to(device=dev, dtype=dt)
            self.register_buffer(cos_name, cos_cast, persistent=False)
            self.register_buffer(sin_name, sin_cast, persistent=False)

    def prepare_dtypes(self, dtypes: Iterable[torch.dtype]) -> None:
        """
        Public helper to be called OUTSIDE any compiled/captured region.
        Lets you add more dtype caches (e.g., after moving model to CUDA).
        """
        self._materialize_dtype_caches(dtypes)

    def _get_cache_for(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tag = _dtype_tag(dtype)
        cos_name, sin_name = f"cos_{tag}", f"sin_{tag}"
        if hasattr(self, cos_name):
            return getattr(self, cos_name), getattr(self, sin_name)
        # Fallback: slice from fp32 and cast ephemerally (no buffer reassignment).
        # Consider calling prepare_dtypes([dtype]) ahead of time to avoid this cast at runtime.
        return self.cos_fp32.to(dtype), self.sin_fp32.to(dtype)

    def forward(self, x: torch.Tensor, *, seq_dim: int = -2, offset: int = 0):
        """
        Returns (cos, sin) broadcastable to x (no state mutation).
        Assumes offset + seq_len <= max_seq_len.
        """
        if seq_dim < 0:
            seq_dim = x.dim() + seq_dim
        seq_len = x.shape[seq_dim]
        end = offset + seq_len
        if end > self.max_seq_len:
            raise RuntimeError(
                f"RoPE: requested positions [{offset}:{end}) exceed max_seq_len={self.max_seq_len}"
            )

        cos_base, sin_base = self._get_cache_for(x.dtype)  # [S, D] on module device
        # If model was moved (e.g., .to('cuda')), these buffers moved with it.
        # If x is on a different device, that's a user/model bug—keep devices consistent.

        # Slice (creates a view) and then shape to broadcast onto x
        cos = cos_base[offset:end, :]  # [S, D]
        sin = sin_base[offset:end, :]
        view = [1] * x.dim()
        view[seq_dim] = seq_len
        view[-1] = x.shape[-1]
        return cos.view(*view), sin.view(*view)


@torch.jit.script
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1) // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
