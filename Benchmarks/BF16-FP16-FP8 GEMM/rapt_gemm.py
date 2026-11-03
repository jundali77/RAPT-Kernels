"""RAPT GEMM (release) -- Python wrapper for the gfx1201 GEMM kernels in librapt_gemm.so.

Supported products: AMD Radeon AI PRO R9700 and R9600D only (device_supported() returns False on anything else,
and linear() then returns False instead of running).

    w_packed = rapt_gemm.pack_weight(w)        # once per weight, w [N, K]
    y = rapt_gemm.linear(x, w_packed)          # y[M,N] = x[M,K] @ w^T
                                               # bf16 -> bf16, fp16 -> fp16, fp8 e4m3 -> bf16 (no scales)
Shape rules: N % 256 == 0, K % 256 == 0 (fp8: K % 512 == 0), any M; x contiguous [M, K] on the GPU.
"""
import ctypes
from pathlib import Path

import torch

print("[RAPT GEMM] This kernel is specially tuned for the M=N=K=8192 case; for other shapes the performance "
      "might vary.")

_LIB = ctypes.CDLL(str(Path(__file__).resolve().parent / "librapt_gemm.so"))
for _f in (_LIB.rapt_gemm_bf16, _LIB.rapt_gemm_f16, _LIB.rapt_gemm_fp8):
    _f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _f.restype = ctypes.c_int
_LIB.rapt_gemm_device_supported.argtypes = [ctypes.c_int]
_LIB.rapt_gemm_device_supported.restype = ctypes.c_int
FP8 = torch.float8_e4m3fn                      # OCP E4M3, the gfx12 WMMA fp8 format
_FN = {torch.bfloat16: _LIB.rapt_gemm_bf16, torch.float16: _LIB.rapt_gemm_f16, FP8: _LIB.rapt_gemm_fp8}
BN, GSM = 256, 8
_warned = set()


def device_supported(device=None) -> bool:
    """True only on an AMD Radeon AI PRO R9700 or R9600D."""
    if not torch.cuda.is_available():
        return False
    d = torch.device("cuda") if device is None else torch.device(device)
    if d.type != "cuda":
        return False
    return bool(_LIB.rapt_gemm_device_supported(torch.cuda.current_device() if d.index is None else d.index))


def k_granularity(dtype: torch.dtype) -> int:
    return 512 if dtype == FP8 else 256


def out_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.bfloat16 if dtype == FP8 else dtype


def supported(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    return dtype in _FN and M >= 1 and N % BN == 0 and K % k_granularity(dtype) == 0


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> the kernel's WMMA fragment order (done once per weight; fp8 is packed in 16-bit units)."""
    N, K = w.shape
    if N % BN or K % k_granularity(w.dtype):
        raise ValueError(f"pack_weight: N={N} must be a multiple of {BN}, K={K} of {k_granularity(w.dtype)}")
    u = w.contiguous().view(torch.int16) if w.dtype == FP8 else w.contiguous()
    p = u.view(N // 16, 16, u.shape[1] // 128, 8, 2, 8).permute(0, 2, 3, 4, 1, 5).contiguous()
    return p.view(w.dtype) if w.dtype == FP8 else p


def linear(x: torch.Tensor, w_packed: torch.Tensor, out: torch.Tensor = None):
    """x [M, K] @ w^T with w packed by pack_weight; returns [M, N] in out_dtype(x.dtype), or False when the GPU is
    not an R9700 / R9600D."""
    if not device_supported(x.device):
        if x.device not in _warned:
            _warned.add(x.device)
            name = torch.cuda.get_device_name(x.device) if x.is_cuda else str(x.device)
            print(f"[RAPT GEMM] unsupported product '{name}': only AMD Radeon AI PRO R9700 / R9600D are supported")
        return False
    M, K = x.shape
    N = w_packed.shape[0] * 16
    if not supported(M, N, K, x.dtype) or w_packed.dtype != x.dtype:
        raise ValueError(f"unsupported GEMM M={M} N={N} K={K} {x.dtype} (weights {w_packed.dtype})")
    if not (x.is_contiguous() and w_packed.is_contiguous()):
        raise ValueError("x and w_packed must be contiguous GPU tensors")
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype(x.dtype), device=x.device)
    with torch.cuda.device(x.device):
        err = _FN[x.dtype](x.data_ptr(), w_packed.data_ptr(), out.data_ptr(), M, N, K, GSM,
                           torch.cuda.current_stream(x.device).cuda_stream)
    if err:
        raise RuntimeError(f"rapt_gemm launch failed (hipError {err})")
    return out
