"""RAPT GEMM (release) benchmark at M=N=K=8192 on AMD Radeon AI PRO R9700 / R9600D.

Protocol: torch.rand inputs (U[0,1); fp8 = torch.rand cast to E4M3), weights packed once (not timed), 50 warm-up +
200 timed launches back to back, time = CUDA(HIP)-event elapsed / 200. Precision: torch.allclose(rtol=1.6e-2, atol=1e-4)
against torch's GEMM on the same inputs and output dtype (torch.matmul for bf16/fp16, torch._scaled_mm with unit scales
and bf16 output for fp8), plus the share of output elements that are not identical to torch's.
usage: python bench.py [--dtypes bf16,fp16,fp8] [--warmup 50] [--iters 200] [--shapes 8192x8192x8192,...]
"""
import argparse
import sys

import torch

import rapt_gemm

DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": rapt_gemm.FP8}
RTOL, ATOL = 1.6e-2, 1e-4                                    # torch's standard bf16 test tolerance


def torch_gemm(x, w):
    if x.dtype == rapt_gemm.FP8:
        one = torch.ones((), dtype=torch.float32, device=x.device)
        return torch._scaled_mm(x, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    return torch.matmul(x, w.t())


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / iters                       # ms per launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="8192x8192x8192", help="comma list of MxNxK")
    ap.add_argument("--dtypes", default="bf16,fp16,fp8")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    print(f"{p.name} ({getattr(p, 'gcnArchName', '?')})  torch {torch.__version__}  HIP {torch.version.hip}")
    if not rapt_gemm.device_supported(dev):
        print("[RAPT GEMM] unsupported product: this package runs only on AMD Radeon AI PRO R9700 / R9600D")
        sys.exit(1)
    print(f"protocol: torch.rand, {a.warmup} warm-up + {a.iters} timed, event time / {a.iters}\n")
    print(f"{'shape MxNxK':>20} {'dtype':>5} | {'ms':>8} {'TFLOPS':>7} | precision test vs torch "
          f"(rtol={RTOL:g}, atol={ATOL:g}) | elements different from torch")
    for s in a.shapes.split(","):
        M, N, K = map(int, s.lower().split("x"))
        for dn in a.dtypes.split(","):
            dt = DT[dn]
            torch.manual_seed(0)
            if dt == rapt_gemm.FP8:
                x = torch.rand(M, K, device=dev).to(dt)
                w = torch.rand(N, K, device=dev).to(dt)
            else:
                x = torch.rand(M, K, dtype=dt, device=dev)
                w = torch.rand(N, K, dtype=dt, device=dev)
            if not rapt_gemm.supported(M, N, K, dt):
                print(f"{s:>20} {dn:>5} | unsupported shape (needs N%256==0, K%{rapt_gemm.k_granularity(dt)}==0)")
                continue
            wp = rapt_gemm.pack_weight(w)
            y = torch.empty(M, N, dtype=rapt_gemm.out_dtype(dt), device=dev)
            rapt_gemm.linear(x, wp, out=y)
            ref = torch_gemm(x, w)
            ok = torch.allclose(y, ref, rtol=RTOL, atol=ATOL)
            ndiff = int((y != ref).sum())
            del ref
            ms = time_fn(lambda: rapt_gemm.linear(x, wp, out=y), a.warmup, a.iters)
            tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12
            print(f"{s:>20} {dn:>5} | {ms:8.3f} {tf:7.1f} | {'PASSED' if ok else 'FAILED':>38} | "
                  f"{100.0 * ndiff / (M * N):.3f}%")


if __name__ == "__main__":
    main()
