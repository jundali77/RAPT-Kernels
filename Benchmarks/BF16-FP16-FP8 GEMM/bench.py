"""RAPT GEMM (release) benchmark at M=N=K=8192 on AMD Radeon AI PRO R9700 / R9600D.

Protocol: torch.rand inputs (U[0,1); fp8 = torch.rand cast to E4M3), weights packed once (not timed), 50 warm-up +
200 timed launches back to back, time = CUDA(HIP)-event elapsed / 200. Precision: torch.allclose(rtol=1.6e-2, atol=1e-4)
against torch's GEMM on the same inputs and output dtype (torch.matmul for bf16/fp16; for fp8 torch._scaled_mm with
per-tensor float32 scales scale_a=0.75, scale_b=1.25 and bf16 output, RAPT given the very same scale tensors),
plus the share of output elements that are not identical to torch's.
Speed: torch's GEMM on the same inputs is timed the same way. Each round times RAPT and torch once; the order alternates
between rounds (RAPT, torch | torch, RAPT | ...) so clock/temperature drift at the power cap cancels; reported = mean.
--gap S idles S seconds before every timed run (use e.g. --gap 30 --warmup 500 --iters 2000 for power-steady numbers).
usage: python bench.py [--dtypes bf16,fp16,fp8] [--warmup 50] [--iters 200] [--rounds 2] [--gap 0]
                       [--shapes 8192x8192x8192,...]
"""
import argparse
import sys
import time

import torch

import rapt_gemm

DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": rapt_gemm.FP8}
RTOL, ATOL = 1.6e-2, 1e-4                                    # torch's standard bf16 test tolerance
SCALE_A, SCALE_B = 0.75, 1.25                                # fp8 per-tensor scales (non-unit, so both are checked)


def torch_gemm(x, w, out=None, sa=None, sb=None):
    if x.dtype == rapt_gemm.FP8:
        return torch._scaled_mm(x, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    return torch.matmul(x, w.t(), out=out)


def time_fn(fn, warmup, iters, gap=0.0):
    if gap > 0:
        time.sleep(gap)
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
    ap.add_argument("--rounds", type=int, default=2, help="RAPT/torch timing rounds, order alternating")
    ap.add_argument("--gap", type=float, default=0.0, help="idle seconds before every timed run")
    a = ap.parse_args()
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    print(f"{p.name} ({getattr(p, 'gcnArchName', '?')})  torch {torch.__version__}  HIP {torch.version.hip}")
    if not rapt_gemm.device_supported(dev):
        print("[RAPT GEMM] unsupported product: this package runs only on AMD Radeon AI PRO R9700 / R9600D")
        sys.exit(1)
    print(f"protocol: torch.rand, {a.warmup} warm-up + {a.iters} timed, event time / {a.iters}, {a.rounds} rounds "
          f"(RAPT/torch order alternating), {a.gap:g} s idle before each timed run\n")
    print(f"{'shape MxNxK':>20} {'dtype':>5} | {'RAPT ms':>8} {'TFLOPS':>7} | {'torch ms':>8} {'TFLOPS':>7} | "
          f"{'speedup':>7} | precision vs torch (rtol={RTOL:g}, atol={ATOL:g}) | elements different")
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
            sa = sb = None
            if dt == rapt_gemm.FP8:
                sa = torch.tensor(SCALE_A, dtype=torch.float32, device=dev)
                sb = torch.tensor(SCALE_B, dtype=torch.float32, device=dev)
            rapt_gemm.linear(x, wp, out=y, scale_a=sa, scale_b=sb)
            ref = torch_gemm(x, w, sa=sa, sb=sb)
            ok = torch.allclose(y, ref, rtol=RTOL, atol=ATOL)
            ndiff = int((y != ref).sum())
            yt = ref if dt != rapt_gemm.FP8 else None
            fns = [lambda: rapt_gemm.linear(x, wp, out=y, scale_a=sa, scale_b=sb),
                   lambda: torch_gemm(x, w, out=yt, sa=sa, sb=sb)]
            t = [0.0, 0.0]
            for r in range(a.rounds):
                for j in ((0, 1) if r % 2 == 0 else (1, 0)):
                    t[j] += time_fn(fns[j], a.warmup, a.iters, a.gap)
            ms, tms = t[0] / a.rounds, t[1] / a.rounds
            del ref, yt
            tf, ttf = (2.0 * M * N * K / (v * 1e-3) / 1e12 for v in (ms, tms))
            print(f"{s:>20} {dn:>5} | {ms:8.3f} {tf:7.1f} | {tms:8.3f} {ttf:7.1f} | {tms / ms:6.3f}x | "
                  f"{'PASSED' if ok else 'FAILED':>35} | {100.0 * ndiff / (M * N):.3f}%")


if __name__ == "__main__":
    main()
