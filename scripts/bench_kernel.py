"""带宽基准：对比 v0 / v1(split-K)，同时给出 fp16 cuBLAS 基线。

用法（仓库根目录）：
    python scripts/bench_kernel.py                          # 默认 v0 + v1(split=4)
    python scripts/bench_kernel.py --kernel v0 v1 --split 1 2 4 8
    python scripts/bench_kernel.py --block-k 256 --split 4 8

指标：
    有效带宽 = 搬运字节 / 耗时
    pct      = 有效带宽 / 峰值带宽
    speedup  = fp16 耗时 / 本 kernel 耗时

注意：
    M=1 时小 shape 的耗时接近 kernel 启动开销（5~10us），pct 会被严重低估；
    看 BIG 那一行（合成大层，N=16384）更能反映 kernel 本身效率。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w4a16.kernel import w4a16_gemv                    # noqa: E402
from w4a16.kernel_v1 import w4a16_gemv_v1             # noqa: E402
from w4a16.quantize import gemv_ref, quantize_sym     # noqa: E402

G = 128

PEAK_BW = {
    "NVIDIA GeForce RTX 5090": 1792.0,
    "NVIDIA GeForce RTX 4090": 1008.0,
    "NVIDIA GeForce RTX 3090": 936.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA L20": 864.0,
}

SHAPES = [
    ("q_proj", 4096, 4096),
    ("k_proj", 1024, 4096),
    ("gate_proj", 12288, 4096),
    ("down_proj", 4096, 12288),
    ("BIG", 16384, 4096),
]


def w4a16_bytes(K: int, N: int, M: int) -> int:
    return K * (N // 8) * 4 + (K // G) * N * 2 + M * K * 2 + M * N * 4


def fp16_bytes(K: int, N: int, M: int) -> int:
    return K * N * 2 + M * K * 2 + M * N * 4


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", nargs="+", default=["v0", "v1"], choices=["v0", "v1"])
    ap.add_argument("--split", type=int, nargs="+", default=[4])
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--M", type=int, nargs="+", default=[1])
    ap.add_argument("--csv", type=str, default="results/s1_bandwidth.csv")
    ap.add_argument("--peak", type=float, default=None)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    peak = args.peak or PEAK_BW.get(gpu, 1008.0)
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count

    print(f"GPU   : {gpu}  ({n_sm} SM)")
    print(f"torch : {torch.__version__}   triton: {triton.__version__}")
    print(f"峰值带宽: {peak} GB/s   BLOCK_K={args.block_k}\n")

    rows = []
    for kern in args.kernel:
        for split in (args.split if kern == "v1" else [1]):
            for M in args.M:
                for name, N, K in SHAPES:
                    if kern == "v1" and K % (split * G) != 0:
                        print(f"skip  {kern} split={split} {name}: K={K} 不能整除 split*G")
                        continue

                    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
                    qw, s = quantize_sym(w)
                    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

                    call_v0 = lambda: w4a16_gemv(x, qw, s)                    # noqa: E731
                    call_v1 = lambda: w4a16_gemv_v1(                           # noqa: E731
                        x, qw, s, split=split, block_k=args.block_k)
                    fn = call_v0 if kern == "v0" else call_v1

                    if not args.no_check:
                        y2 = gemv_ref(x, qw, s)
                        y1 = fn()
                        err = ((y1 - y2).abs().max() / y2.abs().max()).item()
                        assert err < 1e-3, f"{kern} split={split} {name} 结果不对: err={err:.3e}"

                    t_k = bench(fn)
                    bw_k = w4a16_bytes(K, N, M) / (t_k * 1e-3) / 1e9

                    t_f = bench(lambda: torch.matmul(x, w.t()))
                    bw_f = fp16_bytes(K, N, M) / (t_f * 1e-3) / 1e9

                    rows.append(dict(
                        time=time.strftime("%Y-%m-%d %H:%M"), gpu=gpu,
                        kernel=kern, split=(split if kern == "v1" else 1),
                        block_k=args.block_k, M=M, layer=name, N=N, K=K,
                        us=round(t_k * 1000, 1),
                        GBs=round(bw_k, 1),
                        pct=round(bw_k / peak * 100, 1),
                        fp16_us=round(t_f * 1000, 1),
                        fp16_pct=round(bw_f / peak * 100, 1),
                        speedup=round(t_f / t_k, 2),
                    ))

                    print(
                        f"{kern} split={split if kern == 'v1' else 1} M={M:<3d} {name:<10s} "
                        f"N={N:6d} K={K:6d} | {t_k*1000:8.1f}us {bw_k:7.1f} GB/s "
                        f"({bw_k/peak*100:5.1f}%) | fp16 {t_f*1000:7.1f}us "
                        f"({bw_f/peak*100:5.1f}%) | {t_f/t_k:5.2f}x"
                    )

                    del w, qw, s, x
                    torch.cuda.empty_cache()

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n结果已写入 {args.csv}")

    big = [r for r in rows if r["layer"] == "BIG"]
    if big:
        print("\nBIG 层汇总：")
        for r in big:
            print(f"  {r['kernel']:2s} split={r['split']:<2d} {r['us']:8.1f}us  "
                  f"{r['pct']:5.1f}%  {r['speedup']:5.2f}x vs fp16")


if __name__ == "__main__":
    main()