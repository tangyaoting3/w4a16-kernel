"""v0 带宽基准测试：测 kernel 的有效带宽 vs 峰值带宽。

用法（在仓库根目录）：
    python scripts/bench_kernel.py                 # 默认 M=1
    python scripts/bench_kernel.py --M 1 8 32      # 多个 M
    python scripts/bench_kernel.py --excel results/s1_bandwidth.csv

指标说明：
    有效带宽 = 搬运字节 / kernel 耗时
    带宽利用率 = 有效带宽 / 峰值带宽（5090 为 1792 GB/s）

注意：
    M=1 时单层耗时可能只有几微秒，接近 kernel 启动开销（5~10 us），
    此时带宽比例会被显著低估。看 BIG 那一行（合成大层）更能反映 kernel 本身效率。
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

from w4a16.kernel import w4a16_gemv          # noqa: E402
from w4a16.quantize import gemv_ref, quantize_sym   # noqa: E402

G = 128

PEAK_BW = {
    "NVIDIA GeForce RTX 5090": 1792.0,
    "NVIDIA GeForce RTX 4090": 1008.0,
    "NVIDIA GeForce RTX 3090": 936.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA L20": 864.0,
}

# (name, N, K)  —— 对应 Qwen3-8B 的几类 linear，最后一个是合成大层
SHAPES = [
    ("q_proj", 4096, 4096),
    ("k_proj", 1024, 4096),
    ("gate_proj", 12288, 4096),
    ("down_proj", 4096, 12288),
    ("BIG", 16384, 4096),      # 合成：让耗时远离启动开销，反映 kernel 真实效率
]


def w4a16_bytes(K: int, N: int, M: int) -> int:
    return K * (N // 8) * 4 + (K // G) * N * 2 + M * K * 2 + M * N * 4


def fp16_bytes(K: int, N: int, M: int) -> int:
    return K * N * 2 + M * K * 2 + M * N * 4


def bench(fn) -> float:
    """返回中位耗时（毫秒）。"""
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1])
    ap.add_argument("--csv", type=str, default="results/s1_bandwidth.csv")
    ap.add_argument("--peak", type=float, default=None, help="手动指定峰值带宽 GB/s")
    ap.add_argument("--no-check", action="store_true", help="跳过正确性校验")
    args = ap.parse_args()

    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    peak = args.peak or PEAK_BW.get(gpu, 1008.0)

    print(f"GPU   : {gpu}")
    print(f"torch : {torch.__version__}   triton: {triton.__version__}")
    print(f"峰值带宽: {peak} GB/s\n")

    rows = []
    for M in args.M:
        for name, N, K in SHAPES:
            w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
            qw, s = quantize_sym(w)
            x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

            # 正确性：先确认在测的是对的 kernel
            if not args.no_check:
                y1 = w4a16_gemv(x, qw, s)
                y2 = gemv_ref(x, qw, s)
                err = ((y1 - y2).abs().max() / y2.abs().max()).item()
                assert err < 1e-3, f"{name} 结果不对: err={err:.3e}"

            t_k = bench(lambda: w4a16_gemv(x, qw, s))
            b_k = w4a16_bytes(K, N, M)
            bw_k = b_k / (t_k * 1e-3) / 1e9

            t_f = bench(lambda: torch.matmul(x, w.t()))
            b_f = fp16_bytes(K, N, M)
            bw_f = b_f / (t_f * 1e-3) / 1e9

            rows.append(dict(
                time=time.strftime("%Y-%m-%d %H:%M"),
                gpu=gpu, M=M, layer=name, N=N, K=K,
                kernel_us=round(t_k * 1000, 1),
                kernel_GBs=round(bw_k, 1),
                kernel_pct=round(bw_k / peak * 100, 1),
                fp16_us=round(t_f * 1000, 1),
                fp16_GBs=round(bw_f, 1),
                fp16_pct=round(bw_f / peak * 100, 1),
                speedup=round(t_f / t_k, 2),
            ))

            print(
                f"M={M:<3d} {name:<10s} N={N:6d} K={K:6d} | "
                f"w4a16 {t_k*1000:8.1f}us {bw_k:7.1f} GB/s ({bw_k/peak*100:5.1f}%) | "
                f"fp16 {t_f*1000:8.1f}us {bw_f:7.1f} GB/s ({bw_f/peak*100:5.1f}%) | "
                f"speedup {t_f/t_k:5.2f}x"
            )

            del w, qw, s, x
            torch.cuda.empty_cache()

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n结果已写入 {args.csv}")

    # 一句话结论
    big = [r for r in rows if r["layer"] == "BIG"]
    if big:
        b = big[0]
        print(f"结论：v0 在 BIG 上的带宽利用率 = {b['kernel_pct']}%，"
              f"相对 fp16 提速 {b['speedup']}x")


if __name__ == "__main__":
    main()