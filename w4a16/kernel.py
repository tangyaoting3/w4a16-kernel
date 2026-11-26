"""W4A16 decode GEMV —— Triton v0（只求正确，不求性能）。

格式见 docs/format.md：
    qweight : [K, N//8] int32，沿 N 打包，低 4 bit 存较小的 n
    scales  : [K//G, N] bf16
    x       : [M, K]     y : [M, N]

v0 的设计取舍：
- 一个 program 负责 BLOCK_NB 个 word（= BLOCK_NB * 8 个输出通道）
- 外层按 group（G=128）循环，group 内按 BLOCK_K 分块，scale 在 group 内是常数
- 一个 word 展开 8 个 nibble → 8 个独立累加器，对应 n = 8*nb + i
- M > 1 由 Python 层循环调用，先保证正确
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

G = 128
BLOCK_K = 64
BLOCK_NB = 16          # 每个 program 处理 16 个 word = 128 个输出通道


@triton.jit
def _w4a16_gemv_kernel(
    x_ptr, qw_ptr, s_ptr, y_ptr,
    K: tl.constexpr, N: tl.constexpr, NB: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_NB: tl.constexpr,
):
    pid = tl.program_id(0)
    nb = pid * BLOCK_NB + tl.arange(0, BLOCK_NB)      # word 编号
    mask_nb = nb < NB

    # 8 个累加器：第 i 个对应输出通道 n = 8*nb + i
    acc = [tl.zeros([BLOCK_NB], dtype=tl.float32) for _ in range(8)]

    for g in range(0, K // G):
        gacc = [tl.zeros([BLOCK_NB], dtype=tl.float32) for _ in range(8)]
        for kk in range(0, G, BLOCK_K):
            k = g * G + kk + tl.arange(0, BLOCK_K)                  # [BLOCK_K]

            xk = tl.load(x_ptr + k)                                 # [BLOCK_K] fp32
            w32 = tl.load(
                qw_ptr + k[:, None] * NB + nb[None, :],
                mask=mask_nb[None, :], other=0,
            )                                                       # [BLOCK_K, BLOCK_NB] int32

            for i in range(8):
                q = (w32 >> (4 * i)) & 0xF                          # 无符号 nibble
                q = tl.where(q >= 8, q - 16, q)                     # 4bit 补码转有符号
                gacc[i] += tl.sum(q.to(tl.float32) * xk[:, None], axis=0)

        # group 内累加完毕，乘一次该 group 的 scale
        for i in range(8):
            s = tl.load(s_ptr + g * N + nb * 8 + i, mask=mask_nb, other=0.0).to(tl.float32)
            acc[i] += gacc[i] * s

    for i in range(8):
        tl.store(y_ptr + nb * 8 + i, acc[i], mask=mask_nb)


def w4a16_gemv(x: torch.Tensor, qw: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """x [M, K] (bf16/fp16/fp32) + 量化权重 -> y [M, N] float32。"""
    assert x.dim() == 2 and x.shape[1] == qw.shape[0], (x.shape, qw.shape)
    K, NB = qw.shape
    N = NB * 8
    M = x.shape[0]

    x32 = x.to(torch.float32).contiguous()
    y = torch.empty((M, N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(NB, BLOCK_NB),)
    for m in range(M):
        _w4a16_gemv_kernel[grid](
            x32[m], qw, scales, y[m],
            K=K, N=N, NB=NB, BLOCK_K=BLOCK_K, BLOCK_NB=BLOCK_NB,
        )
    return y