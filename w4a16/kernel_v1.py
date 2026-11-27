"""W4A16 decode GEMV —— Triton v1：在 v0 基础上加 split-K。

相对 v0 只改一件事（便于归因）：
    v0 每个 program 串行扫完整条 K，program 数 = N/128（q_proj 只有 32 个，
    5090 有 170 个 SM，并行度严重不足）。
    v1 把 K 切成 SPLIT 段，每段一个 program，最后把 SPLIT 份 partial 相加。
    program 数变成 (N/128) × SPLIT，串行长度变成 1/SPLIT。

约束：
    K 必须能被 SPLIT * G 整除（保证每个 split 落在 group 边界上）

其余与 v0 完全一致：
    - 二维累加器 [BLOCK_NB, 8]，Triton 不支持 list comprehension
    - G 走 constexpr 参数，jit 函数读不到模块级变量
    - 沿 N 打包，低 4 bit 存较小的 n
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

G = 128
BLOCK_K = 64
BLOCK_NB = 16          # 每个 program 处理 16 个 word = 128 个输出通道


@triton.jit
def _w4a16_gemv_v1_kernel(
    x_ptr, qw_ptr, s_ptr, part_ptr,
    K: tl.constexpr, N: tl.constexpr, NB: tl.constexpr,
    SPLIT: tl.constexpr, G: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_NB: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    iota = tl.arange(0, 8)
    nb = pid_n * BLOCK_NB + tl.arange(0, BLOCK_NB)
    mask_nb = nb < NB

    # 本 split 负责的 group 区间：[g0, g0 + ng)
    KB: tl.constexpr = K // SPLIT
    ng: tl.constexpr = KB // G
    g0 = pid_s * ng

    acc = tl.zeros([BLOCK_NB, 8], dtype=tl.float32)

    for gi in range(0, ng):
        g = g0 + gi
        gacc = tl.zeros([BLOCK_NB, 8], dtype=tl.float32)

        for kk in range(0, G, BLOCK_K):
            k = g * G + kk + tl.arange(0, BLOCK_K)          # [BLOCK_K]

            xk = tl.load(x_ptr + k)                         # [BLOCK_K]
            w32 = tl.load(
                qw_ptr + k[:, None] * NB + nb[None, :],
                mask=mask_nb[None, :], other=0,
            )                                               # [BLOCK_K, BLOCK_NB]

            for i in tl.static_range(8):
                q = (w32 >> (4 * i)) & 0xF
                q = tl.where(q >= 8, q - 16, q)
                col = tl.sum(q.to(tl.float32) * xk[:, None], axis=0)   # [BLOCK_NB]
                gacc += tl.where(iota[None, :] == i, col[:, None], 0.0)

        s = tl.load(
            s_ptr + g * N + nb[:, None] * 8 + iota[None, :],
            mask=mask_nb[:, None], other=0.0,
        ).to(tl.float32)
        acc += gacc * s

    # 写到本 split 的 partial 槽位
    tl.store(
        part_ptr + pid_s * N + nb[:, None] * 8 + iota[None, :],
        acc, mask=mask_nb[:, None],
    )


def w4a16_gemv_v1(
    x: torch.Tensor,
    qw: torch.Tensor,
    scales: torch.Tensor,
    split: int = 4,
    block_k: int = BLOCK_K,
    block_nb: int = BLOCK_NB,
) -> torch.Tensor:
    """x [M, K] + 量化权重 -> y [M, N] float32（split-K 版本）。

    split=1 时行为与 v0 一致（仅多一次 partial 求和）。
    """
    assert x.dim() == 2 and x.shape[1] == qw.shape[0], (x.shape, qw.shape)
    K, NB = qw.shape
    N = NB * 8
    M = x.shape[0]

    assert K % (split * G) == 0, f"K={K} 不能被 split*G={split * G} 整除"
    assert block_nb * 8 <= N, f"block_nb={block_nb} 对应的 {block_nb * 8} 个通道超过 N={N}"

    x32 = x.to(torch.float32).contiguous()
    part = torch.empty((M, split, N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(NB, block_nb), split)
    for m in range(M):
        _w4a16_gemv_v1_kernel[grid](
            x32[m], qw, scales, part[m],
            K=K, N=N, NB=NB, SPLIT=split, G=G,
            BLOCK_K=block_k, BLOCK_NB=block_nb,
        )

    return part.sum(dim=1)      # [M, N]