"""W4A16 decode GEMV —— Triton v2：去掉第二次 kernel + 归约次数可选。

相对 v1 的两处改动：
1. 输出用 tl.atomic_add 直接累加，省掉 partial buffer 和第二次求和 kernel
   （v1 每次调用启动 2 个 kernel，小 shape 上开销占比很高）
2. 归约次数可选：
   - mode="loop"    ：每个 nibble 一次 tl.sum（8 次），寄存器压力小
   - mode="reshape" ：8 个 nibble 展开成 [BK, BNB, 8] 后 reshape，一次 tl.sum（1 次）
     寄存器压力大，需要较小的 BLOCK_K / BLOCK_NB

约束（v1 漏了，导致 BLOCK_K=256 时结果错）：
    G % BLOCK_K == 0     即 BLOCK_K 必须整除 128
    K % (split * G) == 0
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

G = 128
BLOCK_K = 128
BLOCK_NB = 32


@triton.jit
def _w4a16_v2_loop_kernel(
    x_ptr, qw_ptr, s_ptr, y_ptr,
    K: tl.constexpr, N: tl.constexpr, NB: tl.constexpr,
    SPLIT: tl.constexpr, G: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_NB: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    iota = tl.arange(0, 8)
    nb = pid_n * BLOCK_NB + tl.arange(0, BLOCK_NB)
    mask_nb = nb < NB

    KB: tl.constexpr = K // SPLIT
    ng: tl.constexpr = KB // G
    g0 = pid_s * ng

    acc = tl.zeros([BLOCK_NB, 8], dtype=tl.float32)

    for gi in range(0, ng):
        g = g0 + gi
        gacc = tl.zeros([BLOCK_NB, 8], dtype=tl.float32)

        for kk in range(0, G, BLOCK_K):
            k = g * G + kk + tl.arange(0, BLOCK_K)

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

    tl.atomic_add(
        y_ptr + nb[:, None] * 8 + iota[None, :],
        acc, mask=mask_nb[:, None],
    )


@triton.jit
def _w4a16_v2_reshape_kernel(
    x_ptr, qw_ptr, s_ptr, y_ptr,
    K: tl.constexpr, N: tl.constexpr, NB: tl.constexpr,
    SPLIT: tl.constexpr, G: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_NB: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    shift = 4 * tl.arange(0, 8)                             # [8]
    nb = pid_n * BLOCK_NB + tl.arange(0, BLOCK_NB)          # word 编号
    mask_nb = nb < NB
    n_flat = pid_n * BLOCK_NB * 8 + tl.arange(0, BLOCK_NB * 8)   # 输出通道

    KB: tl.constexpr = K // SPLIT
    ng: tl.constexpr = KB // G
    g0 = pid_s * ng

    acc = tl.zeros([BLOCK_NB * 8], dtype=tl.float32)

    for gi in range(0, ng):
        g = g0 + gi
        gacc = tl.zeros([BLOCK_NB * 8], dtype=tl.float32)

        for kk in range(0, G, BLOCK_K):
            k = g * G + kk + tl.arange(0, BLOCK_K)

            xk = tl.load(x_ptr + k)                         # [BLOCK_K]
            w32 = tl.load(
                qw_ptr + k[:, None] * NB + nb[None, :],
                mask=mask_nb[None, :], other=0,
            )                                               # [BLOCK_K, BLOCK_NB]

            # [BLOCK_K, BLOCK_NB, 8] -> [BLOCK_K, BLOCK_NB*8]
            q3 = (w32[:, :, None] >> shift[None, None, :]) & 0xF
            q3 = tl.where(q3 >= 8, q3 - 16, q3).to(tl.float32)
            q2 = tl.reshape(q3, (BLOCK_K, BLOCK_NB * 8))

            gacc += tl.sum(q2 * xk[:, None], axis=0)        # 一次归约

        s = tl.load(s_ptr + g * N + n_flat).to(tl.float32)
        acc += gacc * s

    tl.atomic_add(y_ptr + n_flat, acc)


def w4a16_gemv_v2(
    x: torch.Tensor,
    qw: torch.Tensor,
    scales: torch.Tensor,
    split: int = 8,
    block_k: int = BLOCK_K,
    block_nb: int = BLOCK_NB,
    mode: str = "loop",
) -> torch.Tensor:
    """x [M, K] + 量化权重 -> y [M, N] float32。"""
    assert x.dim() == 2 and x.shape[1] == qw.shape[0], (x.shape, qw.shape)
    assert mode in ("loop", "reshape"), mode
    K, NB = qw.shape
    N = NB * 8
    M = x.shape[0]

    assert K % (split * G) == 0, f"K={K} 不能被 split*G={split * G} 整除"
    assert G % block_k == 0, f"BLOCK_K={block_k} 必须整除 G={G}（否则跨 group 用错 scale）"
    assert block_nb * 8 <= N, (block_nb, N)

    x32 = x.to(torch.float32).contiguous()
    y = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(NB, block_nb), split)
    kern = _w4a16_v2_loop_kernel if mode == "loop" else _w4a16_v2_reshape_kernel

    for m in range(M):
        kern[grid](
            x32[m], qw, scales, y[m],
            K=K, N=N, NB=NB, SPLIT=split, G=G,
            BLOCK_K=block_k, BLOCK_NB=block_nb,
        )
    return y