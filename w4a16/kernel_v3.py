"""W4A16 decode GEMV/GEMM —— Triton v3：用 tl.dot 一次处理 M_TILE 行。

v2 的问题：wrapper 里 `for m in range(M)` 逐行调用，每行都把整个权重读一遍，
M=8 时带宽利用率从 49% 掉到 9%。

v3 的做法：
    - x 取 [M_TILE, BLOCK_K] 一块，权重 tile 反量化成 bf16 [BLOCK_K, BLOCK_N]
    - 用 tl.dot 做矩阵乘，归约交给 tensor core（顺便干掉了"8 次 tl.sum"）
    - 一次反量化的权重被 M_TILE 行复用 → M 增大时不再线性变慢

精度说明：
    x / w 都以 bf16 参与 matmul（tensor core 要求），累加 fp32。
    与 fp32 参考的差异来自 bf16 输入，属预期（~1e-3 量级）。

约束：
    G % BLOCK_K == 0          BLOCK_K 必须整除 128
    K % (split * G) == 0
    BLOCK_N % 8 == 0
    M_TILE 建议 16 或 32（tensor core 的 M 维）
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

G = 128
BLOCK_K = 64
BLOCK_N = 64
M_TILE = 32


@triton.jit
def _w4a16_v3_kernel(
    x_ptr, qw_ptr, s_ptr, y_ptr,
    M,
    K: tl.constexpr, N: tl.constexpr, NB: tl.constexpr,
    SPLIT: tl.constexpr, G: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, M_TILE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    shift = 4 * tl.arange(0, 8)
    BNB: tl.constexpr = BLOCK_N // 8

    nb = pid_n * BNB + tl.arange(0, BNB)                # word 编号
    mask_nb = nb < NB
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)         # 输出通道
    mask_n = n < N
    m = pid_m * M_TILE + tl.arange(0, M_TILE)           # 行（token）
    mask_m = m < M

    ng: tl.constexpr = (K // SPLIT) // G
    g0 = pid_s * ng

    acc = tl.zeros([M_TILE, BLOCK_N], dtype=tl.float32)

    for gi in range(0, ng):
        g = g0 + gi
        gacc = tl.zeros([M_TILE, BLOCK_N], dtype=tl.float32)

        for kk in range(0, G, BLOCK_K):
            k = g * G + kk + tl.arange(0, BLOCK_K)

            x_tile = tl.load(
                x_ptr + m[:, None] * K + k[None, :],
                mask=mask_m[:, None], other=0.0,
            )                                           # [M_TILE, BLOCK_K] bf16

            w32 = tl.load(
                qw_ptr + k[:, None] * NB + nb[None, :],
                mask=mask_nb[None, :], other=0,
            )                                           # [BLOCK_K, BNB] int32

            q3 = (w32[:, :, None] >> shift[None, None, :]) & 0xF
            q3 = tl.where(q3 >= 8, q3 - 16, q3)
            w_tile = tl.reshape(q3.to(tl.bfloat16), (BLOCK_K, BLOCK_N))

            gacc += tl.dot(x_tile, w_tile)              # [M_TILE, BLOCK_N] fp32

        s = tl.load(s_ptr + g * N + n, mask=mask_n, other=0.0).to(tl.float32)
        acc += gacc * s[None, :]

    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(y_ptr + m[:, None] * N + n[None, :], acc, mask=out_mask)


def w4a16_gemv_v3(
    x: torch.Tensor,
    qw: torch.Tensor,
    scales: torch.Tensor,
    split: int = 1,
    block_k: int = BLOCK_K,
    block_n: int = BLOCK_N,
    m_tile: int = M_TILE,
) -> torch.Tensor:
    """x [M, K] + 量化权重 -> y [M, N] float32。M 建议 <= m_tile。"""
    assert x.dim() == 2 and x.shape[1] == qw.shape[0], (x.shape, qw.shape)
    K, NB = qw.shape
    N = NB * 8
    M = x.shape[0]

    assert G % block_k == 0, f"BLOCK_K={block_k} 必须整除 G={G}"
    assert K % (split * G) == 0, f"K={K} 不能被 split*G={split * G} 整除"
    assert block_n % 8 == 0, block_n

    xb = x.to(torch.bfloat16).contiguous()
    y = torch.zeros((M, N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(N, block_n), triton.cdiv(M, m_tile), split)
    _w4a16_v3_kernel[grid](
        xb, qw, scales, y, M,
        K=K, N=N, NB=NB, SPLIT=split, G=G,
        BLOCK_K=block_k, BLOCK_N=block_n, M_TILE=m_tile,
    )
    return y