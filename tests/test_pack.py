"""pack / unpack / reference 的正确性测试（CPU 即可，不花租金）。

测试分两类，别混：
1. 实现正确性 —— 和自己的慢速参考实现比，容差极紧（1e-6）
2. 量化误差   —— 和 fp32 全精度比，容差很松（随机高斯权重下理论值 ~0.13）
"""
import pytest
import torch

from w4a16.quantize import (
    dequant_ref,
    gemv_ref,
    pack_int4,
    quantize_sym,
    unpack_int4,
)

G = 128


def test_roundtrip():
    """pack -> unpack 必须逐 bit 一致。"""
    q = torch.randint(-7, 7, (16, 256), dtype=torch.int8)

    qw = pack_int4(q)
    assert qw.shape == (256, 2), f"形状错了: {tuple(qw.shape)}, 期望 (256, 2)"
    assert qw.dtype == torch.int32, qw.dtype

    back = unpack_int4(qw)
    assert torch.equal(back, q), "pack/unpack 不是逐 bit 可逆"


def test_doc_example():
    """docs/format.md 第 7 节的手算例子。"""
    q = torch.zeros(8, 2, dtype=torch.int8)
    q[:, 0] = torch.tensor([1, -2, 3, -4, 5, -6, 7, -7], dtype=torch.int8)

    qw = pack_int4(q)
    assert qw.shape == (2, 1), f"形状错了: {tuple(qw.shape)}, 期望 (2, 1)"

    # 0x97A5C3E1 = 2544223201 > 2**31，作为 int32 读出是负数，必须按无符号比较
    got = int(qw[0, 0]) & 0xFFFFFFFF
    assert got == 0x97A5C3E1, f"packed word = 0x{got:08X}, 期望 0x97A5C3E1"


def test_zero_group_no_nan():
    """全零 group 不能产生 NaN（scale 除零防护）。"""
    w = torch.zeros(8, 256, dtype=torch.bfloat16)

    qw, scales = quantize_sym(w)
    assert torch.isfinite(scales).all(), "scales 出现 NaN/inf"
    assert torch.isfinite(dequant_ref(qw, scales)).all(), "反量化结果出现 NaN/inf"


def test_nibble_isolation():
    """改一个量化值，只影响对应的 (k, word)。"""
    q = torch.zeros(8, 256, dtype=torch.int8)
    qw0 = pack_int4(q)

    q2 = q.clone()
    q2[3, 100] = 5          # 输出通道 n=3, 输入通道 k=100
    qw1 = pack_int4(q2)

    diff = qw1 != qw0
    assert int(diff.sum()) == 1, f"影响了 {int(diff.sum())} 个 word，期望 1"
    assert bool(diff[100, 0]), "应该只改 k=100、nb=0 这一个 word"


# --------------------------------------------------------------------------- #
# 实现正确性：和独立写的慢速参考比（不能和 gemv_ref 自己比，那是同义反复）
# --------------------------------------------------------------------------- #
def _slow_gemv(x: torch.Tensor, qw: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """逐输出通道的朴素实现，用 int64 解包、float64 累加。"""
    K, NB = qw.shape
    N = NB * 8
    assert scales.shape == (K // G, N), f"scales 形状应为 {(K // G, N)}，实际 {tuple(scales.shape)}"

    y = torch.zeros(x.shape[0], N, dtype=torch.float64)
    s64 = scales.double()                             # [K/G, N]

    for n in range(N):
        nb, i = n // 8, n % 8
        word = qw[:, nb].to(torch.int64)              # [K]
        q = (word >> (4 * i)) & 0xF                   # 无符号 nibble
        q = torch.where(q >= 8, q - 16, q).double()   # 转有符号
        # scales[:, n] 是第 n 个输出通道的 K/G 个 group scale，沿 K 展开
        w_n = q * s64[:, n].repeat_interleave(G)      # [K]
        y[:, n] = x.double() @ w_n

    return y


def test_gemv_matches_slow_ref():
    """向量化实现必须和朴素实现数值一致（容差 1e-6）。"""
    torch.manual_seed(0)
    N, K = 16, 256

    w = torch.randn(N, K, dtype=torch.bfloat16)
    x = torch.randn(3, K, dtype=torch.bfloat16)

    qw, scales = quantize_sym(w)
    y_fast = gemv_ref(x, qw, scales)                  # [M, N]
    y_slow = _slow_gemv(x, qw, scales)

    err = float((y_fast.double() - y_slow).norm() / y_slow.norm())
    assert err < 1e-6, f"与朴素实现偏差 {err:.2e}"


# --------------------------------------------------------------------------- #
# 量化误差：随机高斯权重下，输出相对误差的理论值约 0.13
#   scale = amax/7 ≈ 0.44σ，量化误差 std ≈ scale/√12 ≈ 0.127σ
#   Σx·ε 与 Σx·w 同比放大 √K，比值 ≈ 0.127
# 真实 LLM 权重比随机高斯友好，最终精度看 PPL，不看这个数
# --------------------------------------------------------------------------- #
def test_quant_error_bound():
    """随机权重下的量化误差应落在合理区间，不能太大也不能为 0。"""
    torch.manual_seed(0)
    N, K = 64, 256

    w = torch.randn(N, K, dtype=torch.bfloat16)
    x = torch.randn(4, K, dtype=torch.bfloat16)

    qw, scales = quantize_sym(w)
    y_ref = gemv_ref(x, qw, scales)
    y_true = x.float() @ w.float().t()

    err = float((y_ref - y_true).norm() / y_true.norm())
    assert 0.05 < err < 0.25, (
        f"相对误差 {err:.4f} 超出预期区间 [0.05, 0.25]；"
        "接近 0 说明量化没生效，明显偏大说明打包/解包有问题"
    )


if __name__ == "__main__":
    # 支持 `python tests/test_pack.py` 直接跑，不依赖 pytest
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")