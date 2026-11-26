import torch

G = 128      # group_size
PACK = 8     # 每个 int32 装 8 个 int4


def quantize_sym(w: torch.Tensor):
    """w: [N, K] bf16 -> qweight int32[K, N//8], scales bf16[K//G, N]"""
    w = w.float()
    N, K = w.shape
    assert K % G == 0 and N % PACK == 0, (N, K)

    g = w.reshape(N, K // G, G)                        # [N, K//G, G]
    amax = g.abs().amax(dim=2, keepdim=True)           # [N, K//G, 1]
    scale = torch.clamp(amax / 7.0, min=1e-8)          # ← 防全零组除零
    q = torch.clamp(torch.round(g / scale), -7, 7)
    q = q.to(torch.int8).reshape(N, K)                 # [N, K]
    scales = scale.squeeze(2).t().contiguous()         # [K//G, N]
    return pack_int4(q), scales


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: int8[N, K]（[-7,7]） -> int32[K, N//8]（沿 N 打包）"""
    assert q.dtype == torch.int8
    N, K = q.shape
    assert N % PACK == 0

    g = (q.reshape(N // PACK, PACK, K).to(torch.int32) & 0xF)   # [N/8, 8, K]
    packed = torch.zeros(N // PACK, K, dtype=torch.int32)
    for i in range(PACK):
        packed |= g[:, i, :] << (4 * i)                         # n = 8*nb + i
    return packed.t().contiguous()                              # [K, N//8]


def unpack_int4(qw: torch.Tensor) -> torch.Tensor:
    """int32[K, N//8] -> int8[N, K]"""
    assert qw.dtype == torch.int32
    K, NB = qw.shape
    t = qw.t().contiguous()                                     # [N/8, K]
    parts = [((t >> (4 * i)) & 0xF) for i in range(PACK)]
    parts = [torch.where(p >= 8, p - 16, p) for p in parts]
    return torch.stack(parts, dim=1).reshape(NB * PACK, K).to(torch.int8)


def dequant_ref(qw, scales) -> torch.Tensor:
    """-> [N, K] float32（kernel 的数值基准）"""
    q = unpack_int4(qw).float()                        # [N, K]
    s = scales.t().repeat_interleave(G, dim=1)         # [N, K/G] -> [N, K]
    assert s.shape == q.shape
    return q * s


def gemv_ref(x, qw, scales) -> torch.Tensor:
    """y = x @ dequant(w).T；x: [M, K] -> [M, N]"""
    return x.float() @ dequant_ref(qw, scales).t()