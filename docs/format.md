# W4A16 权重格式规范

> 本文档定义本项目的权重打包格式，是所有 kernel、量化脚本、参考实现必须遵守的唯一约定。
> 改动此格式前，先改本文档。

---

## 1. 一句话

对称量化（无 zero-point），**沿 N（输出通道）** 把 8 个 int4 打进一个 int32，得到 `qweight: [K, N/8]`，与 SGLang / AutoAWQ 的 `packed_dim=1` 一致。

## 2. 张量清单

| 名称 | shape | dtype | 含义 |
|---|---|---|---|
| `W` | `[N, K]` | bf16 | `nn.Linear.weight`，N = 输出通道，K = 输入通道 |
| `q` | `[N, K]` | int8 | 量化值，值域 `[-8, 7]`，对称 |
| `scales` | `[K/128, N]` | bf16 | 每个（输出通道, 128 个 k 一组）一个 scale |
| `qweight` | `[K, N/8]` | int32 | 打包后的权重 |
| `x` | `[M, K]` | bf16 | 输入，M = batch×seq 展平后的 token 数 |
| `y` | `[M, N]` | bf16 | 输出 |

字节数校验：`qweight.numel() * 4 == K * N // 2`（即 0.5 byte / 权重）。

## 3. 打包流程图

```
W (bf16)          [N, K]
      │  quantize：沿 K 分 group（G=128），per (n, group) 一个 scale
      ▼
q (int8)          [N, K]          值域 [-8, 7]
      │  pack：沿 N 方向，每 8 个连续 n 压进 1 个 int32
      ▼
qweight (int32)   [K, N/8]        ← 存盘 / 上卡的就这个
scales (bf16)     [K/128, N]
```

## 4. 一个 int32 的内部结构

`qweight[k, nb]` 这 32 bit，装的**不是** 8 个 k，而是**同一个 k 上、连续 8 个 n**：

```
 bit:    31..28   27..24   23..20   19..16   15..12   11..8    7..4    3..0
 nibble:   7        6        5        4        3       2       1       0
 n:     8nb+7    8nb+6    8nb+5    8nb+4    8nb+3   8nb+2   8nb+1   8nb
```

- `nb = n // 8`，`i = n % 8`
- **低 4 bit 存较小的 n**（与 AWQ 的 pack 顺序一致）
- 负数用 4 bit 补码表示：`-1 → 0xF`，`-8 → 0x8`

## 5. 内存视图

```
                 nb=0          nb=1                 nb=N/8-1
 k=0        [n0..n7]      [n8..n15]     ...     [n_{N-8}..n_{N-1}]
 k=1        [n0..n7]      [n8..n15]     ...     [n_{N-8}..n_{N-1}]
  ⋮             ⋮              ⋮                        ⋮
 k=K-1      [n0..n7]      [n8..n15]     ...     [n_{N-8}..n_{N-1}]
```

同一行（同一 k）在内存里连续 → 一个 warp 读 32 个连续 word = 256 个输出通道，完美合并。

## 6. 索引公式（唯一权威）

```python
nb   = n // 8
i    = n % 8
word = qweight[k, nb]
q    = (word >> (4 * i)) & 0xF          # 无符号 nibble
q    = q - 16 if q >= 8 else q          # 转有符号
w    = q * scales[k // 128, n]          # 反量化

y[m, n] = sum_k x[m, k] * w             # 逻辑等价形式
```

## 7. 小例子（N=8, K=2, G=128，只有一个 group）

`k=0` 这一行，8 个输出通道的量化值：

```
q    = [ 1, -2,  3, -4,  5, -6,  7, -8]
补码 = [0x1, 0xE, 0x3, 0xC, 0x5, 0xA, 0x7, 0x8]

qweight[0, 0] = (0x8<<28)|(0x7<<24)|(0xA<<20)|(0x5<<16)|(0xC<<12)|(0x3<<8)|(0xE<<4)|0x1
              = 0x87A5C3E1
```

`k=1` 全 0 → `qweight[1, 0] = 0x00000000`。

若 `scales[0, :] = 0.5`，`x = [1.0, 0.0]`，则：

```
y[0, n] = 0.5 * q[n] = [0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0]
```

这个例子要写成单元测试（含 `-8` 这个边界值）。

## 8. 不变量（代码里必须断言）

```python
assert qweight.shape == (K, N // 8)
assert qweight.dtype == torch.int32
assert scales.shape == (K // 128, N)
assert K % 128 == 0 and N % 8 == 0
assert qweight.numel() * 4 == K * N // 2
```

## 9. 线程映射（由布局决定，不是自由选择）

```
block  → 覆盖 N_tile 个输出通道（如 1024）
thread → 负责 8 个输出通道（正好一个 int32 的全部 nibble）

for k in range(K):
    word = qweight[k, nb]              # 1 次 load，32 bit
    xk   = x[m, k]                     # 1 个标量，shared memory 广播
    for i in range(8):
        acc[i] += deq(nibble_i(word)) * xk
    if k % 128 == 0:
        s[0:8] = scales[k // 128, nb*8 : nb*8+8]   # 16B 连续 load，每 128 次 k 才一次
```

**关键约束**：一个线程必须消费整个 int32 的 8 个 nibble。若沿用「1 线程 1 输出通道」，一个 word 只用 1/8 → 权重读取放大 8 倍，有效带宽掉到 12.5%。

## 10. 决策记录

| 候选布局 | 打包维 | 结论 | 理由 |
|---|---|---|---|
| `[K, N/8]` | N | **采用** | 与 SGLang / AWQ checkpoint 同布局，可直接加载官方权重做基线 |
| `[K/8, N]` | K | 否决 | 是 GPTQ 的格式，与 AWQ ckpt 不兼容，混用会静默算错 |
| `[N, K/8]` | K | 否决 | 语义虽等价，但 N 跨步，tensor core 的 B tile 加载吃亏 |
| `[N/8, K]` | N | 否决 | AutoAWQ 的**存储**顺序，与 SGLang 参数互为转置，直接加载会维度错 |

> 注意：packing 开销与布局无关（scale 均占权重字节的 3.1%），
> 决定性能的是**线程映射**是否与布局匹配，而不是打包方向本身。

## 11. 与 SGLang 的对应关系

SGLang `awq.py` 里：

```python
qweight = PackedvLLMParameter(
    data=torch.empty(K, N // pack_factor, dtype=torch.int32),
    input_dim=0,        # dim 0 = K
    output_dim=1,       # dim 1 = N
    packed_dim=1,       # dim 1 已被压 8 倍，切分按 word 计
    packed_factor=8,
)
```

`packed_dim=1` 的语义：TP 列并行沿 N 切时，`N/8` 个 word 切一半 = `N/16` word = `N/2` 个输出通道，不能按逻辑元素算。

## 12. 常见 bug 定位表

| 现象 | 最可能的原因 |
|---|---|
| 结果整体差 8 倍 / 只有 1/8 通道有值 | 线程映射没改，仍是 1 线程 1 输出 |
| 只有部分通道错，且成组出现（8 个一组） | nibble 顺序反了，或 `n` 与 `nb, i` 换算错 |
| 负数位置全错、正数正常 | 忘了 `q - 16`，或 `& 0xF` 之前就做了符号扩展 |
| `unpack(pack(q)) != q` | pack 时 `.t()` 漏了，或 reshape 的维度顺序错 |
| 与 AWQ 基线对比时全错 | 把 `[N/8, K]` 当成 `[K, N/8]` 直接读入 |

## 13. 验证方法

1. **往返**：`unpack_int4(pack_int4(q))` 与 `q` 逐 bit 相等（含 `-8` / `7` 边界）
2. **小例子**：第 7 节的 `0x87A5C3E1` 手算结果要对上
3. **对官方**：取 AWQ ckpt 的一层，用 SGLang `awq_dequantize` 解出来，与自己的 `unpack + scale` 比对一致
4. **负向测试**：改一个 nibble，确认只有对应输出通道变化
