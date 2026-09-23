# Laya 模型架构详解

> 探查对象：`models/laya-zh`（本文微调产物，mmBERT-base 编码器）与 `convaiinnovations/laya`（英文根检查点，ModernBERT-large 编码器）
> 方法：实际加载权重逐模块打印（`scripts/inspect_model.py`，可复现），参数量均为实测；形状追踪用真实中文样本。

---

## 1. 总体架构

Laya = **双向编码器骨干**（masked LM 预训练，决策任务上全参数微调）+ **从零训练的轻量决策头**。一切决策都在"给选项打分"的框架下完成，没有任何文本生成组件。

```
输入序列（一次构造，单次前向）:
[bos] choice question: <指令> [eos] [mask] 选项0 [mask] 选项1 … [eos] <state 文本> [eos]
      └──────── head 区（≤head_max_len）────────┘ └──── state 区（截断到 max_len）────┘

                                    ┌──────────────────────────────────────────┐
 tokens ──► ① 编码器 mmBERT-base ──►│ 22 层 ModernBERT（全局/滑动注意力交替）  │──► H: (B, L, 768)
            306.94M                 └──────────────────────────────────────────┘
                                        │
             ② + type_emb(题型 3 选 1)  │  H' = H + type_emb[choice/score/noul]（查表广播到每个 token）
                                        ▼
             ③ 决策头 Transformer ────► 2 层 nn.TransformerEncoderLayer（pre-norm, d=768, 12 头, FFN 3072）
                14.18M                   保持 (B, L, 768)
                                        ▼
             ④ gather 选项 marker ────► 在每个选项的 [mask] 位置取向量：M = (B, K, 768)，K=选项数
                                        ▼
             ⑤ scorer ──────────────► LayerNorm → Linear(768,768) → GELU → Linear(768,1)
                0.59M                  每个选项一个标量 logit → (B, K)
                                        ▼
             ⑥ softmax over K 个选项（读出时除以温度）→ 概率分布 / 期望分 / P(true)
                ＋
             ⑦ act_head ────────────► [CLS] 池化(768) ⊕ 4 统计特征 → Linear(772,256) → GELU → Linear(256,2)
                0.20M                  输出 2 类：act（直接采纳）/ escalate（升级给 LLM，成本 0.5）

总参数 321.91M = encoder 306.94M + head 14.18M + scorer 0.59M + act_head 0.20M + type_emb 2.3K
```

三个题型共用同一套网络，只是序列里 head 区的 `<题型>` 前缀和选项渲染方式不同（choice：`标签: 描述`；score：`level i: 描述`；noul：固定 `false: …` / `true: …` 两项）。**K 是运行时变量**，因此换 schema 不需要重训。

## 2. 编码器逐层结构（mmBERT-base，即 ModernBERT 架构）

`encoder/config.json` 实测：hidden **768**，**22 层**，**12 头**（每头 64 维），FFN 有效中间维 **1152**（门控，见下），词表 **256 000**（Gemma 分词器），最大位置 **8192**，**无绝对位置嵌入**（RoPE，theta=160000），**全无偏置**（attention/mlp/LayerNorm 的 bias 均为 False）。

### 2.1 嵌入层（196.61M，占总参数 61%）

```
tok_embeddings  Embedding(256000, 768)   196.60M   ← 巨大词表的代价；与 MLM decoder 权重共享
norm            LayerNorm(768)           768       ← 进第 0 层前的最终归一化
drop            Dropout(0.0)
```

（这就是中文微调时选择冻结 embedding 的原因：它独占六成参数与一半以上优化器显存，且 mmBERT 预训练已包含中文。）

### 2.2 22 个 Transformer 层（110.33M，平均 5.02M/层）

注意力模式**每 3 层一个全局、两个滑动窗口（window=128）交替**，RoPE 对两种模式都生效（实测 layer_types 序列）：

| layer | 0 | 1 | 2 | 3 | 4 | 5 | … | 21 |
|---|---|---|---|---|---|---|---|---|
| 类型 | 全局 | 滑动128 | 滑动128 | 全局 | 滑动 | 滑动 | … | 全局 |

每个层的内部（以实测 layer 1 滑动层 / layer 0 全局层为例，pre-norm 残差结构）：

```
x = x + attn( attn_norm(x) )          attn_norm: LayerNorm（第 0 层为 Identity，因嵌入层末尾已有 norm）
x = x + mlp ( mlp_norm(x) )           mlp_norm : LayerNorm

attn  ModernBertAttention (5.02M 中的 2.36M)
  Wqkv  Linear(768 → 2304)  no bias   Q/K/V 融合为单矩阵（3×768=2304），12 头 × 64 维
  Wo    Linear(768 → 768)    no bias
  （sliding 层用窗口 128 的局部注意力 + flash 实现，这是 mmBERT 能开到 8k 上下文的原因）

mlp   ModernBertMLP (5.02M 中的 2.66M)
  Wi    Linear(768 → 2304)  no bias   输出按维切成 value(1152) 与 gate(1152) 两块
  act   GiLU：act(gate) * value（act 为 GELU）→ 有效中间维 1152
  Wo    Linear(1152 → 768)  no bias

encoder 末尾 final_norm: LayerNorm(768)
```

> 注意与标准 BERT 的差异：RoPE 代替绝对位置嵌入、注意力 3:2 全局/滑动交替、门控 GiLU-MLP、全无偏置、无 NSP 头——这些继承了 ModernBERT 的现代化设计，mmBERT 只是把词表换成 Gemma 的 256k 并在多语言语料上重训。

## 3. 决策头（从零训练，14.98M）

### 3.1 type_emb（2 304 参数）
`Embedding(3, 768)`，choice/score/noul 各一行，加到**每个 token** 的隐向量上，让同一个决策头区分题型。

### 3.2 head（14.18M）
`nn.TransformerEncoder` 含 **2 层** `nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1, norm_first=True, batch_first=True)`（每层 7.09M：自注意力 in_proj 768→2304 + out_proj 768→768 + FFN 768↔3072）。作用：让各选项 marker 之间、以及 marker 与 state 之间再做两轮交互后打分。

### 3.3 scorer（0.59M）——每个选项一个 logit
```
LayerNorm(768) → Linear(768,768) → GELU → Linear(768,1)
```
只作用于 gather 出的 K 个选项向量，输出 K 个标量；**对该问题的 K 个选项做 softmax**（不是全词表），读出时再除以按"题型×选项数分桶"的温度。

### 3.4 act_head（0.20M）——要不要升级
```
输入 = [CLS] 位置池化向量(768) ⊕ 4 维统计特征 → Linear(772,256) → GELU → Linear(256,2)
4 维特征 = [top1 概率, top1−top2 差, 归一化熵, K/255]
输出 2 类：act / escalate（config 中 act_costs={'escalate': 0.5}，错误升级有不对称成本）
```

### 3.5 温度不在网络里
`temperature` 是 `rl_agent_config.json` 里的读出期超参（英文检查点还带 `choice:2 / 3-5 / 6-10 / 11+` 等分桶），推理时在 softmax 前应用；模型内部的 temperature buffer 只是占位（恒 1.0）。微调后我们拟合得到 choice 1.041 / score 1.158 / noul 1.145。

## 4. 前向传播实测（真实样本，B=1）

样本：`"快递显示派送了但没送到…请帮我重新安排派送。"` + 5 选项 choice 问题

| 步骤 | 张量形状 | 备注 |
|---|---|---|
| 序列构造 | 77 tokens | `[bos] choice question: 判断…[eos] [mask]×5选项 [eos] state [eos]`；marker 位于 16/28/33/38/43 |
| ① 编码器输出 | (1, 77, 768) | 22 层 ModernBERT |
| ② + type_emb | (1, 77, 768) | choice 行广播相加 |
| ③ 决策头 2 层 | (1, 77, 768) | pre-norm，pad 位置被 mask |
| ④ gather markers | (1, 5, 768) | 每选项一个 768 维向量 |
| ⑤ scorer | (1, 5) | 实测 logits [−3.67, **1.10**, −3.68, −3.66, −3.42] |
| ⑥ softmax | 5 个概率 | 实测 **logistics 0.965**，其余 ≈0.01（与微调目标一致） |
| ⑦ act_head | (1, 2) | act / escalate |

## 5. 三个检查点的编码器对比（实测）

| | laya（英文根） | laya/multilingual | laya-typed-decisions |
|---|---|---|---|
| 编码器 | ModernBERT-**large** | mmBERT-**base**（同 ModernBERT 架构） | ModernBERT-large |
| hidden / 层数 / 头 | 1024 / 28 / 16 | 768 / 22 / 12 | 1024 / 28 / 16 |
| FFN 有效中间维 | 2624（门控） | 1152（门控） | 2624 |
| 词表 | 50 368 | 256 000 | 50 368 |
| 注意力 | 每 3 层全局+窗口128 滑动 | 同左 | 同左 |
| RoPE / max_pos | 是 / 8192 | 是 / 8192 | 是 / 8192 |
| 总参数（含决策头） | 421M（模型卡） | 321.91M（实测） | 421M（模型卡） |
| max_len / head_max_len | 512 / 192 | 1024 / 256 | 1024 / 256 |

决策头三个检查点完全同构（差异只在 d=768 与 1024 时的宽度）。

## 6. 设计要点小结

1. **答案空间与参数解耦**：选项在 [mask] 标记上打分，K 任意，网络权重固定——新增中文 schema 零成本。
2. **编码器复用成熟预训练 + 头从零学"决策"**：头只占 ~4.7% 参数，RLCD 的策略梯度主要塑形 scorer 与头，编码器以小学习率适配。
3. **置信度是模型输出的一部分**：softmax 分布经熵归一化得到 confidence，配合 act_head 的 escalate——整套设计围绕"可门控的概率"而非 argmax。
4. **参数大头是词表嵌入**（多语版 61%）：这是 mmBERT 多语言能力的载体，也是我们在 16GB Mac 上微调时冻结它的原因（实测冻结后可训练参数仅 125.3M，训练照常收敛）。

复现：`python scripts/inspect_model.py [模型路径]`
