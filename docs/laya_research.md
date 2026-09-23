# Laya 模型研究报告

> 研究对象：<https://www.modelscope.cn/models/convaiinnovations/laya>（同权重发布于 Hugging Face `convaiinnovations/laya`）
> 研究日期：2026-09-22 ｜ 研究方式：模型卡（ModelScope/HF）+ GitHub 仓库 + PyPI SDK 源码（laya 0.3.5）逆向阅读 + 本地实测

---

## 1. 一句话总结

**Laya 是一个"系统一（System 1）"决策引擎**：输入一段文本状态（state：邮件、工单、JSON、对话）和一组**类型化问题**（choice / score / noul），**单次前向传播**（非自回归、不生成文本）直接输出每个选项的**带校准概率**的决策结果。它不是 LLM 的替代品，而是给 LLM Agent 工作流当"快速、可置信度门控的决策中枢"。

核心卖点：

| 特性 | 说明 |
|---|---|
| 快 | 单次前向 ≈ 33 ms（T4 GPU），输出 token 数为 0 |
| 不会幻觉 | 不生成任何文本，只输出类型化决策 + 概率 |
| 概率可校准 | 用严格正则评分规则（proper scoring rules）做强化学习训练，置信度有统计含义 |
| 多语言 | multilingual 检查点支持 100+ 语言（含中文） |
| 答案空间运行时可变 | 选项打分在 [MASK] 标记上进行，新增 schema 无需重训 |

---

## 2. 开发者与许可证

- 开发者：**Convai Innovations**（ModelScope / HF 组织 `convaiinnovations`，GitHub 仓库 `NandhaKishorM/laya`）
- 许可证：Apache 2.0
- 生态：PyPI 包 `laya`（当前 0.3.5），HF 数据集 `LocalLLaMA/typed-decisions`，官方微调 notebook（Kaggle 2×T4 可跑）

## 3. 检查点家族

| 检查点 | 编码器 | 参数量 | 上下文 | 定位 |
|---|---|---|---|---|
| `convaiinnovations/laya`（根目录） | ModernBERT-large | 421M | 512 | 英语最强 |
| `convaiinnovations/laya` 子目录 `multilingual`（= HF `laya-multilingual` / ModelScope `laya-multilingual`） | mmBERT-base | 322M | 1024（mmBERT 架构可到 8k） | 100+ 语言，速度约 2× |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1024 | 在 typed-decisions 基准上微调过（0.766 acc） |

配套 **Router**（路由器）：在前向传播之前用亚毫秒级的文字/语种检测选择检查点。这一步是必须的——英文检查点在非拉丁文字上"自信地错误"（高棉语：准确率 0.000 但置信度 0.952）。

## 4. 架构细节（源自 SDK 源码 `laya/common.py`）

```
[CLS] <问题类型> question: <指令> [SEP] [MASK] 选项0 [MASK] 选项1 … [SEP] <state 文本> [SEP]
```

- **DecisionModel** = 编码器（ModernBERT / mmBERT）+ 2 层 Transformer 决策头（`norm_first`）+ 类型嵌入（choice/score/noul 三类）+ **选项打分器**（每个选项的 [MASK] 位置各出一个 logit，对该问题的选项集合做 softmax）+ act/escalate 头（4 维统计特征 + [CLS] 池化 → 2 类，用于"要不要升级给 LLM"）。
- 关键机制：**每个选项在自己的 [MASK] token 上打分**，所以答案空间在请求时定义，换 schema 不用重训；每个选项文本截断 48 token，head（指令+选项）总预算 `head_max_len`（英语 192 / 多语 256），整条序列截断 `max_len`（512 / 1024）。
- **温度校准**：检查点携带按"题型 × 选项数分桶"（`choice:2`、`choice:3-5`、`choice:6-10`、`choice:11+`、`score:*`、`noul:*`）拟合的温度；SDK 限制温度 ∈ [0.5, 5.0]，超界会 clamp 并告警（官方某个桶拟合出 0.1006，会把 0.24 的概率放大成 0.99，被 SDK 拒绝执行）。
- 置信度定义：`1 − H(p)/log(k)`（归一化香农熵），noul 的置信度即 max(P(true), 1−P(true))。

## 5. 训练方法：RLCD（Reinforcement Learning for Calibrated Decisions）

从官方微调 notebook（`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`）逆向出的完整算法：

1. **策略**：模型对每个问题输出分布 q（选项 logit 的 softmax）。
2. **探索**：给 logit 加零均值高斯噪声（σ 从 0.4 余弦退火到 0.1），并在选项维度上投影去均值；每组采 G=4 个噪声样本。
3. **奖励**：严格正则评分规则组合 —— log score + 0.75×spherical score（对软标签友好），score 题型额外减 1.0×RPS（Ranked Probability Score，序数感知）。
4. **策略梯度**：组内减均值除标准差（GRPO 式 advantage），高斯 log-likelihood 加权求和；同时混合权重 1.0 的软标签交叉熵"指导损失"。
5. **优化**：编码器 lr=2.5e-5、决策头 lr=1e-4，AdamW + 余弦退火，梯度裁剪 1.0，梯度累积，编码器开梯度检查点。
6. **后处理**：训练后用 LBFGS 对三种题型分别拟合温度（最小化 NLL），写入 `rl_agent_config.json`。
7. 多轮对话轨迹用 TD(λ=1.0) 把最终标签回传到前缀切片（`td_lambda_targets`）。

**为什么概率是校准的**：直接最小化 log score 等严格正则评分规则的最优解就是真实条件概率，RL 奖励即评分规则，因此收敛目标天然是校准概率，而非只优化 argmax 的分类损失。

## 6. 已发布基准（模型卡自述）

- MASSIVE 意图（51 语言）、XNLI、AG News、BoolQ、DAIR Emotion、SST-5、Banking77、prompt-injection 集（n=116）。
- typed-decisions 基准（400 案例 / 2000 决策，四个工作流）：发票处理 0.804、安全事件 0.766、客服 0.764、agent-trace 可观测性 0.730。
- 校准：英语检查点出厂 ECE 0.466，温度拟合后 0.081（"ship over-confident"是官方承认的问题）。
- **基座检查点在 typed-decisions 零样本上接近随机（0.36 vs 随机 0.318）**——官方明说"微调才有价值"，微调后 0.766（高于教师自洽上限 0.735）。

## 7. 已知局限（模型卡自述 + 源码佐证）

1. **高基数 choice 弱**：>20 个选项时退化（Banking77 上 0.425，对手 Jev 0.870）；解法：调大 `head_max_len` 或用 `predict_shortlist` 先嵌入召回 top-k。
2. **序数 score 最弱**：SST-5 只有 0.372。
3. **出厂过自信**：必须做温度校准后才可放心做置信度门控。
4. **英文检查点在非英文上崩溃**（且自信）；multilingual 检查点在英语上又偏弱 → 必须用 Router 或显式选择。
5. 上下文窗口有限（512/1024 token），长文档需先摘要或切片。

## 8. 使用方式

```bash
pip install laya
```

```python
import laya
from laya import Router

# 直连（推荐按语言手动选检查点）
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
result = agent.predict(state, questions)   # 所有问题一次前向

# 或路由器自动选择
router = Router(preload=True)
res = router.predict(state, questions)

# 工作流预设
laya.triage_questions() / laya.guard_questions() / laya.moderation_questions() / laya.router_questions()
```

问题 schema（三种题型）：

```python
questions = {
  "intent":  {"type": "choice",
              "instructions": "Classify the customer's primary intent.",
              "criteria": {"refund": "wants money back", "shipping": "delivery issue"}},
  "urgency": {"type": "score",
              "instructions": "Rate urgency 0-3.",
              "criteria": ["calm", "mild", "urgent", "critical"]},
  "phishing":{"type": "noul",
              "instructions": "Is this message a phishing attempt?"},   # P(true)
}
```

返回：`answers[qid] = {choice|score|noul, probabilities, confidence, action.act_probability}`，`usage.output_tokens` 恒为 0。

## 9. 与本项目相关的验证与微调结论（见其余文档）

- 验证脚本与结果：`docs/verification_report.md`（英文复现 + 中文基线 + 延迟/校准实测）
- 中文微调：`docs/finetune_report.md`（复现官方 RLCD 循环 + 自建中文数据 + 前后对比）
- 代码：`scripts/verify_laya.py`、`scripts/build_cn_dataset.py`、`scripts/finetune_cn.py`、`scripts/eval_cn.py`

## 10. 参考链接

- ModelScope：`convaiinnovations/laya`、`convaiinnovations/laya-multilingual`、`convaiinnovations/laya-typed-decisions`
- Hugging Face：`convaiinnovations/laya`（含 multilingual / typed-decisions 子目录）
- GitHub：`NandhaKishorM/laya`（SDK 源码、微调 notebook）
- PyPI：`laya`（0.3.5）
- 数据集：HF `LocalLLaMA/typed-decisions`
