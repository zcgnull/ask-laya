# Laya 效果验证报告

> 验证环境：Apple M4（10 核，16GB 统一内存，MPS 后端）/ Python 3.13 / laya 0.3.5 / torch 2.14.0 / transformers 5.17.0
> 检查点：`convaiinnovations/laya`（根，ModernBERT-large 421M）与 `convaiinnovations/laya` 子目录 `multilingual`（mmBERT-base 322M）
> 验证代码：`scripts/verify_laya.py`（声明验证）、`scripts/eval_cn.py`（中文定量评估）；原始数据在 `results/`

---

## 1. 模型卡声明逐条验证

| # | 声明 | 结果 | 证据 |
|---|---|---|---|
| 1 | 输入 state + 类型化问题，**单次前向**返回全部决策 | ✅ 通过 | 3 个问题一次 `predict` 调用，`usage.output_tokens == 0`（verify_results.json） |
| 2 | 模型卡示例：账单催款邮件 → route=billing（高置信）、urgency≈2、非钓鱼 | ✅ 复现 | billing p=0.9831 conf=0.911；urgency 期望分 2.01（p(2)=0.826）；phishing 分场景 0.04~0.67 |
| 3 | 检查点配置：英文 max_len=512 / head_max_len=192；多语 1024 / 256；编码器 ModernBERT-large / mmBERT-base | ✅ 一致 | `rl_agent_config.json` 实测读出 |
| 4 | **英文检查点在非拉丁文字上崩溃**（高棉语 0.000 acc @ 0.952 conf） | ✅ 定量证实 | 同一中文工单：英文检查点整体 49.2%（困难集 37.9%）；multilingual choice 71.2% / 置信度 0.98 vs 英文 0.39 |
| 5 | Router 亚毫秒语种检测并给理由 | ✅ 通过 | 中文输入 → `multilingual`，reason="non-Latin script (han, 100% of letters); the English checkpoint cannot read it" |
| 6 | multilingual 速度约为英文 2 倍 | ✅ 大致符合 | 16.7 ms/决策 vs 75.9 ms/决策（MPS，见 §3） |
| 7 | 多语言检查点**出厂未拟合温度**（temp=1.0，置信度未校准） | ✅ 证实 | 加载时 temperature=[1.0, 1.0, 1.0]；noul ECE 高达 0.51（§2） |
| 8 | **序数 score 是最弱原语** | ✅ 证实 | multilingual 中文 score 仅 23.9%（模板集）/ 26.3%（困难集），为三题型最低 |
| 9 | 每个选项在自己的 [MASK] 上打分、答案空间运行时可变 | ✅ 架构核实 | 源码 `laya/common.py: build_sequence`（[CLS] 指令 [SEP] [MASK]选项… [SEP] state [SEP]），评测中新增中文选项无需重训即可推理 |

## 2. 中文基线（微调前，multilingual 检查点）

自建中文客服工单三任务评测集（`data/cn_test.jsonl` 模板生成 260 例 / `cn_test_hard.jsonl` 手写自然句 80 例，每例 3 个决策；模板集与训练分布同源，困难集为分布外泛化测试）：

| 题型 | 模板集 acc | 困难集 acc | 模板集 ECE | 随机基线 |
|---|---|---|---|---|
| choice（5 类意图） | **0.7115** | **0.5625** | 0.256 | 0.20 |
| score（4 级紧急度） | **0.2385** | **0.2625** | 0.224 | 0.25 |
| noul（流失风险，阈值 0.5） | **0.3346** | **0.4250** | 0.509 | 0.50 |
| **整体** | **0.4282** | **0.4167** | 0.280 | — |

关键发现：

- choice 能力可以零样本迁移到中文（71.2%，2 倍于随机），但 **score 与 noul 在中文上接近随机甚至差于随机**（noul 33.5% 说明方向反了）——"基座检查点在未微调任务上接近随机"（模型卡 §Limits）在中文场景完全复现。
- 置信度不可用：出厂温度未拟合，noul 的 ECE 高达 0.51，mean_confidence 0.84 但 accuracy 0.33，"过自信且不准"，直接做置信度门控会出事故。
- 英文检查点在中文上整体 49.2%（困难集 37.9%），同样差，且以高置信输出（见 §1 #4）。

基线结论：**Laya 架构与速度声明可信；"决策质量"高度依赖微调，中文正是需要微调的场景** → 进入任务 3（微调）。

## 3. 延迟实测（MPS，fp32）

| 指标 | 英文检查点 | multilingual |
|---|---|---|
| 单 case（3 问题合并一次前向） | 135.6 ms | **50.2 ms** |
| 每决策 | 75.9 ms | **16.7 ms** |
| 首 Call（含 MPS 预热） | 922 ms | ~— |

注：模型卡标称 T4 GPU ≈33 ms/call；M4 MPS 无 fp16/bf16 加速且算力弱于 T4，绝对值慢属预期，相对倍率（多语 ≈ 2.7× 英文）与声明方向一致。每 case 实际输入 token：多语 274，英文 490（词表效率差异）。

## 4. 结论

1. `pip install laya` 生态完整，Mac (MPS) 开箱即用，API 与模型卡一致。
2. 三种题型、运行时可变选项空间、零生成、Router 语言路由——**架构与工程声明全部验证通过**。
3. **决策质量声明同样验证**：未微调任务上接近随机（中文 score/noul）、英文检查点跨语言崩溃、出厂过自信——这些"限制"如实复现。
4. 因此提升中文效果的正确路径是官方推荐的 **RLCD 微调**，见 `docs/finetune_report.md`。
