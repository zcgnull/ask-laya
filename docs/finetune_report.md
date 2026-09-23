# Laya 中文微调报告（RLCD）

> 目标：验证"微调是 laya 发挥价值的方式"（模型卡原话），并实际提升中文决策效果。
> 方法：完整复现官方 RLCD 训练循环（`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`），在自建中文客服数据上微调 `multilingual` 检查点。
> 训练代码：`scripts/finetune_cn.py` ｜ 产出模型：`models/laya-zh/` ｜ 评估代码：`scripts/eval_cn.py`

---

## 1. 数据

自建中文客服工单数据（`scripts/build_cn_dataset.py` 生成，格式与 `LocalLLaMA/typed-decisions` 一致：state + questions + gold 软目标分布）：

| 数据集 | 规模 | 用途 |
|---|---|---|
| `data/cn_train.jsonl` | 1200 条（3600 决策） | 训练 + 温度校准（校准取训练集切片，与测试无交集） |
| `data/cn_test.jsonl` | 260 条（780 决策） | 模板内分布测试 |
| `data/cn_test_hard.jsonl` | 80 条（240 决策） | **手写自然句**，分布外泛化测试（含口语化、复合诉求、边界案例） |

三个任务：intent（choice，5 类）、urgency（score，4 级）、churn（noul）。生成时让 urgency/churn 与 intent 仅部分相关，避免标签捷径。

## 2. 训练配置

沿用官方超参，针对 Apple M4 16GB（MPS，fp32，无 CUDA）适配：

| 项 | 官方（2×T4） | 本项目（M4 MPS） |
|---|---|---|
| 检查点 | laya（ModernBERT-large 421M） | multilingual（mmBERT-base 322M） |
| 算法 | RLCD：G=4 零均值高斯探索（σ 0.4→0.1）、proper scoring rule 奖励（log + 0.75·spherical + 1.0·RPS）、组内归一化 advantage、高斯策略梯度 + 1.0×软标签 CE 指导 | 完全相同 |
| 学习率 | 编码器 2.5e-5 / 头 1e-4，AdamW，余弦退火，裁剪 1.0 | 相同 |
| 批量 | 8×4 累积×2 卡 | 8×2 累积（单卡） |
| 轮数 | 4 | 3 |
| 显存优化 | 编码器梯度检查点 + fp16 | 梯度检查点 + **冻结 embedding**（mmBERT 词表 256k ≈ 197M 参数冻结，可训练 125.3M / 共 321.9M）+ fp32 |
| 训练时长 | ~4-5 小时 | **951 秒（约 16 分钟）** |

训练后按官方流程用 LBFGS 对三题型分别拟合温度（校准集来自训练集切片）：choice 1.041 / score 1.158 / noul 1.145，写入 `rl_agent_config.json`。

## 3. 结果

### 3.1 模板内分布测试（cn_test，260 例 / 780 决策）

| 题型 | 微调前 | 微调后 | 变化 |
|---|---|---|---|
| choice | 71.2% | **100.0%** | +28.8pp |
| score | 23.9% | **74.6%** | +50.7pp |
| noul | 33.5% | **96.9%** | +63.4pp |
| **整体准确率** | **42.8%** | **90.5%** | **+47.7pp** |
| **ECE（校准误差）** | 0.280 | **0.089** | **−68%** |
| Brier | 0.62* | 0.11* | — |

### 3.2 分布外泛化（cn_test_hard，手写 80 例 / 240 决策）

| 题型 | 微调前 | 微调后 | 变化 |
|---|---|---|---|
| choice | 56.3% | **66.3%** | +10.0pp |
| score | 26.3% | 22.5% | −3.8pp |
| noul | 42.5% | **77.5%** | +35.0pp |
| **整体准确率** | **41.7%** | **55.4%** | **+13.7pp** |

### 3.3 定性对比（微调前后同输入）

| 输入（摘录） | gold | 基座 urgency / churn | 微调后 urgency / churn |
|---|---|---|---|
| "快递显示派送了但没送到…我腿脚不方便，请帮我重新安排派送" | 1 / false | 2.52 / 0.64 | **1.67 / 0.06** |
| "手表洗澡能不能戴" | 0 / false | 1.22 / 0.21 | **1.04 / 0.05** |
| "退款逾期十天…再不处理我就向消协投诉"（复合诉求） | 3 / true | 2.70 / 0.99 | 1.76 / 0.94（churn 对，urgency 偏保守） |

### 3.4 发现与诚实结论

1. **分布内大幅提升且校准修复**：整体 42.8%→90.5%，ECE 0.28→0.089（已达到模型卡"温度拟合后 ECE≈0.08"的水准）。score 从 23.9%（≈随机）到 74.6%，noul 从"反向"（33.5%）到 96.9%——**中文短板题型被补齐**。
2. **分布外提升真实但有限**：手写自然句 +13.7pp。noul 泛化最好（+35pp），choice 中等（+10pp）。
3. **score 在分布外没有提升**（26.3%→22.5%）：模板生成的紧急度线索（关键词）在自然句中形态多变，序数任务又恰是 laya 最弱原语；要泛化需要更大规模、更贴近真实分布的中文语料（数千条真实工单 + 教师软标签）。
4. 边界案例暴露标注歧义（如"投诉退款不到位"既是 complaint 又是 refund），复合诉求建议在业务上拆成多问题而非单选题。
5. 推理速度不变（微调只改权重，不改架构）：16.5 ms/决策（MPS）。

## 4. 复现

```bash
pip install laya
python scripts/build_cn_dataset.py                       # 生成中文数据
python scripts/eval_cn.py --model convaiinnovations/laya \
    --subfolder multilingual --tag ml-base               # 基线
python scripts/finetune_cn.py --epochs 3 --freeze-embeddings   # RLCD 微调（M4 约 16 分钟）
python scripts/eval_cn.py --model models/laya-zh --tag zh-finetuned  # 微调后评估
```

有 CUDA GPU 时去掉 `--freeze-embeddings`、调大 `--micro-batch`，可完全对齐官方配置。

原始评估 JSON：`results/eval_ml-base.json`、`results/eval_zh-finetuned.json`、`results/eval_en-base-on-cn.json`、`results/verify_results.json`；训练日志：`models/finetune.log`。


---

## 5. 官方中文基准迭代（feishu_zh，2026-09 追加）

laya 上游新增官方中文基准 `research/benchmarks/feishu_zh`（64 条飞书式职场消息，8 错误家族 × 4 均衡标签，
choice 四分类 + 4 个是非题按规则解码，冻结 prompt，3 repeats 取首次）。我们用它作为"官方评测方法"对
各版本做了完整对比。

### 5.1 定向语料

`scripts/build_feishu_train.py`：按基准文档描述的 8 个错误家族（ownership/lifecycle/urgency/knowledge/
conditional/untrusted_content/thread_context/cross_chat）**全新生成**同分布训练场景（飞书式多群消息、
不同人物/角色/群名，不复制任何评测 case），prompt 直接使用官方冻结版。生成后按 4 标签均衡重采样。

### 5.2 迭代结果（官方协议：64 case × 2 模式 × 3 repeats，取首次）

| 版本 | 训练数据 | choice 准确率 | four_noul 准确率 | choice macro-F1 |
|---|---|---|---|---|
| 基座 multilingual（零样本） | — | 31.2% | 28.1% | 0.257 |
| V1 = laya-zh（客服模板数据） | 1200 客服行 | 37.5% | 42.2% | 0.336 |
| **V2 = laya-zh-v2（最优）** | 客服 1200 行 + 分诊 700 cases | **51.6%** | 39.1% | **0.482** |
| V3 = laya-zh-v3 | 客服 1200 行 + 分诊 1200 cases（新种子） | 35.9% | 26.6% | 0.309 |

（官方档案记录的基座成绩为 choice 20/64、four_noul 18/64，与我们复测的 31.2%/28.1% 同为接近随机的水平。）

其他指标：V2 的误报动作仅 **1/32**（基座 22/32）、four_noul 误报动作 **0/32**；自建集无回退（整体 90.9%）。

### 5.3 迭代发现

1. **定向合成语料是最大杠杆**：V1→V2 仅增加 700 个同分布场景，choice +14pp。
2. **同模板数据翻倍反而伤害泛化**：V3 把 cases 从 700 翻倍到 1200（同模板、新种子），choice 掉到 35.9%——
   模板表面记忆淹没了语义学习。后续提升应换"更真实的语料源"（如脱敏真实 IM 语料或 LLM 生成的多样表述），
   而不是加量同模板。
3. **four_noul 是剩余短板**：解码规则要求 related/action≥0.5 且 urgent≥0.75 才判 urgent，V2 漏动作 25-28/32
   主要卡在 urgent 阈值。可选手段：按 four_noul 的解码阈值做阈值级校准（而非温度），或增大 urgent 密度。
4. 训练资源：V2 用 `--micro-batch 4 --grad-accum 4 --max-len 512 --threads 4`（内存受限参数，M4 16GB 稳定
   运行约 70 分钟）；微调期间建议停掉本机 webapp 服务以让出统一内存。
