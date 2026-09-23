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
