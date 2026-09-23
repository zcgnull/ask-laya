# 问问 Laya · Ask Laya

> 围绕 [convaiinnovations/laya](https://www.modelscope.cn/models/convaiinnovations/laya) 的完整研究项目：
> **研究文档 → 声明验证 → 中文 RLCD 微调 → 可部署的全栈 Web 应用**
>
> `[English below](#english)`

Laya 是一个"系统一（System 1）"非自回归决策引擎：输入一段状态文本（state）和若干**类型化问题**（choice / score / noul），单次前向传播输出全部决策的**校准概率**——不生成文本、不会幻觉、约几十毫秒。本项目把它从模型卡一路带到了能用的产品：

| 交付物 | 说明 |
|---|---|
| 📚 研究文档 × 4 | 模型研究 / 架构详解 / 声明验证 / 微调报告（含实测数据） |
| 🔬 验证脚本 | 模型卡 9 项声明逐条实测，中英文基线与延迟 |
| 🇨🇳 中文微调 | 复现官方 RLCD 训练循环，中文准确率 **42.8% → 90.5%**，ECE **0.28 → 0.089** |
| 🌐 Web 应用 | 复古风决策网站：生活裁决 + 六大工作流场景实验室 + 自定义 schema + Jev 兼容 API |
| ☁️ 一键部署 | ModelScope 创空间（在线 Demo）+ frp 公网 + 本地三套运行形态 |

**在线 Demo**：<https://modelscope.cn/studios/zcgnull/ask-laya>（公开，免费 CPU 档）

---

## 目录

- [快速开始](#快速开始)
- [Web 应用](#web-应用)
- [关键结果](#关键结果)
- [项目结构](#项目结构)
- [文档](#文档)
- [架构速览](#架构速览)
- [已知限制](#已知限制)
- [致谢与许可](#致谢与许可)

## 快速开始

```bash
git clone https://github.com/zcgnull/ask-laya.git && cd ask-laya
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1) 复现中文微调（Apple M4 约 16 分钟，CUDA 更快）
.venv/bin/python scripts/build_cn_dataset.py
.venv/bin/python scripts/finetune_cn.py --epochs 3 --freeze-embeddings

# 2) 启动网页（生活裁决 + 场景实验室）
.venv/bin/python webapp/server.py            # → http://127.0.0.1:8123

# 3) 评估任意检查点
.venv/bin/python scripts/eval_cn.py --model models/laya-zh --tag zh-finetuned
```

依赖：Python ≥ 3.10，`requirements.txt`（torch / transformers / laya / fastapi 等；Apple Silicon 走 MPS，CUDA 可选）。

## Web 应用

参考 [kuhung/ask-jev](https://github.com/kuhung/ask-jev) 的简约实验，后端是**真实的 laya 模型本地推理**——无 API Key、无广告、无追踪。

### 🎮 生活裁决

| 模式 | laya 题型 | 输出 |
|---|---|---|
| 该不该？ | noul | P(值得做) + 执行/劝退裁决 |
| 选哪个？ | choice | 按中文「还是/或者/、」与英文 `or/vs` 切分候选，逐选项概率条 |
| 冲动指数 | score | 4 级期望分 → 1-10 冲动指数 |

- 🔥 九个热门场景快捷入口（买不买 / 吃不吃 / 去不去 / 说不说 / 辞职 / 旅行 / 养宠物 / 报班考证 / 断舍离），12 组场景化人设文案（中英双语）
- **中英自适应**：`laya.lang.is_english` 检测语言，英文问题自动切换英文 prompt 与裁决文案
- 智能纠错：「A or B」是非题自动升级为选择模式；事实型问句自动平行化选项（"Is Beijing in the United States or in China?" 修复后正确率 78%）

### 🧪 场景实验室（对齐官方 [laya-demo](https://modelscope.cn/studios/AI-ModelScope/laya-demo)）

六大生产工作流 + 自定义 schema，state 可编辑，**所有问题在同一次前向里并行作答**：

| 场景 | schema 来源 |
|---|---|
| 🎫 客服工单分诊 / 📧 邮件+钓鱼 / 🛡️ LLM 护栏 / 🧹 内容审核 / 🧭 模型路由 | `laya.presets` 官方预设 |
| 📄 RAG 段落过滤 / 🛠️ 自定义 | 自建（是非 / 评分 / 选择任意组合，≤12 问，零训练换 schema） |

每个答案卡展示：中文问题名、题型徽章、主判定（跟随大概率一侧）、完整概率分布、置信度与**直答门控值**（act head 输出，"阈值由你定，不是模型说了算"）。

### ☁️ 部署形态

| 形态 | 地址 / 命令 | 说明 |
|---|---|---|
| ModelScope 创空间 | [zcgnull/ask-laya](https://modelscope.cn/studios/zcgnull/ask-laya) | 免费 CPU 档，`deploy_studio.sh` 一键部署/更新 |
| 本地 | `python webapp/server.py` | 完整体验，MPS/CPU |
| frp 公网 | `http://39.98.88.195:18081` | 本机在线时可用 |

**Jev 兼容 SystemOne API**（laya 0.3.7 衍生功能）：`POST /v1/systemone` 接受 TypeSafe Jev 线协议 `{state, questions, model?}`。把 ask-jev 官网「TypeSafe 接口设置」的 endpoint 填成本站地址，即可用真实 laya 模型驱动官方站点；hs-jev 等 Jev 客户端即插即用。

## 关键结果

### 声明验证（9/9 通过，含如实复现的"限制"）

工程声明全部属实：单次前向 0 输出 token、Router 语言路由（中文自动路由 multilingual）、[MASK] 选项打分运行时可变、多语言版速度约 2×。同时**如实复现了模型卡承认的三条限制**：

- 基座检查点在未微调任务上接近随机：multilingual 中文零样本整体 **42.8%**（score 23.9% ≈ 随机，noul 33.5% 反向）
- 英文检查点跨语言崩溃：同一批中文数据整体仅 **49.2%**
- 出厂过自信：multilingual 未拟合温度，noul ECE **0.51**

### 中文 RLCD 微调（`models/laya-zh`）

| 测试集 | 基线 | 微调后 | 备注 |
|---|---|---|---|
| 模板测试集（260 例 / 780 决策） | 42.8% | **90.5%** | choice 100% / score 74.6% / noul 96.9% |
| 模板测试集 ECE | 0.280 | **0.089** | 达到官方温度校准后水准 |
| 手写自然句困难集（80 例） | 41.7% | **55.4%** | noul +35pp，choice +10pp，score 持平 |

训练开销：Apple M4 单机 **约 16 分钟**（3 epoch / 3600 决策，冻结 256k 词表 embedding + 梯度检查点）；推理速度不受影响（MPS 16.5ms/决策）。

### 官方中文基准（feishu_zh）多版本迭代

| 版本 | choice 准确率 | four_noul 准确率 |
|---|---|---|
| 基座 multilingual 零样本 | 31.2% | 28.1% |
| V1 客服模板微调 | 37.5% | 42.2% |
| **V2 + 定向分诊语料 700 cases** | **51.6%**（macro-F1 0.482，误报动作 1/32） | 39.1% |
| V3 语料翻倍至 1200 cases | 35.9%（模板过拟合，反例） | 26.6% |

评测使用上游官方协议：`research/feishu_zh` 64 条冻结场景 × choice/four_noul × 3 repeats。
详见 [docs/finetune_report.md](docs/finetune_report.md) 第 5 节。

## 项目结构

```
├── docs/                      # 研究 / 架构 / 验证 / 微调 四份文档
├── scripts/
│   ├── build_cn_dataset.py    # 中文数据生成（1200 训练 + 260 测试 + 80 手写困难集）
│   ├── verify_laya.py         # 模型卡声明验证
│   ├── eval_cn.py             # 中文定量评估（准确率/ECE/Brier/延迟）
│   ├── finetune_cn.py         # RLCD 微调（单卡 MPS/CPU/CUDA）
│   ├── inspect_model.py       # 模型结构逐层探查
│   └── make_cover.py          # 创空间封面生成
├── data/                      # 生成的中文数据集（jsonl）
├── results/                   # 评估原始结果（json）
├── webapp/
│   ├── server.py              # FastAPI：页面 + 全部 API（含 Jev 兼容 /v1/systemone）
│   ├── static/index.html      # 复古风单页前端（零依赖）
│   ├── studio/                # ModelScope 创空间入口（薄启动器 + 封面）
│   └── deploy_studio.sh       # 创空间一键部署/更新
└── models/                    # 微调产物（git 忽略，按上文命令生成）
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/laya_research.md](docs/laya_research.md) | 模型研究：定位、三检查点、RLCD 训练方法、基准、局限 |
| [docs/architecture.md](docs/architecture.md) | 架构详解：总体架构、逐层结构与实测参数量、前向形状追踪 |
| [docs/verification_report.md](docs/verification_report.md) | 模型卡声明逐条验证 + 中文基线 + 延迟实测 |
| [docs/finetune_report.md](docs/finetune_report.md) | 中文微调：数据、RLCD 配置、前后对比、诚实结论 |

## 架构速览

```
[bos] <题型> question: <指令> [eos] [mask]选项0 [mask]选项1 … [eos] state [eos]
  ↓ mmBERT-base 编码器（22 层 ModernBERT，全局/滑动注意力 3:2 交替，RoPE，306.94M）
  ↓ + 题型嵌入 → 2 层 Transformer 决策头（14.18M）
  ↓ 在 K 个选项的 [mask] 位置打分（scorer，0.59M）→ softmax → 校准概率
  ＋ act_head（0.20M）输出 act / escalate 直答门控
```

- **答案空间与参数解耦**：选项在 [mask] 标记上打分，K 运行时可变——换 schema 零成本，也是"自定义场景"功能的结构基础
- **校准来自训练目标**：RLCD 以严格正则评分规则（log + spherical + RPS）为奖励，收敛目标天然是真实条件概率

## 已知限制

- 序数 score 在分布外仍是弱项（laya 最弱原语 + 模板数据泛化有限），需要真实中文工单语料再训
- 自定义/事实型问题可能出现自信的错误答案——multilingual 基座零样本能力的固有限制
- 创空间免费档 CPU 推理约 100-400ms/次；本地 MPS 约 20-90ms
- frp 公网入口依赖本机在线；创空间入口独立于本机

## 致谢与许可

- 模型与 SDK：[Convai Innovations / laya](https://github.com/NandhaKishorM/laya)（Apache-2.0）
- 灵感来源：[kuhung/ask-jev](https://github.com/kuhung/ask-jev)
- 本项目代码 Apache-2.0；概率输出仅供娱乐参考，重大决定请综合多方意见

---

<a id="english"></a>
## English

A full research project around **Laya**, a non-autoregressive "System 1" decision engine (typed questions → calibrated probabilities in a single forward pass, zero text generation):

- **Research docs** — model study, layer-by-layer architecture, claim verification, fine-tuning report
- **Claim verification** — all 9 model-card claims tested; known limits (near-random zero-shot, English-checkpoint cross-lingual collapse, over-confidence) reproduced honestly
- **Chinese RLCD fine-tuning** — official GRPO-style loop with proper-scoring-rule rewards; template accuracy **42.8% → 90.5%**, ECE **0.28 → 0.089** on an Apple M4 in ~16 minutes
- **Web app** — retro-styled decision site (life decisions + six production workflow labs + custom schemas) running real local inference, with a **Jev-compatible `/v1/systemone` API** so any TypeSafe Jev client (including the ask-jev website) can use this model
- **One-click deploy** to a ModelScope Studio: <https://modelscope.cn/studios/zcgnull/ask-laya>

Quick start and details above; docs are in Chinese. Apache-2.0.
