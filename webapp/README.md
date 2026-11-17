# Ask Laya

参考 [kuhung/ask-jev](https://github.com/kuhung/ask-jev) 的简约决策网站，后端换成 **真实的 laya 模型本地推理**（无 API Key、无广告位）。

## 启动

```bash
cd /Volumes/zcgnull/project/laya
.venv/bin/python webapp/server.py                # 默认 multilingual 检查点，http://127.0.0.1:8123
# 也可指定任意检查点（如本项目微调产物）：
.venv/bin/python webapp/server.py --model models/laya-zh --port 8124
```

## 三种模式（与 ask-jev 对齐）

| 模式 | laya 题型 | 输出 |
|---|---|---|
| 该不该？ | noul | P(值得做) + 执行/劝退裁决 |
| 选哪个？ | choice | 问题中按「还是/或者/、」切分候选，逐选项概率条 |
| 冲动指数 | score | 4 级期望分 → 换算 1-10 冲动指数 |

页面额外展示 laya 特有信息：逐选项校准概率条、裁决耗时（毫秒）、0 输出 token。

## Jev 兼容 SystemOne API（laya 0.3.7 适配衍生）

laya 0.3.7 新增 `laya-serve`（TypeSafe Jev 的 `/v1/systemone` 线协议）。本站把该协议直接挂进了同一个 FastAPI 进程（`mount_jev_api()`，复用本进程模型）：

- `POST /v1/systemone`：接受 Jev 形状请求 `{state, questions, model?}`，返回 `{model, answers, usage}`；未知 `model` 字段自动忽略
- `GET /health`：健康探针

**玩法**：在 ask-jev 官网的「TypeSafe 接口设置」里把 endpoint 填成 `http://39.98.88.195:18081/v1/systemone`（或局域网 `http://127.0.0.1:8123/v1/systemone`），官方站点就会用我们本地的真实 laya 模型出裁决。任何按 Jev 协议写的客户端（如 hs-jev）同样即插即用。

## 量化 GGUF 决策模型支持（AgentScope Piccolo 等 blink 架构）

框架原生适配量化决策模型（如 `AgentScope/Piccolo-4B-Q4_K_M`）：自动下载 → 托管拉起补丁版
llama-server（随机空闲端口、禁用前缀缓存、8192 ctx 对齐）→ 健康检查 → 进程退出自动清理。

```bash
# 一条命令：自动下载模型（若未缓存）、启动运行时、整站跑在量化后端上
.venv/bin/python webapp/server.py --gguf AgentScope/Piccolo-4B-Q4_K_M
# 等价环境变量：ASK_LAYA_GGUF=AgentScope/Piccolo-4B-Q4_K_M
# 本地 GGUF 路径也行：--gguf /path/to/xxx.gguf（同目录需有 decision-inference.json）
```

- **后端优先级**：`--piccolo-url`（复用外部 Jev 服务）> `--gguf`（托管量化后端）> 默认 laya 检查点；
- **运行时要求**：QwenPaw 补丁版 llama.cpp（双向 prompt attention + 决策端点；构建步骤见
  `gguf_backend.py` 的报错提示，或用 `PICCOLO_LLAMA_SERVER` 指向已有二进制）；
- **协议适配**：`RemoteJevAgent` 内做 score↔choice 双向转换（Piccolo 端点按上游参考实现只收
  choice）、noul 置信度合成，站内全部功能（生活裁决 / 七场景实验室 / Jev API）无需改动即可使用；
- `/health` 的 `backend` 字段标识当前后端：`local-checkpoint` / `quantized-gguf` / `remote-url`；
- 质量 vs 延迟（官方 feishu_zh 协议实测）：Piccolo-4B-Q4 零样本 choice 100% / four_noul 95.3%，
  但 ~0.7-3s/请求（laya 检查点 ~60ms）——按场景选择后端即可。

**页面内切换模型**：页头下拉框可在运行时切换后端（`GET/POST /api/backends`）——Piccolo 量化与
各 laya 检查点懒加载（laya 首次 ~8s、LLRU 缓存驻留 2 个；GGUF 单例 llama-server 复用，
切回 0 秒），切换期间展示加载提示，失败自动保留原后端。

管理器实现：[webapp/gguf_backend.py](webapp/gguf_backend.py)；基准驱动：
`scripts/piccolo_http_feishu_eval.py`（任意 /v1/systemone 后端可测）。

## laya 0.4.21 新能力集成（2026-11）

- **场景实验室**：新增「弃答门控」阈值（0.3–0.7，答案带 `low_confidence` 时卡片显示"⚠ 低置信，建议复核"）与「选项稳定性校验」（上游 OptionStabilityHook：选项重排/改名 metamorphic 一致性，卡片显示 稳定性/改名一致/软稳定）；答案卡新增上游校准置信 `answer_confidence`
- **Jev API 对齐上游 0.4.x**：`/v1/systemone` 的 choice/score 答案附 `x_jev_confidence`（Jev 口径 max(p) 置信度），`LAYA_JEV_STRICT=1` 时投影到严格 Jev 契约，state 截断记日志告警
- **新端点**：`POST /v1/route`（仅语种路由诊断，零前向）、`POST /v1/systemone/batch`（批量 ≤32 条，优先走 Agent.predict_batch）
- 旧版 laya 自动降级：新字段/开关缺失时不影响原有功能

## 依赖版本适配（laya 0.3.7 → 0.4.21）

- `load / Agent.predict / build_sequence / presets` 签名与 0.3.5 完全兼容，本仓库代码无需改动即通过回归；
- 新增能力：`laya.serve`（Jev 协议服务器，本站已集成）、`research/benchmarks/feishu_zh`（可复现中文工作流基准）、分语言评测 harness、Docker/Compose 快速启动；
- 可选后续：用 `feishu_zh` 基准的 cases 复评中文微调检查点 `models/laya-zh`。

## 场景实验室（对齐官方 laya-demo）

页面顶部切换到「🧪 场景实验室」，六大生产工作流，每场景预置 schema 与示例，state 可编辑，**所有问题同时裁决**：

| 场景 | schema 来源 | 判定内容 |
|---|---|---|
| 🎫 客服工单分诊 | `laya.triage_questions` | 意图 / 紧急 / 挫败度 / 退款 / 流失 |
| 📧 邮件 + 钓鱼 | `laya.email_questions` | 分类 / 垃圾 / 钓鱼 / 紧急 / 需回复 |
| 🛡️ LLM 护栏 | `laya.guard_questions` | 越狱 / 注入 / 敏感信息 / 危害分级 / 话题 |
| 🧹 内容审核 | `laya.moderation_questions` | 毒性 / 攻击 / 威胁 / 广告 / 严重度 |
| 🧭 模型路由 | `laya.router_questions` | 难度 / 领域 / 需工具 / 敏感后果 |
| 📄 RAG 段落过滤 | 自建（官方 demo 同款工作流） | 相关 / 主题一致 / 信息价值（state 为 JSON） |
| 🛠️ 自定义 | **你在页面上现场定义** | 是非 / 评分 / 选择任意组合（≤12 问），选项运行时给定、零训练 |

每个答案卡展示：中文问题名、题型徽章、判定结果、完整概率分布、置信度与 **直答门控值**（act head 输出，"阈值是你的，不是模型的"）。

> 自定义场景就是 laya 的核心卖点演示：选项打分在 [MASK] 标记上进行，答案空间在请求时定义——换任何 schema 都不用重训模型。内置 7 组示例 schema（中文客服工单 / Support ticket EN / 中文内容安全 / Content safety EN / 中文邮件分诊 / Email triage EN / Product review EN），点示例名一键载入。

## 中英双语适配（v2）

- 用 `laya.lang.is_english` 检测问题语言：英文问题自动使用英文 instructions/criteria 与英文裁决文案（含买/吃/健身/辞职等场景的英文专属话术）；
- choice 切分支持英文 `or / vs`（中文 `还是/或者/、` 不变）；超长分句按词边界截断，绝不丢弃；
- **是非题自动升级**：yes_no 模式下检测到「A or B」句式（中英文皆可）自动切换 choice 模式，结果卡 meta 会注明；
- 事实型疑问句（Is/Are/Do… 开头）自动**平行化选项**（"Is Beijing in the United States or in China?" → 选项变为 "in the United States" / "in China"），并把指令切换为"选出正确答案"——分类器对平行选项的判别显著更稳（该例从 56/44 错向修正为 78% 正确）。

## 结构

```
webapp/
  server.py        # FastAPI：加载 laya agent，POST /api/ask，GET /api/about，托管静态页
  static/index.html # 单页前端（内联 CSS/JS，零依赖）
  studio/          # ModelScope 创空间版（gradio 壳内嵌原味页面，逻辑与 server.py 同源）
  deploy_studio.sh # 一键部署到创空间：MODELSCOPE_API_KEY=xx bash deploy_studio.sh
```

## 线上部署

- **ModelScope 创空间**：<https://modelscope.cn/studios/zcgnull/ask-laya>（公开，免费 2vCPU/8G；现已改用中文微调模型 [zcgnull/laya-zh-v2](https://modelscope.cn/models/zcgnull/laya-zh-v2)，站内下载，CPU 裁决约 100-400ms）
- **本地/FRP 公网**：<http://39.98.88.195:18081>（FastAPI 原味前端版，依赖本机在线）

更新创空间代码：改 `studio/` 后重跑 `deploy_studio.sh` 即可（脚本已处理平台初始提交的合并）。
