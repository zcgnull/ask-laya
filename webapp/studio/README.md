---
title: Ask Laya 问问Laya
emoji: ⚖️
colorFrom: red
colorTo: yellow
sdk: gradio
pinned: false
license: apache-2.0
---

# Ask Laya · 问问 Laya

> 运行方式：gradio SDK 容器内直接运行 **FastAPI 应用**（`app.py` 薄启动器 → uvicorn），
> 页面为原味复古单页（static/index.html），不经过 gradio 渲染；
> gradio 仅提供一个全屏 iframe 壳以满足平台代理的协议握手。

参考 [kuhung/ask-jev](https://github.com/kuhung/ask-jev) 的生活微决策玩具，后端是真实的
[convaiinnovations/laya](https://www.modelscope.cn/models/convaiinnovations/laya)
（multilingual 检查点，mmBERT-base 322M）——非自回归"系统一"决策引擎，
**让 Laya 裁决**，输出全部决策（0 个生成 token），概率由 RLCD（严格正则评分规则）训练校准。

三种模式：该不该？（noul）／选哪个？（choice，支持中英文候选切分）／冲动指数（score）。
中英双语：英文问题自动切换英文 prompt 与裁决文案，"A or B" 是非题自动升级为选择模式。

本地版（FastAPI + 原味前端）：见项目 `webapp/server.py`。
