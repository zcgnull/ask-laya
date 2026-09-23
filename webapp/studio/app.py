"""Ask Laya — ModelScope 创空间入口（非 gradio 渲染）。

直接运行与本地 webapp/server.py 相同的 FastAPI 应用：
  /                原味复古单页（static/index.html，与本地版完全一致）
  /api/ask         生活裁决（该不该 / 选哪个 / 冲动指数）
  /api/playground  场景实验室（六大工作流 + 自定义 schema）
  /v1/systemone    Jev 兼容 SystemOne 协议（laya.serve）
  /health          健康探针

模型在启动时从 ModelScope 站内下载 convaiinnovations/laya 的 multilingual
子目录（mmBERT-base 322M），CPU 单次裁决推理。gradio SDK 只是平台的运行
容器，页面本身不经过 gradio。
"""
import os


def _ensure_deps():
    """补齐基础镜像可能缺失的依赖（torch/gradio/modelscope 镜像自带）。"""
    import subprocess
    import sys

    for pkg in ("fastapi", "uvicorn", "laya>=0.3.5"):
        name = pkg.split(">=")[0].split("==")[0]
        try:
            __import__(name)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True)


_ensure_deps()

import laya  # noqa: E402


def _load_agent():
    """ModelScope 站内下载优先（站内加速），失败回退 HF。"""
    override = os.environ.get("ASK_LAYA_MODEL_DIR")
    if override:
        return laya.load(override)
    try:
        from modelscope import snapshot_download as ms_dl
        d = ms_dl("convaiinnovations/laya", allow_patterns=["multilingual/*"])
        return laya.load(os.path.join(d, "multilingual"))
    except Exception as e:  # noqa: BLE001
        print(f"[ask-laya] modelscope download failed ({e}), fallback to HF", flush=True)
        return laya.load("convaiinnovations/laya", subfolder="multilingual")


print("[ask-laya] loading model ...", flush=True)
import time  # noqa: E402

_t0 = time.time()
AGENT = _load_agent()
print(f"[ask-laya] ready in {time.time() - _t0:.0f}s on {AGENT.device}", flush=True)

# 复用与本地完全相同的 FastAPI 应用（原味页面 + 全部 API）
import server  # noqa: E402

server._state["agent"] = AGENT
server._state["label"] = "laya-multilingual"
app = server.app

# 平台代理要求根路径呈现 gradio 前端：用最小 Blocks 壳全屏内嵌我们的原味页面
import gradio as gr  # noqa: E402

from fastapi.responses import FileResponse  # noqa: E402


@app.get("/index.html")
def _studio_index():
    return FileResponse(os.path.join(
        os.path.dirname(os.path.abspath(server.__file__)), "static", "index.html"))


with gr.Blocks(css="body{margin:0} .gradio-container{margin:0!important;padding:0!important;"
                    "max-width:100%!important;min-height:100vh!important} footer{display:none!important}") as demo:
    gr.HTML('<iframe src="/index.html" title="Ask Laya" '
            'style="position:fixed;inset:0;width:100vw;height:100vh;border:0;display:block"></iframe>')

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or 7860)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
