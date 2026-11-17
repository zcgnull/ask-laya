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

    # 镜像源可能滞后：基线放宽，启动后尽力升到 0.4.x（新特性自动启用，旧版优雅降级）
    for pkg in ("fastapi", "uvicorn", "laya>=0.3.5"):
        name = pkg.split(">=")[0].split("==")[0]
        try:
            __import__(name)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True)


_ensure_deps()

import laya  # noqa: E402


MODEL_CANDIDATES = [
    ("zcgnull/laya-zh-v2", None),                    # 中文微调版（首选）
    ("convaiinnovations/laya", "multilingual"),      # 官方 multilingual（回退）
]


def _load_agent():
    """ModelScope 站内下载优先（站内加速），失败回退 HF；逐个候选尝试。"""
    override = os.environ.get("ASK_LAYA_MODEL_DIR")
    if override:
        return laya.load(override)
    errors = []
    for repo_id, sub in MODEL_CANDIDATES:
        try:
            from modelscope import snapshot_download as ms_dl
            d = ms_dl(repo_id, allow_patterns=[f"{sub}/*"] if sub else None)
            path = os.path.join(d, sub) if sub else d
            return laya.load(path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{repo_id}: {e}")
            print(f"[ask-laya] modelscope load {repo_id} failed ({e})", flush=True)
        try:
            return laya.load(repo_id, subfolder=sub)
        except Exception as e:  # noqa: BLE001
            errors.append(f"HF {repo_id}: {e}")
            print(f"[ask-laya] HF load {repo_id} failed ({e})", flush=True)
    raise RuntimeError("all model candidates failed: " + "; ".join(errors))


print("[ask-laya] loading model ...", flush=True)
import time  # noqa: E402

_t0 = time.time()
AGENT = _load_agent()
print(f"[ask-laya] ready in {time.time() - _t0:.0f}s on {AGENT.device}", flush=True)

# 复用与本地完全相同的 FastAPI 应用（原味页面 + 全部 API）
import server  # noqa: E402

server._state["agent"] = AGENT
server._state["label"] = "laya-zh-v2 (中文微调)"
server._state["backend_id"] = "laya-zh-v2"
server._LAYA_CACHE.clear()  # 注册表切换器懒加载，避免双检查点驻留内存
app = server.app


def _default_piccolo_worker():
    """后台把默认后端切成 Piccolo-4B-Q4_K_M：GGUF 站内下载 + 随附二进制拉起 llama-server。

    失败（下载/启动异常）则保持 laya-zh-v2 服务，页面切换器仍可手动重试。
    """
    import os
    import subprocess
    import threading
    import traceback

    def _ensure_binary():
        """预编译二进制优先（本仓库 bin/ 或 runtime-src 仓库缓存）；否则站内下载源码原生编译，
        编译成功后把二进制回传 runtime-src 仓库，后续部署直接下载（免 ~45 分钟编译）。"""
        binary = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "bin", "llama-server-linux-x86_64")
        if os.path.isfile(binary):
            return binary
        workdir = "/mnt/workspace/piccolo-build"
        built = os.path.join(workdir, "build", "bin", "llama-server")
        if os.path.isfile(built):
            return built
        try:  # runtime-src 仓库可能已有此前构建回传的二进制
            from modelscope import snapshot_download
            cached = snapshot_download("zcgnull/piccolo-runtime-src",
                                       allow_patterns=["bin/llama-server-linux-x86_64"])
            cand = os.path.join(cached, "bin", "llama-server-linux-x86_64")
            if os.path.isfile(cand) and os.path.getsize(cand) > 100000:
                os.makedirs(os.path.join(workdir, "bin"), exist_ok=True)
                import shutil
                shutil.copy(cand, os.path.join(workdir, "bin", "llama-server-linux-x86_64"))
                return os.path.join(workdir, "bin", "llama-server-linux-x86_64")
        except Exception:  # noqa: BLE001
            pass
        print("[ask-laya] no prebuilt llama-server; building from source ...", flush=True)
        import subprocess
        import sys as _sys
        os.makedirs(workdir, exist_ok=True)
        tarball = os.path.join(workdir, "piccolo-src.tar.gz")
        if not os.path.isfile(tarball):
            import shutil
            import time as _time
            from modelscope import snapshot_download
            last_err = None
            for attempt in range(4):          # 新建仓库索引有延迟，404 时退避重试
                try:
                    src = snapshot_download("zcgnull/piccolo-runtime-src")
                    shutil.copy(os.path.join(src, "piccolo-src.tar.gz"), tarball)
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    print(f"[ask-laya] runtime-src download attempt {attempt+1} failed: {e}", flush=True)
                    _time.sleep(30)
            if last_err is not None:
                raise last_err
        subprocess.run(["tar", "xzf", tarball, "-C", workdir], check=True)
        subprocess.run([_sys.executable, "-m", "pip", "install", "--quiet", "cmake"], check=False)
        if subprocess.run(["sh", "-c", "command -v cc || command -v gcc"], capture_output=True).returncode != 0:
            subprocess.run(["sh", "-c", "apt-get update -qq && apt-get install -y -qq build-essential"],
                           check=False)
        subprocess.run(["cmake", "-S", workdir, "-B", os.path.join(workdir, "build"),
                        "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF",
                        "-DLLAMA_BUILD_EXAMPLES=OFF"], check=True, capture_output=True)
        subprocess.run(["cmake", "--build", os.path.join(workdir, "build"),
                        "--target", "llama-server", "-j", "2"], check=True)
        try:  # 回传二进制到 runtime-src 仓库，后续部署免编译
            token = os.environ.get("MODELSCOPE_API_TOKEN")
            if token:
                from modelscope.hub.api import HubApi
                api = HubApi()
                api.login(token)
                api.upload_file(repo_id="zcgnull/piccolo-runtime-src",
                                path_or_fileobj=built,
                                path_in_repo="bin/llama-server-linux-x86_64",
                                commit_message="prebuilt llama-server (linux x86_64, auto-cached)")
                print("[ask-laya] prebuilt binary uploaded for future deploys", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ask-laya] binary upload-back skipped: {e}", flush=True)
        return built

    def run():
        try:
            binary = _ensure_binary()
            os.chmod(binary, 0o755)
            os.environ["PICCOLO_LLAMA_SERVER"] = binary
            from gguf_backend import get_quantized_agent
            agent = get_quantized_agent("AgentScope/Piccolo-4B-Q4_K_M", port=8300)
            server._LAYA_CACHE.clear()
            with server._state["lock"]:
                server._state["agent"] = agent
                server._state["label"] = "Piccolo-4B Q4 量化"
                server._state["backend_id"] = "piccolo-4b-q4"
            print("[ask-laya] default backend switched -> Piccolo-4B-Q4_K_M", flush=True)
        except Exception:  # noqa: BLE001
            print("[ask-laya] piccolo default switch failed, stay on laya:\n"
                  + traceback.format_exc(), flush=True)

    threading.Thread(target=run, daemon=True).start()


_default_piccolo_worker()

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
