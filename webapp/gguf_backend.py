"""量化 GGUF 决策模型后端管理器（AgentScope Piccolo 等 blink 架构）。

把"下载量化模型 → 拉起补丁版 llama-server → 健康检查 → 进程清理"整条链自动化，
使框架原生适配 AgentScope/Piccolo-*-Q4_K_M 这类量化决策模型：

    from gguf_backend import make_quantized_backend
    agent = make_quantized_backend("AgentScope/Piccolo-4B-Q4_K_M")

约束（与上游协议一致）：
  - 运行时必须是 QwenPaw 补丁版 llama.cpp（双向 prompt attention + 决策端点），
    标准 llama.cpp / Ollama 不实现该协议；二进制查找顺序：
      $PICCOLO_LLAMA_SERVER > <项目>/piccolo-runtime/build/bin/llama-server > PATH
  - 禁用前缀缓存（--cache-reuse 0），context/batch/microbatch 对齐（8192），
    超预算显式失败而非静默截断
  - 模型仓库需含 decision-inference.json（blink 架构描述）与 *.gguf
"""
import atexit
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_CANDIDATES = [
    ROOT / "piccolo-runtime" / "build" / "bin" / "llama-server",
    ROOT.parent / "piccolo-runtime" / "build" / "bin" / "llama-server",
]


class GgufBackendError(RuntimeError):
    """量化后端装配失败（模型缺失 / 运行时缺失 / 健康检查超时）。"""


def find_llama_server() -> Path:
    """定位补丁版 llama-server：环境变量 > 项目约定路径 > PATH。"""
    env = os.environ.get("PICCOLO_LLAMA_SERVER")
    if env and Path(env).is_file():
        return Path(env)
    for cand in DEFAULT_RUNTIME_CANDIDATES:
        if cand.is_file():
            return cand
    which = __import__("shutil").which("llama-server")
    if which:
        return Path(which)
    raise GgufBackendError(
        "未找到补丁版 llama-server。请先构建 QwenPaw Piccolo 运行时：\n"
        "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp\n"
        "  git fetch --depth 1 origin f4d8af608a6b39946406f9439edfb6e83ca6a923 && git checkout FETCH_HEAD\n"
        "  curl -sL https://raw.githubusercontent.com/agentscope-ai/QwenPaw/main/docs/patches/"
        "piccolo-llama.cpp.patch | git apply\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j\n"
        "或用环境变量 PICCOLO_LLAMA_SERVER 指向已有二进制。")


def resolve_gguf(model_spec: str) -> tuple[Path, dict]:
    """解析模型规格（本地 GGUF 路径 或 ModelScope/HF 仓库 ID）→ (gguf路径, decision-inference)。

    仓库 ID 自动 snapshot_download（取 *.gguf 与 decision-inference.json；多分片仓库会取全部分片）。
    """
    p = Path(model_spec).expanduser()
    if p.suffix == ".gguf" and p.is_file():
        gguf = p
        di = p.with_name("decision-inference.json")
    elif p.is_dir():
        di_path = p / "decision-inference.json"
        ggf = sorted(p.glob("*.gguf"))
        if not ggf or not di_path.is_file():
            raise GgufBackendError(f"{p} 目录缺少 *.gguf 或 decision-inference.json")
        gguf, di = ggf[0], di_path
    else:
        print(f"[gguf-backend] downloading {model_spec} from ModelScope ...", flush=True)
        try:
            from modelscope import snapshot_download
            d = snapshot_download(model_spec, allow_patterns=["*.gguf", "decision-inference.json"])
        except Exception as e:  # noqa: BLE001
            raise GgufBackendError(f"模型仓库下载失败 {model_spec}: {e}") from e
        d = Path(d)
        ggf = sorted(d.rglob("*.gguf"))
        di_path = next(d.rglob("decision-inference.json"), None)
        if not ggf or di_path is None:
            raise GgufBackendError(f"{model_spec} 仓库中没有 *.gguf / decision-inference.json")
        gguf, di = ggf[0], di_path

    descriptor = json.loads(di.read_text())
    if descriptor.get("attention_mode") != "bidirectional_prompt":
        raise GgufBackendError(f"{gguf.name} 不是 blink 决策架构（decision-inference.json 不匹配）")
    return gguf, descriptor


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ManagedLlamaServer:
    """补丁版 llama-server 子进程：拉起 → /health 就绪 → atexit 清理。"""

    def __init__(self, gguf: Path, port: int | None = None, ctx: int = 8192, ngl: int = 99):
        self.gguf = gguf
        self.port = port or _free_port()
        self.ctx = ctx
        self.ngl = ngl
        self.proc: subprocess.Popen | None = None
        atexit.register(self.stop)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 180.0):
        binary = find_llama_server()
        cmd = [str(binary), "-m", str(self.gguf), "--port", str(self.port),
               "-ngl", str(self.ngl), "-c", str(self.ctx), "-b", str(self.ctx),
               "-ub", str(self.ctx), "--cache-reuse", "0"]
        print("[gguf-backend] starting:", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise GgufBackendError(f"llama-server 提前退出（exit {self.proc.returncode}），"
                                       "检查 GGUF 与运行时版本是否匹配")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=3) as resp:
                    if json.loads(resp.read()).get("status") == "ok":
                        print(f"[gguf-backend] ready at {self.base_url}", flush=True)
                        return self.base_url
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                time.sleep(1.5)
        self.stop()
        raise GgufBackendError(f"llama-server {timeout}s 内未就绪（模型加载或参数错误）")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def make_quantized_backend(model_spec: str, port: int | None = None,
                           external_url: str | None = None, name: str | None = None):
    """装配量化后端：external_url 直接复用外部服务；否则托管拉起 llama-server。

    返回 (agent, base_url)；agent 具备 RemoteJevAgent 兼容的 .predict()。
    """
    # 延迟导入避免循环依赖；RemoteJevAgent 在 server.py 定义
    from server import RemoteJevAgent

    if external_url:
        return RemoteJevAgent(external_url, name or "quantized (external)"), external_url
    gguf, descriptor = resolve_gguf(model_spec)
    srv = ManagedLlamaServer(gguf, port=port)
    base_url = srv.start()
    agent = RemoteJevAgent(base_url, name or f"{gguf.stem} (quantized)")
    agent.managed_server = srv          # 引用持有，进程随 agent 生命周期清理
    agent.cfg = {"max_len": descriptor.get("max_context_length", 65536)}
    return agent, base_url
