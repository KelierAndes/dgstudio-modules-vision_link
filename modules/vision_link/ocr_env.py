"""OCR 运行环境：用应用自带的 _python 解释器在子进程里跑 RapidOCR。

主进程（PyInstaller 冻结版）里 import onnxruntime 会段错误，所以识别依赖绝不装进
模块进程：随包的模块依赖放在 wheels/（宿主会解到 _deps、进模块进程），OCR 那一套
单独放 ocr_wheels/（宿主不碰），首次用到 OCR 时由应用自带的 _python 解释器安装，
识别请求交给 ocr_worker.py 子进程，主进程只收发 JSON。
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading

# 离线装时逐个点名：rapidocr 的依赖里写的是 opencv-python，随包给的是 headless
# 版；让 pip 自己解依赖就会联网去拉全量 opencv，--no-deps 点名才走随包 wheel。
OFFLINE_PACKAGES = ("rapidocr-onnxruntime", "onnxruntime", "pyclipper",
                    "shapely", "pyyaml", "pillow", "six", "tqdm", "numpy",
                    "opencv-python-headless")
# rapidocr-onnxruntime 的元数据写着 Requires-Python <3.13，但它是纯 Python 包，
# 实测在应用内置的 3.14 解释器里跑得通；不加这个参数 pip 会直接拒装。
PIP_FLAGS = ("--disable-pip-version-check", "--no-input",
             "--ignore-requires-python")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "ocr_worker.py")
# -E -s：内置 Python 必须只看自己的 site-packages。宿主机的用户级
# %APPDATA%\Python\314\site-packages 里若恰好有同名包，pip 会说「already
# satisfied」而跳过随包 wheel，换一台干净机器就装不出来。
PY_FLAGS = ("-X", "utf8", "-E", "-s")
CHECK_TIMEOUT_S = 90.0
INSTALL_TIMEOUT_S = 1800.0
START_TIMEOUT_S = 180.0
REQUEST_TIMEOUT_S = 30.0

_state = "idle"          # idle / preparing / ready / failed
_note = ""
_client = None
_guard = threading.Lock()


def runner_python() -> str:
    """OCR 用哪个解释器：打包版是 exe 旁的 _python，开发版用环境变量指定。"""
    if getattr(sys, "frozen", False):
        path = os.path.join(os.path.dirname(os.path.abspath(sys.executable)),
                            "_python", "python.exe")
    else:
        path = str(os.environ.get("DGSTUDIO_OCR_PYTHON") or "")
    return path if path and os.path.isfile(path) else ""
def status() -> tuple[str, str]:
    with _guard:
        return _state, _note


def client():
    with _guard:
        return _client if _state == "ready" else None


def stop() -> None:
    with _guard:
        target = _client
    if target is not None:
        target.close()


def prepare_async(log=None) -> str:
    """后台把 OCR 依赖装进内置 Python；返回调用时刻的状态。"""
    with _guard:
        if _state in ("preparing", "ready"):
            return _state
        python = runner_python()
        if not python:
            _apply("idle", "未找到 OCR 用的 Python 运行时"
                   "（打包版看 exe 旁 _python/，开发版设 DGSTUDIO_OCR_PYTHON）",
                   None)
            return _state
        _apply("preparing", "", None)
    threading.Thread(target=_prepare, args=(python, log), daemon=True).start()
    return "preparing"


def _apply(state: str, note: str, new_client) -> None:
    """只在 _guard 里改动状态；new_client 为 None 时保留现有客户端。"""
    global _state, _note, _client
    _state = state
    _note = str(note or "")[:300]
    if new_client is not None:
        _client = new_client


def _log(log, text: str) -> None:
    if log is not None:
        try:
            log(text)
        except Exception:
            pass


def _prepare(python: str, log) -> None:
    ok, note = _check(python)
    if not ok:
        _log(log, "OCR：正在为内置 Python 下载识别依赖（首次约 1-2 分钟）")
        ok, out = _install(python, log)
        note = "" if ok else out
        if ok:
            ok, note = _check(python)
    with _guard:
        if ok:
            _apply("ready", "", OcrClient(python))
        else:
            _apply("failed", note, None)
    if ok:
        _log(log, "OCR：内置 Python 识别环境就绪")
    else:
        _log(log, f"OCR 环境不可用，数字/文字检测回退模板匹配：{note}")


def _clean_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("PYTHON")}
    env.update(PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def _run(args: list[str], timeout: float) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout, env=_clean_env())
    except OSError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired as exc:
        return False, f"命令超时: {exc}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, out


def _check(python: str) -> tuple[bool, str]:
    return _run([python, *PY_FLAGS, WORKER, "--check"], CHECK_TIMEOUT_S)


def _wheel_dirs() -> list[str]:
    """随包 wheel 目录：ocr_wheels/ 给识别栈，wheels/ 补 numpy 与 headless opencv。"""
    base = os.path.dirname(WORKER)
    return [os.path.join(base, name) for name in ("ocr_wheels", "wheels")
            if os.path.isdir(os.path.join(base, name))]


def _install(python: str, log) -> tuple[bool, str]:
    base = [python, *PY_FLAGS, "-m", "pip", "install", *PIP_FLAGS]
    dirs = _wheel_dirs()
    ok, out = False, ""
    if dirs:
        offline = [*base, "--no-index", "--no-deps"]
        for path in dirs:
            offline += ["--find-links", path]
        ok, out = _run([*offline, *OFFLINE_PACKAGES], INSTALL_TIMEOUT_S)
        if not ok:
            _log(log, "OCR：随包 wheel 装不进内置 Python，改从网络安装")
    if not ok:
        ok, out = _run([*base, "--no-deps", *OFFLINE_PACKAGES],
                       INSTALL_TIMEOUT_S)
    for line in out.strip().splitlines()[-3:]:
        _log(log, f"[ocr deps] {line}")
    return ok, out


class OcrClient:
    """常驻 OCR 子进程：进程坏了自动重开一次，仍不可用就交回模板法。"""

    def __init__(self, python: str):
        self._python = python
        self._proc: subprocess.Popen | None = None
        self._replies: dict[str, queue.Queue] = {}
        self._startup: queue.Queue = queue.Queue()
        self._seq = 0
        self._cond = threading.Lock()

    def _start(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                [self._python, *PY_FLAGS, WORKER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                errors="replace", env=_clean_env())
        except OSError:
            self._proc = None
            return False
        threading.Thread(target=self._read_loop, daemon=True).start()
        try:
            first = self._startup.get(timeout=START_TIMEOUT_S)
        except queue.Empty:
            self.close()
            return False
        if not first.get("ok"):
            self.close()
            return False
        return True

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if "id" not in payload:
                self._startup.put(payload)
                continue
            box = self._replies.get(str(payload.get("id")))
            if box is not None:
                box.put(payload)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        try:
            proc.terminate()
        except OSError:
            pass

    def _encode(self, image) -> str:
        import cv2
        ok, buf = cv2.imencode(".png", image)
        return base64.b64encode(buf.tobytes()).decode() if ok else ""

    def __call__(self, image):
        """与 RapidOCR 同形状：返回 (识别行, 耗时)；引擎不可用时 (None, None)。"""
        png = self._encode(image)
        if not png:
            return None, None
        for _attempt in range(2):
            reply = self._request(png)
            if reply is None:
                continue
            if reply.get("error"):
                return None, None
            return reply.get("result") or [], reply.get("elapse")
        return None, None

    def _request(self, png: str):
        with self._cond:
            if self._proc is None and not self._start():
                return None
            self._seq += 1
            key = str(self._seq)
            box: queue.Queue = queue.Queue()
            self._replies[key] = box
            try:
                self._proc.stdin.write(
                    json.dumps({"id": key, "png": png}) + "\n")
                self._proc.stdin.flush()
            except OSError:
                self._replies.pop(key, None)
                return None
        try:
            return box.get(timeout=REQUEST_TIMEOUT_S)
        except queue.Empty:
            self.close()
            return None
        finally:
            self._replies.pop(key, None)
