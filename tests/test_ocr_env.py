from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401  定位 DGStudio 核心仓库

import base64
import json
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from modules.vision_link import detect, ocr_env
from modules.vision_link.bridge import VisionBridge

STUB_RAPIDOCR = '''
class RapidOCR:
    def __call__(self, image):
        h, w = image.shape[:2]
        box = [[0, 0], [w, 0], [w, h], [0, h]]
        return [[[box, "1280", 0.97], [box, "低置信", 0.31]], 0.02]
'''

STUB_WORKER = '''
import json
import sys

sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    job = json.loads(line)
    box = [[0, 0], [9, 0], [9, 6], [0, 6]]
    sys.stdout.write(json.dumps({"id": job["id"],
                                 "result": [[box, "42", 0.91]],
                                 "elapse": 0.01}) + "\\n")
    sys.stdout.flush()
'''


def _png(image) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class EnvStateMixin(unittest.TestCase):
    """还原 ocr_env 的模块级状态与被替换的函数，测试之间互不污染。"""

    def setUp(self):
        self._state = (ocr_env._state, ocr_env._note, ocr_env._client)
        ocr_env._client = None
        ocr_env._apply("idle", "", None)
        self._patched: dict[str, object] = {}
        self._files: list[str] = []
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.close_client()
        ocr_env._client = self._state[2]
        ocr_env._apply(self._state[0], self._state[1], None)
        for name, original in self._patched.items():
            setattr(ocr_env, name, original)
        self._patched.clear()

    def patch(self, **kwargs) -> None:
        for name, value in kwargs.items():
            if name not in self._patched:
                self._patched[name] = getattr(ocr_env, name)
            setattr(ocr_env, name, value)

    def write(self, name: str, text: str) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        self._files.append(path)
        return path

    def close_client(self) -> None:
        current = ocr_env._client
        if isinstance(current, ocr_env.OcrClient):
            current.close()


class RunnerTests(EnvStateMixin):

    def test_dev_runner_comes_from_environment(self):
        if getattr(sys, "frozen", False):
            self.skipTest("打包版解释器固定在 exe 旁的 _python/")
        old = os.environ.get("DGSTUDIO_OCR_PYTHON")
        try:
            os.environ.pop("DGSTUDIO_OCR_PYTHON", None)
            self.assertEqual(ocr_env.runner_python(), "")
            os.environ["DGSTUDIO_OCR_PYTHON"] = os.path.join(self.tmp, "x.exe")
            self.assertEqual(ocr_env.runner_python(), "")
            os.environ["DGSTUDIO_OCR_PYTHON"] = sys.executable
            self.assertEqual(ocr_env.runner_python(), sys.executable)
        finally:
            if old is None:
                os.environ.pop("DGSTUDIO_OCR_PYTHON", None)
            else:
                os.environ["DGSTUDIO_OCR_PYTHON"] = old

    def test_prepare_without_runtime_stays_idle(self):
        logs: list[str] = []
        self.patch(runner_python=lambda: "")
        self.assertEqual(ocr_env.prepare_async(logs.append), "idle")
        state, note = ocr_env.status()
        self.assertEqual(state, "idle")
        self.assertIn("_python", note)
        self.assertIsNone(ocr_env.client())


class PrepareTests(EnvStateMixin):

    def _fake(self, checks, install_ok=True):
        calls = {"install": 0}
        pending = list(checks)

        def fake_check(python):
            return pending.pop(0) if pending else (True, "")

        def fake_install(python, log):
            calls["install"] += 1
            if install_ok:
                return True, "Successfully installed rapidocr-onnxruntime"
            return False, "ERROR: no matching distribution for nope"

        self.patch(_check=fake_check, _install=fake_install)
        return calls

    def test_installs_once_then_ready(self):
        calls = self._fake([(False, "ModuleNotFoundError"), (True, "")])
        logs: list[str] = []
        ocr_env._prepare(sys.executable, logs.append)
        self.assertEqual(calls["install"], 1)
        self.assertEqual(ocr_env.status(), ("ready", ""))
        self.assertIsNotNone(ocr_env.client())
        self.assertTrue(any("就绪" in line for line in logs))

    def test_install_failure_marks_failed_for_template_fallback(self):
        self._fake([(False, "no module")], install_ok=False)
        logs: list[str] = []
        ocr_env._prepare(sys.executable, logs.append)
        state, note = ocr_env.status()
        self.assertEqual(state, "failed")
        self.assertIn("no matching distribution", note)
        self.assertIsNone(ocr_env.client())
        self.assertTrue(any("回退模板匹配" in line for line in logs))

    def test_ready_environment_skips_install(self):
        calls = self._fake([(True, "")])
        ocr_env._prepare(sys.executable, None)
        self.assertEqual(calls["install"], 0)
        self.assertEqual(ocr_env.status()[0], "ready")

    def test_prepare_async_waits_for_existing_worker(self):
        ready = ocr_env.OcrClient(sys.executable)
        ocr_env._apply("ready", "", ready)
        calls: list = []
        self.patch(_check=lambda python: calls.append(1) or (True, ""),
                   _install=lambda python, log: (True, ""))
        self.assertEqual(ocr_env.prepare_async(), "ready")
        self.assertEqual(calls, [])
        self.assertIs(ocr_env.client(), ready)


class WorkerProtocolTests(EnvStateMixin):

    def test_worker_speaks_the_client_protocol(self):
        stub = os.path.join(self.tmp, "stub")
        os.mkdir(stub)
        with open(os.path.join(stub, "rapidocr_onnxruntime.py"), "w",
                  encoding="utf-8") as handle:
            handle.write(STUB_RAPIDOCR)
        env = dict(os.environ)
        env["PYTHONPATH"] = stub
        request = json.dumps({"id": "1",
                              "png": _png(np.zeros((12, 30, 3), np.uint8))})
        proc = subprocess.run([sys.executable, "-X", "utf8", ocr_env.WORKER],
                              input=request + "\n", capture_output=True,
                              text=True, encoding="utf-8", timeout=90, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(line) for line in proc.stdout.splitlines() if line]
        self.assertTrue(lines[0]["ready"])
        self.assertEqual(lines[1]["id"], "1")
        self.assertEqual(lines[1]["result"][0][1], "1280")
        self.assertEqual(len(lines[1]["result"][0][0]), 4)

    def test_check_reports_broken_environment(self):
        worker = self.write("bad_worker.py", "import nope_missing_module\n")
        self.patch(WORKER=worker)
        ok, out = ocr_env._check(sys.executable)
        self.assertFalse(ok)
        self.assertIn("nope_missing_module", out)

    def test_check_reports_healthy_environment(self):
        worker = self.write(
            "good_worker.py",
            "import json, sys\n"
            "sys.stdout.write(json.dumps({'ok': True}) + '\\n')\n")
        self.patch(WORKER=worker)
        self.assertEqual(ocr_env._check(sys.executable)[0], True)


class ClientTests(EnvStateMixin):

    def test_client_roundtrip_reuses_one_process(self):
        self.patch(WORKER=self.write("worker.py", STUB_WORKER))
        client = ocr_env.OcrClient(sys.executable)
        self.addCleanup(client.close)
        image = np.zeros((10, 20, 3), np.uint8)
        rows, elapse = client(image)
        box = [[0, 0], [9, 0], [9, 6], [0, 6]]
        self.assertEqual(rows, [[box, "42", 0.91]])
        self.assertGreater(elapse, 0.0)
        again, _ = client(image)
        self.assertEqual(again, rows)
        self.assertIsNone(client._proc.poll())

    def test_client_returns_none_when_worker_cannot_start(self):
        self.patch(WORKER=self.write("dead_worker.py",
                                     "import nope_missing_module\n"))
        client = ocr_env.OcrClient(sys.executable)
        self.addCleanup(client.close)
        self.assertEqual(client(np.zeros((6, 6, 3), np.uint8)), (None, None))

    def test_ocr_lines_falls_back_when_engine_reports_failure(self):
        frame = np.zeros((24, 40, 3), np.uint8)
        self.assertIsNone(detect.ocr_lines(
            frame, None, ocr=lambda image: (None, None)))
        self.assertEqual(detect.ocr_lines(
            frame, None, ocr=lambda image: ([], 0.0)), [])
        self.assertIsNone(detect.ocr_lines(frame, None, ocr=False))


class StubBridge:
    _ensure_ocr = VisionBridge._ensure_ocr

    def __init__(self):
        self._ocr = None
        self._ocr_preparing = False
        self.errors: list[str] = []

    def _err(self, msg: str) -> None:
        self.errors.append(msg)


class BridgeOcrTests(EnvStateMixin):

    def setUp(self):
        super().setUp()
        self.original_get = detect.get_ocr

    def tearDown(self):
        detect.get_ocr = self.original_get
        super().tearDown()

    def test_ensure_ocr_uses_subprocess_client(self):
        sentinel = object()
        calls: list = []
        detect.get_ocr = lambda: None
        self.patch(client=lambda: sentinel,
                   prepare_async=lambda log=None: calls.append(log))
        bridge = StubBridge()
        bridge._ensure_ocr()
        self.assertIs(bridge._ocr, sentinel)
        self.assertEqual(calls, [])

    def test_ensure_ocr_prepares_only_once(self):
        calls: list = []
        detect.get_ocr = lambda: None
        self.patch(client=lambda: None, runner_python=lambda: sys.executable,
                   prepare_async=lambda log=None: calls.append(1) or "preparing")
        bridge = StubBridge()
        bridge._ensure_ocr()
        bridge._ensure_ocr()
        self.assertEqual(calls, [1])
        self.assertTrue(bridge._ocr_preparing)

    def test_ensure_ocr_logs_template_fallback_without_runtime(self):
        detect.get_ocr = lambda: None
        self.patch(client=lambda: None, runner_python=lambda: "")
        bridge = StubBridge()
        bridge._ensure_ocr()
        self.assertIsNone(bridge._ocr)
        self.assertTrue(any("回退模板匹配" in line for line in bridge.errors))


if __name__ == "__main__":
    unittest.main()
