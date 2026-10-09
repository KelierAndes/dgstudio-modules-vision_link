"""OCR 子进程：由应用内置 Python 运行，逐行收发 JSON。

图片以 base64 PNG 传递（避免临时文件与中文路径问题），结果结构与 RapidOCR
一致（[[[x, y] × 4, 文本, 置信度], ...]），画面识别侧不需要区分引擎在哪。
"""

from __future__ import annotations

import base64
import json
import sys
import time


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _rows(result) -> list:
    rows = []
    for item in result or []:
        try:
            box, text, score = item[0], item[1], item[2]
            rows.append([[[float(p[0]), float(p[1])] for p in box],
                         str(text), float(score)])
        except (IndexError, TypeError, ValueError):
            continue
    return rows


def main() -> int:
    try:
        import cv2
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    if "--check" in sys.argv:
        _emit({"ok": True})
        return 0
    try:
        engine = RapidOCR()
    except Exception as exc:
        _emit({"ok": False, "error": f"模型初始化失败 {type(exc).__name__}: {exc}"})
        return 1
    _emit({"ok": True, "ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except ValueError:
            _emit({"id": None, "error": "请求不是合法 JSON"})
            continue
        started = time.perf_counter()
        try:
            raw = base64.b64decode(job.get("png") or "")
            image = cv2.imdecode(np.frombuffer(bytearray(raw), np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("PNG 解码失败")
            result, _elapse = engine(image)
            _emit({"id": job.get("id"), "result": _rows(result),
                   "elapse": round(time.perf_counter() - started, 3)})
        except Exception as exc:
            _emit({"id": job.get("id"),
                   "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
