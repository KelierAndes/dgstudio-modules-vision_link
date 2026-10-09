
from __future__ import annotations

import asyncio
import os
import time

import cv2

from dglab.mapping import MappingEngine, signal_specs
from dglab.params import (build_dispatchers, core_alias_values, core_inputs,
                          device_state_values)

from modules.vision_link import detect, ocr_env

DEFAULTS = {
    "interval": 0.5,
    "hold": 1.0,
    "debug_dump": False,
    "detectors": [],
    "mappings": [],
    "outputs": [],
}

_KINDS = ("color", "image", "number", "bar")
KIND_LABELS = {"color": "检测颜色", "image": "检测图片",
               "number": "检测数值", "bar": "检测数值条"}


def _clamp(value, low, high, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(low, min(high, out))


def _int_field(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def normalize_detector(raw) -> tuple[dict | None, str]:
    if not isinstance(raw, dict):
        return None, "配置项需为对象"
    try:
        name = detect.parse_name(raw.get("name"))
    except ValueError as exc:
        return None, str(exc)
    kind = str(raw.get("kind") or "")
    if kind not in _KINDS:
        return None, f"未知检测类型 {kind!r}"
    spec: dict = {"name": name, "kind": kind}
    rect_raw = raw.get("rect")
    if rect_raw is None or rect_raw == "" or rect_raw == [0, 0, 0, 0]:
        if kind != "image":
            return None, "缺少检测区域（x,y,w,h）"
        spec["rect"] = None
    else:
        try:
            spec["rect"] = list(detect.parse_rect(rect_raw))
        except ValueError as exc:
            return None, f"检测区域无效：{exc}"
    if kind == "color":
        try:
            spec["rgb"] = detect.parse_color(raw.get("color"))
        except ValueError as exc:
            return None, str(exc)
        spec["tol"] = _clamp(raw.get("tol"), 0, 255, 40)
        spec["ratio"] = _clamp(raw.get("ratio"), 0.001, 1.0, 0.05)
    elif kind == "image":
        file = str(raw.get("file") or "").strip()
        if not file:
            return None, "缺少例图文件（可截图选取生成）"
        spec["file"] = file
        spec["thresh"] = _clamp(raw.get("thresh"), 0.05, 1.0, 0.8)
    elif kind == "number":
        fmt = str(raw.get("fmt") or "int")
        if fmt not in ("int", "float", "text"):
            return None, f"数值格式 {fmt!r} 需为 int/float/text"
        spec["fmt"] = fmt
        spec["thresh"] = _clamp(raw.get("thresh"), 0.05, 1.0, 0.6)
        if fmt == "text":
            text = str(raw.get("text") or "").strip()
            if not text:
                return None, "文字模式需填写期望文本"
            spec["text"] = text
    elif kind == "bar":
        spec["min"] = _int_field(raw.get("min"), 0)
        spec["max"] = _int_field(raw.get("max"), 100)
        anchor = str(raw.get("anchor") or "").strip()
        if not anchor:
            return None, "缺少锚点例图（取填充色与背景色交界区域）"
        spec["anchor"] = anchor
        spec["thresh"] = _clamp(raw.get("thresh"), 0.05, 1.0, 0.7)
    return spec, ""


class Detector:

    def __init__(self, name: str, kind: str, spec: dict | None):
        self.name = name
        self.kind = kind
        self.spec = spec
        self.error = ""
        self.last_ok: float | None = None
        self.value = 0.0


class _DeviceApi:

    def __init__(self, bridge: "VisionBridge"):
        self._bridge = bridge

    @property
    def _ctx(self):
        return self._bridge.ctx

    def resolve_slot(self, family: str = "") -> str | None:
        return self._ctx.resolve_slot(
            family=str(family or "COYOTE").upper(), output_only=True)

    def wave_order(self, family: str = "") -> list[str]:
        return self._ctx.wave_order(str(family or "COYOTE").upper())

    def wave_selection(self) -> dict:
        return self._ctx.wave_selection() or {}

    def set_strength(self, channel, value, slot_id=None):
        return self._ctx.set_strength(channel, value, slot_id=slot_id)

    def set_wave(self, channel, name, slot_id=None):
        return self._ctx.set_wave(channel, name, slot_id=slot_id)

    def zap(self, channel, seconds=1.0, slot_id=None):
        return self._ctx.zap(channel, seconds, slot_id=slot_id)

    def fire_start(self, slot_id=None, channel=None):
        return self._ctx.fire_start(slot_id=slot_id, channel=channel)

    def fire_stop(self, slot_id=None, channel=None):
        return self._ctx.fire_stop(slot_id=slot_id, channel=channel)

    def emergency_stop(self):
        return self._ctx.emergency_stop()

    def run(self, coro) -> None:
        self._bridge._spawn(coro)


class VisionBridge:
    def __init__(self, ctx, config: dict | None = None):
        self.ctx = ctx
        self.config = dict(DEFAULTS)
        self.config.update({k: v for k, v in (config or {}).items()
                            if k in self.config})
        self.engine = MappingEngine(self._dispatch,
                                    device_vars=self.device_vars)
        self.engine.set_ranges(signal_specs())
        self._api = _DeviceApi(self)
        self.actions = build_dispatchers(self._api, core_inputs())
        self.detectors: list[Detector] = []
        self._templates: dict[str, "cv2.Mat"] = {}
        self._anchors: dict[str, "cv2.Mat"] = {}
        self._digits: dict = {}
        self._dot = None
        self._text_cache: dict = {}
        self._ocr = None
        self._ocr_preparing = False
        self._running = False
        self._task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._primed = False
        self._last_err = 0.0
        self._last_dump = 0.0


    @property
    def data_dir(self) -> str:
        base = os.path.dirname(os.path.abspath(str(
            getattr(self.ctx.settings, "path", ""))))
        return os.path.join(base, "vision_link")

    @property
    def template_dir(self) -> str:
        return os.path.join(self.data_dir, "templates")

    @property
    def debug_dir(self) -> str:
        return os.path.join(self.data_dir, "debug")


    def apply_config(self) -> None:
        first = not self._primed
        if first:
            self.engine.armed = False
        self.engine.set_mappings(self.config.get("mappings") or [])
        self.engine.set_outputs(self.config.get("outputs") or [])
        if first:
            self.engine.armed = True
            self._primed = True
        self.rebuild()

    def rebuild(self) -> None:
        dets: list[Detector] = []
        seen: dict[str, str] = {}
        for raw in (self.config.get("detectors") or []):
            spec, err = normalize_detector(raw)
            if spec is None:
                label = str((raw or {}).get("name") or "(未命名)") \
                    if isinstance(raw, dict) else "(未命名)"
                kind = str((raw or {}).get("kind") or "") \
                    if isinstance(raw, dict) else ""
                dets.append(Detector(label, kind, None))
                dets[-1].error = err
                continue
            det = Detector(spec["name"], spec["kind"], spec)
            owner = seen.get(det.name)
            if owner is not None:
                det.error = f"参数名与{KIND_LABELS[owner]}重复"
            else:
                seen[det.name] = det.kind
            dets.append(det)
        self.detectors = dets
        self._text_cache.clear()
        self._load_assets()

    def _load_assets(self) -> None:
        self._templates = {}
        self._anchors = {}
        self._digits = {}
        self._dot = None
        for det in self.detectors:
            if det.error:
                continue
            if det.kind == "image":
                path = os.path.join(self.template_dir,
                                    str(det.spec.get("file") or ""))
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    det.error = f"例图缺失: {det.spec.get('file')}"
                else:
                    self._templates[det.name] = img
            elif det.kind == "bar":
                path = os.path.join(self.template_dir,
                                    str(det.spec.get("anchor") or ""))
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    det.error = f"锚点例图缺失: {det.spec.get('anchor')}"
                else:
                    self._anchors[det.name] = img
            elif det.kind == "number" and det.spec.get("fmt") != "text":
                try:
                    self._digits = detect.ensure_digit_templates(
                        os.path.join(self.template_dir, "digits"))
                    self._dot = detect.ensure_dot_template(
                        os.path.join(self.template_dir, "digits"))
                except Exception as exc:
                    det.error = f"数字模板不可用: {exc!r}"


    async def start(self) -> None:
        if self._running:
            return
        self.apply_config()
        self._running = True
        self._task = asyncio.ensure_future(self._loop())
        self.ctx.log(f"画面识别采集已启动（间隔 "
                     f"{_clamp(self.config.get('interval'), 0.05, 60, 0.5):g}s）")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self.engine.reset()

    def close(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        ocr_env.stop()
        self.engine.reset()

    def is_running(self) -> bool:
        return self._running

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                await asyncio.sleep(max(0.05, _clamp(
                    self.config.get("interval"), 0.05, 60, 0.5)))
                if not self.detectors:
                    continue
                try:
                    frame = await loop.run_in_executor(None, detect.grab_frame)
                except Exception as exc:
                    self._err(f"画面采集失败: {exc!r}")
                    frame = None
                if frame is None:
                    results = {det.name: self._detect_one(det, None)
                               for det in self.detectors}
                else:
                    try:
                        results = await loop.run_in_executor(
                            None, self._detect_once, frame)
                    except Exception as exc:
                        self._err(f"画面检测失败: {exc!r}")
                        continue
                self._apply(results)
                self.engine.pump()
        except asyncio.CancelledError:
            pass

    def _detect_once(self, frame) -> dict:
        out = {}
        for det in self.detectors:
            out[det.name] = self._detect_one(det, frame)
        if bool(self.config.get("debug_dump")):
            self._dump_debug(frame)
        return out

    def _detect_one(self, det: Detector, frame) -> tuple[bool, float]:
        now = time.monotonic()
        value = None
        if frame is not None and det.spec is not None and not det.error:
            try:
                value = self._measure(det, frame)
            except Exception as exc:
                self._err(f"检测器 {det.name} 异常: {exc!r}")
                value = None
        if value is not None:
            det.last_ok = now
            det.value = float(value)
            return (True, det.value)
        hold = max(0.0, _clamp(self.config.get("hold"), 0, 60, 1.0))
        if det.last_ok is not None and now - det.last_ok <= hold:
            return (False, det.value)
        return (False, 0.0)

    def _ensure_ocr(self) -> None:
        if self._ocr is not None or self._ocr_preparing:
            return
        self._ocr = detect.get_ocr() or ocr_env.client()
        if self._ocr is not None:
            return
        self._ocr_preparing = True
        if not ocr_env.runner_python():
            self._err("RapidOCR 不可用，数字/文字检测回退模板匹配"
                      "（pip install rapidocr-onnxruntime 启用 OCR）")
            return
        ocr_env.prepare_async(self._err)

    def _measure(self, det: Detector, frame):
        spec = det.spec
        if det.kind == "color":
            value = detect.color_present(frame, spec["rect"], spec["rgb"],
                                         spec["tol"], spec["ratio"])
            return None if value is None else (1.0 if value else 0.0)
        if det.kind == "image":
            tpl = self._templates.get(det.name)
            if tpl is None:
                return None
            score = detect.match_template(frame, tpl, spec["rect"])
            if score is None:
                return None
            return 1.0 if score >= spec["thresh"] else 0.0
        if det.kind == "number":
            self._ensure_ocr()
            engine = self._ocr if self._ocr is not None else False
            if spec["fmt"] == "text":
                value = detect.text_present(frame, spec["rect"],
                                            spec["text"], spec["thresh"],
                                            engine, self._text_cache)
                return None if value is None else (1.0 if value else 0.0)
            return detect.read_number(frame, spec["rect"], spec["fmt"],
                                      spec["thresh"], engine,
                                      self._digits, self._dot)
        if det.kind == "bar":
            anchor = self._anchors.get(det.name)
            if anchor is None:
                return None
            return detect.bar_anchor_ratio(frame, spec["rect"],
                                           spec["min"], spec["max"],
                                           anchor, spec["thresh"])
        return None

    def _apply(self, results: dict) -> None:
        for name, (_ok, value) in results.items():
            self.engine.signal(name, round(float(value), 3))


    def _dump_debug(self, frame) -> None:
        now = time.monotonic()
        if frame is None or now - self._last_dump < 1.0:
            return
        self._last_dump = now
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            vis = frame.copy()
            for det in self.detectors:
                if det.spec is None or det.spec.get("rect") is None:
                    continue
                x, y, w, h = det.spec["rect"]
                color = (0, 255, 0) if not det.error else (0, 0, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
                cv2.putText(vis, det.name, (x, max(10, y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                            cv2.LINE_AA)
                region = detect.crop(frame, det.spec["rect"])
                if region is not None:
                    cv2.imwrite(os.path.join(self.debug_dir,
                                             f"{det.name}.png"), region)
            cv2.imwrite(os.path.join(self.debug_dir, "frame.png"), frame)
            cv2.imwrite(os.path.join(self.debug_dir, "boxes.png"), vis)
        except Exception:
            pass

    def _err(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_err < 10.0:
            return
        self._last_err = now
        self.ctx.log(msg)

    def _dispatch(self, target: str, value: int) -> None:
        action = self.actions.get(target)
        if action is None:
            return
        try:
            action(value)
        except Exception as exc:
            self.ctx.log(f"画面识别派发 {target}={value} 失败: {exc!r}")

    def device_vars(self) -> dict:
        try:
            state = self.ctx.get_state()
        except Exception:
            return {}
        vals = device_state_values(state)
        vals.update(core_alias_values(vals))
        return vals
