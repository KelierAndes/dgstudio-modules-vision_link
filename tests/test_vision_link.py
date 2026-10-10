from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401  定位 DGStudio 核心仓库

import asyncio
import os
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from dglab.state import EngineState, Slot
from modules.vision_link import detect
from modules.vision_link.bridge import (Detector, VisionBridge,
                                        normalize_detector)
from modules.vision_link.plugin import (VisionLinkModule, drop_legacy_tables,
                                        migrate_legacy)


class DeviceWrite(Exception):
    """桩上下文抛出它：模块一旦直写设备，测试立刻失败。"""


def _deny(name: str):

    def deny(self, *args, **kwargs):
        raise DeviceWrite(f"模块不得直写设备：{name}()")
    return deny


BLOCKED_DEVICE_METHODS = ("set_strength", "add_strength", "reset_strength",
                          "set_wave", "push_pulse_stream", "fire",
                          "fire_start", "fire_stop", "zap",
                          "set_intensity_param")


class FakeCtx:
    """ModuleContext 桩：放行读状态 / 登记变量，拦截全部设备直写。"""

    def __init__(self, settings_path: str | None = None):
        self.logs: list[str] = []
        self.emergency_calls = 0
        self.state = EngineState(
            connected=True, paired=True,
            slots={"s1": Slot(slot_id="s1", name="Coyote 03", type="COYOTE",
                              strength={"A": 0, "B": 0}, battery=77)})
        from plugins import JsonDict
        self.settings = JsonDict(settings_path or os.path.join(
            tempfile.mkdtemp(), "vision_link.json"), {})

    def log(self, msg):
        self.logs.append(msg)

    def resolve_slot(self, slot_id=None, family=None, output_only=False):
        return "s1"

    def get_state(self):
        return self.state

    def wave_order(self, family="COYOTE"):
        return ["静默", "呼吸", "波浪"]

    def wave_selection(self) -> dict:
        return {"A": "呼吸", "B": ""}

    def emergency_stop(self):
        self.emergency_calls += 1

    set_strength = _deny("set_strength")
    add_strength = _deny("add_strength")
    reset_strength = _deny("reset_strength")
    set_wave = _deny("set_wave")
    push_pulse_stream = _deny("push_pulse_stream")
    fire = _deny("fire")
    fire_start = _deny("fire_start")
    fire_stop = _deny("fire_stop")
    zap = _deny("zap")
    set_intensity_param = _deny("set_intensity_param")


class NormalizeTests(unittest.TestCase):
    def test_color_defaults(self):
        spec, err = normalize_detector(
            {"name": "c1", "kind": "color", "rect": "1,2,30,40",
             "color": "#FF0000"})
        self.assertEqual(err, "")
        self.assertEqual(spec["rect"], [1, 2, 30, 40])
        self.assertEqual(spec["rgb"], (255, 0, 0))
        self.assertEqual(spec["tol"], 40.0)
        self.assertEqual(spec["ratio"], 0.05)

    def test_number_text_and_float(self):
        spec, err = normalize_detector(
            {"name": "n1", "kind": "number", "rect": [0, 0, 50, 20],
             "fmt": "text", "text": " RELOAD "})
        self.assertEqual(err, "")
        self.assertEqual(spec["text"], "RELOAD")
        spec, err = normalize_detector(
            {"name": "n1", "kind": "number", "rect": [0, 0, 50, 20],
             "fmt": "float"})
        self.assertEqual(err, "")
        self.assertEqual(spec["fmt"], "float")

    def test_bar_defaults(self):
        spec, err = normalize_detector(
            {"name": "b1", "kind": "bar", "rect": [0, 0, 200, 20],
             "anchor": "a.png"})
        self.assertEqual(err, "")
        self.assertEqual(spec["min"], 0)
        self.assertEqual(spec["max"], 100)
        self.assertEqual(spec["thresh"], 0.7)

    def test_image_rect_optional(self):
        spec, err = normalize_detector(
            {"name": "i1", "kind": "image", "file": "x.png"})
        self.assertEqual(err, "")
        self.assertIsNone(spec["rect"])

    def test_errors(self):
        cases = [
            {"name": "1bad", "kind": "color", "rect": [0, 0, 9, 9]},
            {"name": "ok", "kind": "magic", "rect": [0, 0, 9, 9]},
            {"name": "ok", "kind": "color", "rect": None},
            {"name": "ok", "kind": "color", "rect": "1,2,3"},
            {"name": "ok", "kind": "color", "rect": [0, 0, 9, 9],
             "color": "nope"},
            {"name": "ok", "kind": "image", "file": ""},
            {"name": "ok", "kind": "number", "rect": [0, 0, 9, 9],
             "fmt": "hex"},
            {"name": "ok", "kind": "number", "rect": [0, 0, 9, 9],
             "fmt": "text", "text": ""},
            {"name": "ok", "kind": "bar", "rect": [0, 0, 9, 9],
             "anchor": ""},
        ]
        for raw in cases:
            spec, err = normalize_detector(raw)
            self.assertIsNone(spec, raw)
            self.assertTrue(err, raw)

    def test_parse_color(self):
        self.assertEqual(detect.parse_color("#1020FF"), (16, 32, 255))
        self.assertEqual(detect.parse_color("10,20,30"), (10, 20, 30))
        with self.assertRaises(ValueError):
            detect.parse_color("zz")


class ColorDetectTests(unittest.TestCase):
    def test_present_and_absent(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[10:60, 10:60] = (0, 0, 255)
        self.assertIs(detect.color_present(frame, (0, 0, 100, 100),
                                           (255, 0, 0), 40, 0.05), True)
        self.assertIs(detect.color_present(frame, (0, 0, 100, 100),
                                           (0, 255, 0), 40, 0.05), False)

    def test_out_of_frame(self):
        self.assertIsNone(detect.color_present(
            np.zeros((50, 50, 3), np.uint8), (100, 100, 20, 20),
            (255, 0, 0), 40, 0.05))


class TemplateDetectTests(unittest.TestCase):
    def test_match_found_and_absent(self):
        rng = np.random.default_rng(7)
        patch = rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)
        frame = np.zeros((150, 200, 3), np.uint8)
        frame[60:90, 100:140] = patch
        self.assertGreater(detect.match_template(frame, patch), 0.99)
        self.assertLess(detect.match_template(
            np.zeros((150, 200, 3), np.uint8), patch), 0.5)


class NumberDetectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.digits = detect.ensure_digit_templates(
            os.path.join(self.tmp, "digits"))
        self.dot = detect.ensure_dot_template(
            os.path.join(self.tmp, "digits"))
        self.ocr = detect.get_ocr()

    def _frame_with(self, text):
        frame = np.zeros((80, 260, 3), np.uint8)
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (255, 255, 255), 5, cv2.LINE_AA)
        return frame

    def _rendered(self, text):
        tpl = detect.render_text(text, 34)
        th, tw = tpl.shape
        frame = np.zeros((th + 30, tw + 40, 3), np.uint8)
        for c in range(3):
            frame[15:15 + th, 20:20 + tw, c] = tpl
        return frame

    def test_read_int_glyph_fallback(self):
        for text in ("42", "0", "7", "109"):
            value = detect.read_number(self._frame_with(text), None, "int",
                                       0.6, False, self.digits, self.dot)
            self.assertEqual(value, int(text), text)

    def test_read_float_glyph_fallback(self):
        value = detect.read_number(self._frame_with("3.5"), None, "float",
                                   0.6, False, self.digits, self.dot)
        self.assertAlmostEqual(value, 3.5, places=3)

    def test_glyph_fallback_without_templates(self):
        self.assertIsNone(detect.read_number(
            np.zeros((80, 260, 3), np.uint8), None, "int", 0.6,
            False, {}, None))

    def test_parse_number_rules(self):
        self.assertEqual(detect._parse_number("1,280 HP", "int"), 1280)
        self.assertEqual(detect._parse_number("HP 62", "int"), 62)
        self.assertEqual(detect._parse_number("62.5%", "float"), 62.5)
        self.assertEqual(detect._parse_number("33", "float"), 33.0)
        self.assertIsNone(detect._parse_number("HP", "int"))
        self.assertIsNone(detect._parse_number("", "float"))

    @unittest.skipIf(detect.get_ocr() is None,
                     "rapidocr-onnxruntime 未安装")
    def test_read_int_ocr(self):
        for text in ("42", "1280", "109"):
            value = detect.read_number(self._rendered(text), None, "int",
                                       0.5, self.ocr)
            self.assertEqual(value, int(text), text)

    @unittest.skipIf(detect.get_ocr() is None,
                     "rapidocr-onnxruntime 未安装")
    def test_dark_hud_wide_region(self):
        frame = np.full((101, 538, 3), 32, np.uint8)
        tpl = detect.render_text("96", 44)
        th, tw = tpl.shape
        frame[20:20 + th, 60:60 + tw] = np.stack([tpl] * 3, axis=2)
        value = detect.read_number(frame, None, "int", 0.5, self.ocr)
        self.assertEqual(value, 96)

    @unittest.skipIf(detect.get_ocr() is None,
                     "rapidocr-onnxruntime 未安装")
    def test_read_float_ocr(self):
        value = detect.read_number(self._rendered("62.5"), None, "float",
                                   0.5, self.ocr)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, 62.5, places=1)

    @unittest.skipIf(detect.get_ocr() is None,
                     "rapidocr-onnxruntime 未安装")
    def test_text_present_ocr(self):
        frame = self._rendered("START")
        self.assertIs(detect.text_present(frame, None, "start", 0.5, self.ocr),
                      True)
        self.assertIs(detect.text_present(frame, None, "PAUSE", 0.5, self.ocr),
                      False)

    def test_ocr_lines_empty_region(self):
        self.assertIsNone(detect.ocr_lines(
            np.zeros((50, 50, 3), np.uint8), (500, 500, 10, 10), self.ocr))


class PreprocessTests(unittest.TestCase):
    def test_prep_light_on_dark(self):
        region = np.full((60, 200, 3), 32, np.uint8)
        tpl = detect.render_text("96", 30)
        th, tw = tpl.shape
        region[10:10 + th, 30:30 + tw] = np.stack([tpl] * 3, axis=2)
        out = detect._prep_ocr_image(region)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[2], 3)
        border = np.concatenate([out[0, :], out[-1, :], out[:, 0],
                                 out[:, -1]])
        self.assertGreater(float(np.mean(border)), 200.0)

    def test_prep_empty_region(self):
        self.assertIsNone(detect._prep_ocr_image(
            np.zeros((60, 200, 3), np.uint8)))

    def test_dedup_overlapping_boxes(self):
        rows = [("9", 0.98, (10, 0, 40, 50)),
                ("96", 0.95, (10, 0, 90, 50))]
        lines = detect._dedup_lines(rows)
        self.assertEqual([text for text, _s in lines], ["96"])

    def test_dedup_disjoint_sorted_by_position(self):
        rows = [("6", 0.9, (60, 0, 90, 50)),
                ("9", 0.9, (10, 0, 40, 50))]
        lines = detect._dedup_lines(rows)
        self.assertEqual([text for text, _s in lines], ["9", "6"])


class TextDetectTests(unittest.TestCase):
    def _frame_with(self, text, height=30):
        tpl = detect.render_text(text, height)
        self.assertIsNotNone(tpl)
        th, tw = tpl.shape
        frame = np.zeros((th + 40, tw + 60, 3), np.uint8)
        for c in range(3):
            frame[20:20 + th, 30:30 + tw, c] = tpl
        return frame

    def test_present_and_absent_template_fallback(self):
        cache: dict = {}
        frame = self._frame_with("START")
        self.assertIs(detect._text_present_template(frame, None, "START",
                                                    0.65, cache), True)
        self.assertIs(detect._text_present_template(frame, None, "PAUSE",
                                                    0.65, cache), False)

    def test_render_missing_font_returns_none(self):
        import unittest.mock as mock
        with mock.patch.object(detect, "FONT_CANDIDATES", ("nope.ttf",)):
            self.assertIsNone(detect.render_text("X", 20))


class BarDetectTests(unittest.TestCase):
    def _frame_and_anchor(self, ox=0, oy=0):
        frame = np.zeros((40 + oy + 10, 200 + ox + 10, 3), np.uint8)
        frame[oy:oy + 40, ox:ox + 200] = (0, 0, 0)
        frame[oy:oy + 40, ox:ox + 100] = (0, 0, 255)
        anchor = frame[oy:oy + 40, ox + 96:ox + 104].copy()
        return frame, anchor

    def test_ratio_midpoint_absolute(self):
        frame, anchor = self._frame_and_anchor()
        ratio = detect.bar_anchor_ratio(frame, (0, 0, 200, 40), 0, 200,
                                        anchor, 0.7)
        self.assertAlmostEqual(ratio, 0.5, places=2)

    def test_ratio_absolute_coords_offset_region(self):
        frame, anchor = self._frame_and_anchor(ox=50, oy=30)
        ratio = detect.bar_anchor_ratio(frame, (50, 30, 200, 40), 50, 250,
                                        anchor, 0.7)
        self.assertAlmostEqual(ratio, 0.5, places=2)

    def test_ratio_min_max_window(self):
        frame, anchor = self._frame_and_anchor()
        ratio = detect.bar_anchor_ratio(frame, (0, 0, 200, 40), 20, 180,
                                        anchor, 0.7)
        self.assertAlmostEqual(ratio, (100 - 20) / 160.0, places=2)

    def test_ratio_clamped(self):
        frame, anchor = self._frame_and_anchor()
        ratio = detect.bar_anchor_ratio(frame, (0, 0, 200, 40), 150, 200,
                                        anchor, 0.7)
        self.assertEqual(ratio, 0.0)

    def test_anchor_absent_fails(self):
        frame, anchor = self._frame_and_anchor()
        plain = np.zeros((40, 200, 3), np.uint8)
        self.assertIsNone(detect.bar_anchor_ratio(plain, (0, 0, 200, 40),
                                                  0, 200, anchor, 0.7))


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    def make_bridge(self, config=None):
        self.ctx = FakeCtx()
        return VisionBridge(self.ctx, config or {})

    def test_rebuild_flags_errors(self):
        bridge = self.make_bridge({"detectors": [
            {"name": "1bad", "kind": "color", "rect": [0, 0, 9, 9],
             "color": "FF0000"},
            {"name": "c1", "kind": "color", "rect": [0, 0, 9, 9],
             "color": "FF0000"},
            {"name": "c1", "kind": "color", "rect": [1, 1, 9, 9],
             "color": "00FF00"},
            {"name": "i1", "kind": "image", "file": "missing.png"},
            {"name": "n1", "kind": "number", "rect": [0, 0, 9, 9]},
        ]})
        bridge.rebuild()
        by_name = lambda name: [det for det in bridge.detectors
                                if det.name == name]
        self.assertNotEqual(by_name("1bad")[0].error, "")
        self.assertEqual(by_name("c1")[0].error, "")
        self.assertIn("重复", by_name("c1")[1].error)
        self.assertIn("例图缺失", by_name("i1")[0].error)
        self.assertEqual(by_name("n1")[0].error, "")

    def test_measure_color(self):
        bridge = self.make_bridge()
        det = Detector("c1", "color",
                       {"name": "c1", "kind": "color", "rect": [0, 0, 50, 50],
                        "rgb": (255, 0, 0), "tol": 40.0, "ratio": 0.05})
        frame = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(bridge._measure(det, frame), 0.0)
        frame[0:50, 0:50] = (0, 0, 255)
        self.assertEqual(bridge._measure(det, frame), 1.0)

    def test_measure_number_glyph_fallback(self):
        bridge = self.make_bridge()
        det = Detector("n1", "number",
                       {"name": "n1", "kind": "number", "rect": None,
                        "fmt": "int", "thresh": 0.6})
        bridge._digits = detect.ensure_digit_templates(
            os.path.join(bridge.template_dir, "digits"))
        bridge._dot = detect.ensure_dot_template(
            os.path.join(bridge.template_dir, "digits"))
        bridge._ocr_preparing = True
        bridge._ocr = None
        frame = np.zeros((80, 260, 3), np.uint8)
        cv2.putText(frame, "42", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (255, 255, 255), 5, cv2.LINE_AA)
        self.assertEqual(bridge._measure(det, frame), 42)

    def test_measure_uses_ocr_when_available(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            self.skipTest("rapidocr-onnxruntime 未安装")
        bridge = self.make_bridge()
        det = Detector("n1", "number",
                       {"name": "n1", "kind": "number", "rect": None,
                        "fmt": "int", "thresh": 0.5})
        tpl = detect.render_text("1280", 34)
        th, tw = tpl.shape
        frame = np.zeros((th + 30, tw + 40, 3), np.uint8)
        for c in range(3):
            frame[15:15 + th, 20:20 + tw, c] = tpl
        value = bridge._measure(det, frame)
        self.assertEqual(value, 1280)
        self.assertTrue(bridge._ocr_preparing or bridge._ocr is not None)
        self.assertIsNotNone(bridge._ocr)

    async def test_detector_values_publish_to_signals(self):
        bridge = self.make_bridge()
        bridge.apply_config()
        bridge._apply({"hp": (True, 62.5), "reload": (False, 0.0)})
        self.assertEqual(bridge.engine.signals["hp"], 62.5)
        self.assertEqual(bridge.engine.signals["reload"], 0.0)
        bridge.close()

    def test_engine_is_signal_board_only(self):
        from modules.vision_link.bridge import SignalBoard

        bridge = self.make_bridge()
        self.assertIsInstance(bridge.engine, SignalBoard)
        self.assertIsInstance(bridge.engine.signals, dict)
        self.assertIsInstance(bridge.engine.errors, dict)
        bridge.engine.signal("hp", "12")
        self.assertEqual(bridge.engine.signals["hp"], 12.0)
        bridge.engine.signal("hp", "nope")
        self.assertEqual(bridge.engine.signals["hp"], 12.0)
        bridge.engine.pump()
        bridge.engine.reset()
        self.assertEqual(bridge.engine.signals, {})
        bridge.close()

    def test_hold_semantics(self):
        bridge = self.make_bridge({"hold": 1.0})
        det = Detector("n1", "number",
                       {"name": "n1", "kind": "number", "rect": [0, 0, 9, 9],
                        "fmt": "int", "thresh": 0.6})
        ok, value = bridge._detect_one(det, None)
        self.assertEqual((ok, value), (False, 0.0))
        det.value, det.last_ok = 42.0, time.monotonic()
        ok, value = bridge._detect_one(det, None)
        self.assertEqual((ok, value), (False, 42.0))
        det.last_ok = time.monotonic() - 5.0
        ok, value = bridge._detect_one(det, None)
        self.assertEqual((ok, value), (False, 0.0))

    async def test_start_stop(self):
        module = VisionLinkModule()
        ctx = FakeCtx()
        module.on_load(ctx)
        await module.start()
        self.assertTrue(module.is_running())
        self.assertIn("画面识别已启动", " ".join(ctx.logs))
        await module.stop()
        self.assertFalse(module.is_running())
        self.assertIsNone(module.bridge)

    def test_signals_are_the_only_public_surface(self):
        bridge = self.make_bridge()
        self.assertEqual(bridge.engine.signals, {})
        self.assertFalse(hasattr(bridge.engine, "mappings"))
        self.assertFalse(hasattr(bridge.engine, "outputs"))
        self.assertFalse(hasattr(bridge, "actions"))
        self.assertFalse(hasattr(bridge, "_dispatch"))
        self.assertFalse(hasattr(bridge, "device_vars"))
        bridge.close()


class NoDeviceWriteGuardTests(unittest.IsolatedAsyncioTestCase):
    """架构契约：模块只登记变量。桩上下文对任何设备直写都抛错，跑一整轮采集不许触发。"""

    async def test_full_capture_cycle_never_writes_devices(self):
        frame = np.zeros((120, 120, 3), np.uint8)
        frame[10:60, 10:60] = (0, 0, 255)
        ctx = FakeCtx()
        bridge = VisionBridge(ctx, {
            "interval": 0.05,
            "detectors": [
                {"name": "red", "kind": "color", "rect": [0, 0, 100, 100],
                 "color": "#FF0000", "tol": 40, "ratio": 0.05},
                {"name": "ammo", "kind": "number", "rect": [0, 0, 90, 30],
                 "fmt": "int"},
            ]})
        with mock.patch.object(detect, "grab_frame",
                               new=lambda *a, **k: frame.copy()):
            await bridge.start()
            try:
                seen: dict[str, float] = {}
                for _ in range(80):
                    await asyncio.sleep(0.05)
                    seen = dict(bridge.engine.signals)
                    if {"red", "ammo"} <= set(seen):
                        break
            finally:
                await bridge.stop()
        self.assertEqual(seen.get("red"), 1.0, seen)
        self.assertIn("ammo", seen)
        self.assertEqual(bridge.engine.errors, {})
        self.assertEqual(ctx.emergency_calls, 0)
        self.assertNotIn("mappings", bridge.config)
        self.assertNotIn("outputs", bridge.config)

    def test_stub_ctx_blocks_every_forbidden_method(self):
        ctx = FakeCtx()
        for name in BLOCKED_DEVICE_METHODS:
            with self.assertRaises(DeviceWrite):
                getattr(ctx, name)("A")
        ctx.emergency_stop()          # 急停是安全通道，仍然放行
        self.assertEqual(ctx.emergency_calls, 1)


class MigrationTests(unittest.TestCase):
    def test_legacy_dsl_to_detectors(self):
        settings = {
            "region": "100,200,800,600",
            "templates": {"icon": "i.png,0.9,10,20,300,40"},
            "digits": {"ammo": "400,50,120,30,0.7"},
            "bars": {"old": "0,0,100,20,0,179,0,255,0,255"},
        }
        logs: list[str] = []
        self.assertTrue(migrate_legacy(settings, logs.append))
        for key in ("region", "templates", "bars", "digits"):
            self.assertNotIn(key, settings)
        dets = {d["name"]: d for d in settings["detectors"]}
        icon = dets["icon"]
        self.assertEqual(icon["kind"], "image")
        self.assertEqual(icon["file"], "i.png")
        self.assertEqual(icon["thresh"], 0.9)
        self.assertEqual(icon["rect"], [110, 220, 300, 40])
        ammo = dets["ammo"]
        self.assertEqual(ammo["kind"], "number")
        self.assertEqual(ammo["rect"], [500, 250, 120, 30])
        self.assertEqual(ammo["fmt"], "int")
        self.assertTrue(any("数值条" in msg for msg in logs))
        self.assertFalse(migrate_legacy(settings, logs.append))

    def test_no_region_full_frame(self):
        settings = {"templates": {"icon": "i.png"}}
        self.assertTrue(migrate_legacy(settings))
        det = settings["detectors"][0]
        self.assertIsNone(det["rect"])
        self.assertEqual(det["thresh"], 0.8)

    def test_duplicate_names_made_unique(self):
        settings = {"templates": {"hp": "a.png", "hp": "b.png"}}
        self.assertTrue(migrate_legacy(settings))
        names = [d["name"] for d in settings["detectors"]]
        self.assertEqual(len(names), len(set(names)))


class PluginTests(unittest.TestCase):
    def test_meta_literal_and_class(self):
        import _bootstrap
        import plugins

        path = os.path.join(_bootstrap.HERE,
                            "modules", "vision_link", "plugin.py")
        meta = plugins._read_meta(path)
        self.assertEqual(meta["id"], "vision_link")
        self.assertEqual(meta["settings_key"], "vision_link")
        self.assertTrue(meta["realtime_manager"])
        self.assertEqual(meta["version"], "0.4.1")
        self.assertNotIn("mappings", meta["config"])
        self.assertNotIn("outputs", meta["config"])
        self.assertIn("事件流", meta["description"])
        self.assertNotIn("映射表", meta["description"])
        self.assertIn("interval", meta["config"])
        self.assertIn("hold", meta["config"])
        self.assertIn("debug_dump", meta["config"])
        self.assertNotIn("templates", meta["config"])
        self.assertNotIn("bars", meta["config"])
        self.assertNotIn("digits", meta["config"])
        cls = plugins._load_plugin_class("vision_link", path)
        self.assertEqual(cls.__name__, "VisionLinkModule")
        inst = cls()
        self.assertFalse(inst.is_running())
        self.assertIsInstance(inst, plugins.ModuleBase)

    def test_realtime_manager_flag_passthrough(self):
        import tempfile as tf

        from dglab.state import StateEvents
        from plugins import PluginManager

        class _Cfg(dict):
            path = os.path.join(tf.mkdtemp(prefix="vl_meta_"), "config.json")

            def save(self):
                pass

        class _Engine:
            def __init__(self):
                self.events = StateEvents()
                self.config = _Cfg()

            def _log(self, msg):
                pass

        import unittest.mock

        import _bootstrap
        import plugins

        roots = unittest.mock.patch("plugins.module_roots",
                                    return_value=[os.path.join(_bootstrap.HERE,
                                                               "modules")])
        roots.start()
        self.addCleanup(roots.stop)
        manager = PluginManager(_Engine())
        self.assertTrue(manager.meta("vision_link")["realtime_manager"])
        self.assertIsNone(manager.meta("strength_logger"))

    def test_link_params_from_settings(self):
        ctx = FakeCtx()
        ctx.settings["detectors"] = [
            {"name": "hp_icon", "kind": "image"},
            {"name": "hp_bar", "kind": "bar"},
            {"name": "1bad", "kind": "color"},
        ]
        module = VisionLinkModule()
        module.on_load(ctx)
        self.assertEqual(module.link_params(),
                         [("hp_icon", "检测图片"), ("hp_bar", "检测数值条")])

    def test_detector_vars_register_as_readable(self):
        ctx = FakeCtx()
        ctx.settings["detectors"] = [{"name": "hp_icon", "kind": "image"}]
        ctx.settings["temps"] = [{"name": "leftover"}]
        module = VisionLinkModule()
        module.on_load(ctx)
        specs = {spec["key"]: spec for spec in module.temp_specs()}
        self.assertEqual(specs["hp_icon"]["dir"], "in")
        self.assertEqual(specs["hp_icon"]["label"], "检测图片")
        self.assertNotIn("leftover", specs)
        self.assertNotIn("temps", ctx.settings)

    def test_on_load_drops_legacy_mapping_tables(self):
        ctx = FakeCtx()
        ctx.settings["mappings"] = [{"param": "in_strength_a", "expr": "{hp}"}]
        ctx.settings["outputs"] = [{"name": "hp_out", "expr": "{hp}"}]
        ctx.settings["interval"] = 0.7
        saved: list[bool] = []
        ctx.settings.save = lambda: saved.append(True)
        module = VisionLinkModule()
        module.on_load(ctx)
        self.assertNotIn("mappings", ctx.settings)
        self.assertNotIn("outputs", ctx.settings)
        self.assertEqual(ctx.settings["interval"], 0.7)
        self.assertGreaterEqual(len(saved), 1)
        self.assertEqual(len(ctx.logs), 1)
        self.assertIn("事件流", ctx.logs[0])
        self.assertIn("写入卡片", ctx.logs[0])

    def test_on_load_stays_quiet_without_legacy_tables(self):
        ctx = FakeCtx()
        VisionLinkModule().on_load(ctx)
        self.assertEqual(ctx.logs, [])

    def test_drop_legacy_tables_helper_is_idempotent(self):
        settings = {"mappings": [{"param": "in_fire", "expr": "{x}"}]}
        logs: list[str] = []
        self.assertTrue(drop_legacy_tables(settings, logs.append))
        self.assertFalse(drop_legacy_tables(settings, logs.append))
        self.assertEqual(len(logs), 1)
        self.assertFalse(drop_legacy_tables({}, logs.append))
        self.assertEqual(len(logs), 1)

    def test_template_dir_follows_settings(self):
        ctx = FakeCtx()
        module = VisionLinkModule()
        module.on_load(ctx)
        base = os.path.dirname(os.path.abspath(
            str(ctx.settings.path)))
        self.assertEqual(module.template_dir(),
                         os.path.join(base, "vision_link", "templates"))


if __name__ == "__main__":
    unittest.main()
