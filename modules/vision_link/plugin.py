
META = {
    "id": "vision_link",
    "name": "画面识别联动",
    "version": "0.4.0",
    "description": "OpenCV 通用画面识别：以「参数名 ← 检测行为」登记实时只读变量"
                   "（检测颜色 / 图片 / 数值 / 数值条，区域例图可截图选取，"
                   "数字/文字 RapidOCR 识别，OCR 跑在应用自带 Python 的子进程里）；"
                   "本模块只发布变量，设备动作请在「事件流」页用写入卡片按这些"
                   "变量编排。",
    "settings_key": "vision_link",
    "actions": [],
    "realtime_manager": True,
    "config": {
        "interval": {
            "label": "检测间隔", "type": "float", "default": 0.5,
            "min": 0.1, "max": 5.0, "step": 0.05, "unit": "s",
            "group": "bridge", "desc": "画面采集与检测周期（后台线程执行）",
        },
        "hold": {
            "label": "失败保持", "type": "float", "default": 1.0,
            "min": 0.0, "max": 10.0, "step": 0.1, "unit": "s",
            "group": "bridge",
            "desc": "数值读不出/截屏失败时沿用上次值的窗口，超时输出 0",
        },
        "debug_dump": {
            "label": "标定调试", "type": "bool", "default": False,
            "group": "bridge",
            "desc": "把最近一帧与检测框写入 config/vision_link/debug/"
                    "（排查识别问题时用）",
        },
    },
}

import os

from plugins import ModuleBase, spec_defaults

from modules.vision_link import detect
from modules.vision_link.bridge import DEFAULTS as _BRIDGE_DEFAULTS
from modules.vision_link.bridge import KIND_LABELS, VisionBridge

_CONFIG_DEFAULTS = spec_defaults(META["config"])

_VALUE_TYPES = {"color": "Bool", "image": "Bool", "bar": "Float"}

_LEGACY_TABLE_KEYS = ("mappings", "outputs")


def detector_value_type(kind: str, fmt: str = "") -> str:
    """检测行为决定的输出类型：颜色 / 图片 / 文字判定为布尔，数值条为浮点。"""
    if kind == "number":
        if fmt == "float":
            return "Float"
        return "Bool" if fmt == "text" else "Int"
    return _VALUE_TYPES.get(kind, "Float")


def drop_legacy_tables(settings, log=None) -> bool:
    """清掉映射表时代留在设置里的行：设备动作已迁到「事件流」的写入卡片。"""
    removed = [key for key in _LEGACY_TABLE_KEYS if key in settings]
    if not removed:
        return False
    for key in removed:
        settings.pop(key, None)
    if hasattr(settings, "save"):
        settings.save()
    if log is not None:
        log("已清除旧版映射表设置（" + "、".join(removed) + "）："
            "本模块只登记画面识别变量，设备动作请在「事件流」页用写入卡片按这些变量编排")
    return True


class VisionLinkModule(ModuleBase):
    id = META["id"]
    name = META["name"]
    version = META["version"]
    description = META["description"]
    settings_key = META["settings_key"]

    def __init__(self):
        self.bridge: VisionBridge | None = None
        self.ctx = None

    def config_spec(self) -> dict:
        return META["config"]

    def template_dir(self) -> str:
        base = os.path.dirname(os.path.abspath(str(
            getattr(self.ctx.settings, "path", "")))) if self.ctx else ""
        return os.path.join(base, "vision_link", "templates")

    def detector_entries(self) -> list[tuple[str, str, str]]:
        """本模块维护的检测参数：(名称, 行为, 数值格式)。"""
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        if self.bridge is not None:
            for det in self.bridge.detectors:
                if det.error or det.name in seen:
                    continue
                seen.add(det.name)
                out.append((det.name, det.kind,
                            str((det.spec or {}).get("fmt") or "")))
            return out
        rows = (self.ctx.settings.get("detectors") or []
                if self.ctx is not None else [])
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            if not str(raw.get("name") or "").strip():
                continue
            try:
                name = detect.parse_name(raw.get("name"))
            except ValueError:
                continue
            kind = str(raw.get("kind") or "")
            if name in seen:
                continue
            seen.add(name)
            out.append((name, kind, str(raw.get("fmt") or "")))
        return out

    def link_params(self) -> list[tuple[str, str]]:
        return [(name, KIND_LABELS.get(kind, kind))
                for name, kind, _fmt in self.detector_entries()]

    def temp_specs(self) -> list[dict]:
        """检测参数由本模块维护：向变量表登记为「可读」并带上输出类型。"""
        return [{"key": name, "label": KIND_LABELS.get(kind, kind),
                 "dir": "in", "type": detector_value_type(kind, fmt),
                 "desc": "画面识别检测值 · 模块每拍维护"}
                for name, kind, fmt in self.detector_entries()]

    def on_load(self, ctx) -> None:
        self.ctx = ctx
        migrate_legacy(ctx.settings, ctx.log)
        drop_legacy_tables(ctx.settings, ctx.log)
        ctx.settings.pop("temps", None)   # 临时变量已并入核心的共享变量表

    def on_unload(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
        self.bridge = None

    async def start(self) -> None:
        if self.bridge is not None and self.bridge.is_running():
            return
        await self.stop()
        cfg = {key: self.ctx.settings.get(key, default)
               for key, default in _BRIDGE_DEFAULTS.items()}
        self.bridge = VisionBridge(self.ctx, cfg)
        await self.bridge.start()
        good = [det for det in self.bridge.detectors if not det.error]
        bad = [det for det in self.bridge.detectors if det.error]
        msg = f"画面识别已启动：{len(good)} 个检测参数"
        if bad:
            msg += (f"；{len(bad)} 个无效（{'、'.join(
                f'{det.name}: {det.error}' for det in bad)}）")
        self.ctx.log(msg)

    async def reload_config(self) -> None:
        if self.bridge is None:
            return
        for key, default in _BRIDGE_DEFAULTS.items():
            self.bridge.config[key] = self.ctx.settings.get(key, default)
        self.bridge.apply_config()

    async def stop(self) -> None:
        if self.bridge is not None:
            await self.bridge.stop()
            self.bridge = None

    def is_running(self) -> bool:
        return self.bridge is not None and self.bridge.is_running()


def _legacy_rect(text: str, ox: int, oy: int):
    parts = [p.strip() for p in str(text or "").split(",") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return [x + ox, y + oy, w, h]


def migrate_legacy(settings, log=None) -> bool:
    if not any(key in settings for key in
               ("templates", "bars", "digits", "region")):
        return False
    region = str(settings.pop("region", "") or "")
    ox, oy, default_rect = 0, 0, None
    parts = [p.strip() for p in region.split(",") if p.strip()]
    if len(parts) == 4:
        try:
            ox, oy, w, h = (int(float(p)) for p in parts)
            if w > 0 and h > 0:
                default_rect = [ox, oy, w, h]
        except ValueError:
            ox, oy = 0, 0

    detectors = [d for d in (settings.get("detectors") or [])
                 if isinstance(d, dict)]

    def _num(text, default):
        try:
            return float(str(text).strip())
        except (TypeError, ValueError):
            return default

    for name, text in (settings.pop("templates", None) or {}).items():
        parts = [p.strip() for p in str(text or "").split(",")]
        if not parts[0]:
            continue
        rect = default_rect
        rest = parts[1:]
        thresh = 0.8
        if rest and rest[0]:
            thresh = min(1.0, max(0.05, _num(rest[0], 0.8)))
            rest = rest[1:]
        if len(rest) >= 4:
            rect = _legacy_rect(",".join(rest[:4]), ox, oy)
        detectors.append({"name": str(name), "kind": "image",
                          "file": parts[0], "thresh": thresh,
                          "rect": rect})

    for name, text in (settings.pop("digits", None) or {}).items():
        parts = [p.strip() for p in str(text or "").split(",")]
        if len(parts) < 4:
            continue
        rect = _legacy_rect(",".join(parts[:4]), ox, oy)
        if rect is None:
            continue
        thresh = min(1.0, max(0.05, _num(parts[4], 0.6))) if len(parts) > 4 \
            else 0.6
        detectors.append({"name": str(name), "kind": "number",
                          "fmt": "int", "thresh": thresh, "rect": rect})

    skipped_bars = len(settings.pop("bars", None) or {})

    used: set[str] = set()
    for det in detectors:
        name = str(det.get("name") or "param")
        if not name or name in used:
            base = name or "param"
            n = 2
            while f"{base}{n}" in used:
                n += 1
            name = f"{base}{n}"
            det["name"] = name
        used.add(name)

    settings["detectors"] = detectors
    if hasattr(settings, "save"):
        settings.save()
    if log is not None:
        log("已将旧版检测器配置迁移为实时参数列表"
            + (f"（{skipped_bars} 个旧数值条为 HSV 口径已停用，"
               "请用锚点例图重新配置）" if skipped_bars else ""))
    return True
