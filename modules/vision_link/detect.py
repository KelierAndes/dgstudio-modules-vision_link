
from __future__ import annotations

import os
import re

import cv2
import numpy as np
from PIL import ImageGrab

from dglab.parsing import IDENT, parse_name, parse_rect

__all__ = ["parse_name", "parse_rect", "parse_color", "grab_frame", "crop",
           "color_present", "match_template", "read_number", "text_present",
           "bar_anchor_ratio", "render_text", "ocr_lines", "get_ocr",
           "ensure_digit_templates", "ensure_dot_template",
           "GLYPH_W", "GLYPH_H", "FONT_CANDIDATES"]

GLYPH_W, GLYPH_H = 36, 48
FONT_CANDIDATES = ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf")

_OCR = None
_OCR_TRIED = False


def get_ocr():
    global _OCR, _OCR_TRIED
    if not _OCR_TRIED:
        _OCR_TRIED = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        except Exception:
            _OCR = None
    return _OCR


def parse_color(color) -> tuple[int, int, int]:
    text = str(color or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if re.fullmatch(r"[0-9A-Fa-f]{6}", text):
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    parts = [p.strip() for p in str(color or "").split(",")]
    if len(parts) == 3:
        try:
            rgb = [max(0, min(255, int(float(p)))) for p in parts]
            return (rgb[0], rgb[1], rgb[2])
        except ValueError:
            pass
    raise ValueError(f"颜色 {color!r} 需为 RRGGBB 或 R,G,B")


def grab_frame(region=None):
    img = ImageGrab.grab(all_screens=True, bbox=region)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def crop(frame, rect):
    if frame is None:
        return None
    fh, fw = frame.shape[:2]
    if rect is None:
        return frame
    x, y, w, h = rect
    x0, y0 = max(0, min(fw, int(x))), max(0, min(fh, int(y)))
    x1, y1 = max(0, min(fw, int(x) + int(w))), max(0, min(fh, int(y) + int(h)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return frame[y0:y1, x0:x1]


def color_present(frame, rect, rgb, tol: float = 40.0,
                  min_ratio: float = 0.05):
    region = crop(frame, rect)
    if region is None:
        return None
    target = np.array([rgb[2], rgb[1], rgb[0]], np.int16)
    diff = np.abs(region.astype(np.int16) - target)
    mask = diff.max(axis=2) <= int(tol)
    total = mask.size
    if total == 0:
        return None
    return (float(np.count_nonzero(mask)) / total) >= float(min_ratio)


def match_template(frame, template, rect=None):
    search = crop(frame, rect)
    if search is None or template is None:
        return None
    th, tw = template.shape[:2]
    if search.shape[0] < th or search.shape[1] < tw:
        return None
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    return max(0.0, float(res.max()))


def _binarize(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    border = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
    if float(np.mean(border)) > 127.0:
        bw = cv2.bitwise_not(bw)
    return bw


def ocr_lines(frame, rect, ocr=None):
    if ocr is False:
        return None
    region = crop(frame, rect)
    if region is None:
        return None
    engine = ocr if ocr is not None else get_ocr()
    if engine is None:
        return None
    prepped = _prep_ocr_image(region)
    for img in ([prepped, region] if prepped is not None else [region]):
        try:
            result, _elapse = engine(img)
        except Exception:
            return None
        rows = _collect_ocr(result)
        if rows:
            return _dedup_lines(rows)
    return []


def _prep_ocr_image(region):
    if region is None or region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 110.0:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    border = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
    if float(np.mean(border)) < 127.0:
        bw = cv2.bitwise_not(bw)
    ys, xs = np.nonzero(bw == 0)
    if xs.size < 8 or ys.size < 8:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    text = bw[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
    th, tw = text.shape
    scale = min(8.0, max(1.0, 48.0 / max(1, th)))
    if scale > 1.05:
        text = cv2.resize(text, (max(1, int(tw * scale)),
                                 max(1, int(th * scale))),
                          interpolation=cv2.INTER_CUBIC)
    out = cv2.copyMakeBorder(text, 20, 20, 20, 20,
                             cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _collect_ocr(result):
    rows = []
    for item in result or []:
        try:
            points = item[0]
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            rows.append((str(item[1]), float(item[2]),
                         (min(xs), min(ys), max(xs), max(ys))))
        except (IndexError, TypeError, ValueError):
            continue
    return rows


def _box_dup(a, b) -> bool:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return False
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return (inter / union >= 0.5
            or inter / max(1e-6, min(area_a, area_b)) >= 0.6)


def _dedup_lines(rows):
    kept = []
    for text, score, box in sorted(rows, key=lambda r: (-len(r[0]), -r[1])):
        if any(_box_dup(box, other) for _t, _s, other in kept):
            continue
        kept.append((text, score, box))
    kept.sort(key=lambda r: (r[2][1], r[2][0]))
    return [(text, score) for text, score, _box in kept]


def read_number(frame, rect, fmt: str, thresh: float = 0.6, ocr=None,
                glyphs: dict | None = None, dot=None):
    lines = ocr_lines(frame, rect, ocr)
    if lines is not None:
        joined = "".join(text for text, score in lines
                         if score >= float(thresh))
        return _parse_number(joined, fmt)
    if glyphs:
        return _read_number_glyphs(frame, rect, glyphs, dot, fmt, thresh)
    return None


def _parse_number(text: str, fmt: str):
    clean = str(text or "").replace(" ", "")
    if not clean:
        return None
    if fmt == "float":
        tokens = re.findall(r"\d+\.\d+|\d+", clean)
        if not tokens:
            return None
        token = next((t for t in tokens if "." in t), tokens[0])
        try:
            return round(float(token), 3)
        except ValueError:
            return None
    tokens = re.findall(r"\d[\d,]*\.?\d*|\.\d+", clean)
    if not tokens:
        return None
    token = tokens[0].replace(",", "").rstrip(".")
    if not token or token == ".":
        return None
    try:
        return int(float(token)) if "." in token else int(token)
    except ValueError:
        return None


def _read_number_glyphs(frame, rect, glyphs: dict, dot, fmt: str,
                        thresh: float = 0.6):
    region = crop(frame, rect)
    if region is None or not glyphs:
        return None
    bw = _binarize(region)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_h = max(6, int(region.shape[0] * 0.3))
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])
    if not boxes:
        return None
    sample = next(iter(glyphs.values()))
    th, tw = sample.shape[:2]
    region_h = region.shape[0]
    chars: list[str] = []
    for x, y, w, h in boxes:
        is_dot = (fmt == "float" and dot is not None
                  and h < min_h and h >= 2 and w <= max(6, min_h // 2)
                  and y + h >= region_h * 0.55)
        if is_dot:
            chars.append(".")
            continue
        if h < min_h or w < 2 or w > region.shape[1]:
            continue
        glyph = cv2.resize(bw[y:y + h, x:x + w], (tw, th),
                           interpolation=cv2.INTER_AREA)
        best_d, best_s = None, -1.0
        for digit, tpl in glyphs.items():
            score = float(cv2.matchTemplate(glyph, tpl,
                                            cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_s:
                best_d, best_s = digit, score
        if best_s < thresh:
            return None
        chars.append(str(best_d))
    if not chars:
        return None
    text = "".join(chars)
    try:
        if fmt == "float":
            return round(float(text), 3)
        return int(text)
    except ValueError:
        return None


def _load_font(height_px: int):
    from PIL import ImageFont
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for name in FONT_CANDIDATES:
        path = os.path.join(windir, "Fonts", name)
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=max(8, int(height_px)))
            except OSError:
                continue
    return None


def render_text(text: str, height_px: int):
    from PIL import Image, ImageDraw
    font = _load_font(height_px)
    if font is None or not str(text):
        return None
    dummy = ImageDraw.Draw(Image.new("L", (4, 4)))
    bbox = dummy.textbbox((0, 0), str(text), font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    img = Image.new("L", (w + 8, h + 8), 0)
    ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), str(text),
                             fill=255, font=font)
    return np.asarray(img)


def text_present(frame, rect, text: str, thresh: float = 0.65, ocr=None,
                 cache: dict | None = None):
    expected = str(text or "").strip()
    if not expected:
        return None
    lines = ocr_lines(frame, rect, ocr)
    if lines is not None:
        joined = "".join(t for t, s in lines if s >= float(thresh))
        return expected.replace(" ", "").lower() in joined.replace(" ", "").lower()
    return _text_present_template(frame, rect, expected, thresh, cache)


def _text_present_template(frame, rect, text: str, thresh: float = 0.65,
                           cache: dict | None = None):
    region = crop(frame, rect)
    if region is None:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    key = "base:" + str(text)
    base = cache.get(key) if cache is not None else None
    if base is None:
        base = render_text(str(text), 48)
        if cache is not None and base is not None:
            cache[key] = base
    if base is None:
        return None
    bh, bw = base.shape
    best = -1.0
    for frac in np.arange(0.3, 1.001, 0.05):
        th = max(8, int(region.shape[0] * float(frac)))
        tw = max(8, int(round(bw * th / bh)))
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue
        tpl = cv2.resize(base, (tw, th), interpolation=cv2.INTER_AREA)
        for base_img in (gray, inverted):
            res = cv2.matchTemplate(base_img, tpl, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
    if best < 0:
        return None
    return best >= thresh


def bar_anchor_ratio(frame, rect, min_pos: float, max_pos: float,
                     anchor, thresh: float = 0.7):
    region = crop(frame, rect)
    if region is None or anchor is None:
        return None
    ah, aw = anchor.shape[:2]
    if region.shape[0] < ah or region.shape[1] < aw:
        return None
    _, score, _, loc = cv2.minMaxLoc(
        cv2.matchTemplate(region, anchor, cv2.TM_CCOEFF_NORMED))
    if score < thresh:
        return None
    span = float(max_pos) - float(min_pos)
    if abs(span) < 1e-6:
        return None
    if region.shape[1] - aw >= region.shape[0] - ah:
        pos = rect[0] + loc[0] + aw / 2.0
    else:
        pos = rect[1] + loc[1] + ah / 2.0
    return max(0.0, min(1.0, (pos - float(min_pos)) / span))


def _gen_digit(digit: int) -> np.ndarray:
    canvas = np.zeros((GLYPH_H * 2, GLYPH_W * 2), np.uint8)
    cv2.putText(canvas, str(digit), (10, GLYPH_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5, cv2.LINE_AA)
    rows, cols = np.nonzero(canvas)
    if rows.size == 0:
        return np.zeros((GLYPH_H, GLYPH_W), np.uint8)
    glyph = canvas[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    img = cv2.resize(glyph, (GLYPH_W, GLYPH_H),
                     interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img


def ensure_digit_templates(directory: str) -> dict:
    os.makedirs(directory, exist_ok=True)
    out = {}
    for digit in range(10):
        path = os.path.join(directory, f"{digit}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = _gen_digit(digit)
            cv2.imwrite(path, img)
        if img.shape[:2] != (GLYPH_H, GLYPH_W):
            img = cv2.resize(img, (GLYPH_W, GLYPH_H))
        out[digit] = img
    return out


def ensure_dot_template(directory: str) -> np.ndarray:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "dot.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((12, 12), np.uint8)
        cv2.circle(img, (6, 6), 4, 255, -1, cv2.LINE_AA)
        cv2.imwrite(path, img)
    return img
