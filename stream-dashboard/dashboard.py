#!/usr/bin/env python3
"""
Stream Dashboard (standalone)
-----------------------------
Watch several YouTube live trading streams on one page. Each stream gets:
  - the live video (YouTube player, muted until you unmute it)
  - a scrollable live transcript beside it (Whisper, running on your PC)
  - a position monitor: spots TradingView Long/Short Position boxes by colour,
    highlights the stream (green = long, red = short), beeps, and keeps a
    snapshot of the chart at the moment the box appeared.

No AI service, no API key, no Telegram: everything runs on your PC for free.
This is separate from the Stream Order Watcher app; both can run at the same time.

Add a stream by pasting a YouTube link at the top of the page; remove one with
the x on its card. Added/removed streams last for this session only;
dashboard.yaml decides what loads at start.

Run:   py dashboard.py      ->  opens http://localhost:8800
"""

import base64
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

import cv2
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 3

log = logging.getLogger("dashboard")
LOCK = threading.Lock()
STATE = {}                 # per-stream state shown on the page (insertion order = grid order)
MONITORS = {}              # name -> Monitor
FRAMES = {}                # latest JPEG per stream (memory only)
SNAPS = {}                 # JPEG at the moment each position opened (memory only)
ALERTS = deque(maxlen=50)  # recent position events across all streams
CFG = {}
TRANSCRIBER = None
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
ON_REPLIT = bool(os.environ.get("REPL_ID") or os.environ.get("REPLIT_DEV_DOMAIN"))


LONG_WORDS = r"\b(long|longs|buy|buying|bought|calls?)\b"
SHORT_WORDS = r"\b(short|shorts|shorting|sell|selling|sold|puts?)\b"
ENTRY_WORDS = r"\b(entered|entry|i'?m in|i am in|got in|filled|took (it|this|the trade)|taking (it|this)|in a (trade|position))\b"


def confirms(text, side):
    """Does this transcript line confirm a position in this direction?"""
    low = text.lower()
    own, other = (LONG_WORDS, SHORT_WORDS) if side == "long" else (SHORT_WORDS, LONG_WORDS)
    if re.search(own, low):
        return True
    return bool(re.search(ENTRY_WORDS, low)) and not re.search(other, low)


def mark_confirmations(st, lines):
    """Tag transcript lines near the position's opening time that confirm it.
    Call with LOCK held."""
    p = st.get("position")
    if not p or p.get("direction") not in ("long", "short"):
        return
    window = CFG.get("dashboard", {}).get("confirm_window_seconds", 90)
    for line in lines:
        if abs(line["ts"] - p["ts"]) <= window and confirms(line["text"], p["direction"]):
            line["confirm"] = p["direction"]
            if not p.get("quote"):
                p["quote"] = line["text"]


def fmt_time(ts=None):
    return datetime.fromtimestamp(ts or time.time()).strftime("%H:%M:%S")


def new_state(name, url):
    return {"name": name, "url": url, "status": "starting", "detail": "connecting…",
            "video_id": None, "boxes": 0, "position": None, "position_note": "",
            "last_box_time": 0, "transcript": deque(maxlen=400), "updated": time.time()}


# --------------------------------------------------------------------------
# Position box detection (TradingView Long/Short Position tool colours)
# --------------------------------------------------------------------------
def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array(r["low"], dtype=np.uint8)
        hi = np.array(r["high"], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps (text, candles)
    return mask


def find_boxes(mask, fcfg, roi_offset=(0, 0)):
    """Return wide, solid rectangles from a colour mask."""
    h_img, w_img = mask.shape
    min_w = fcfg["min_box_width_frac"] * w_img
    min_h = fcfg["min_box_height_px"]
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue  # candles / volume bars are narrow, so they're dropped here
        fill = cv2.contourArea(c) / float(w * h)
        if fill < fcfg["min_fill_ratio"]:
            continue
        boxes.append((x + roi_offset[0], y + roi_offset[1], w, h))
    return boxes


def detect_positions(frame_bgr, cfg):
    """Find green-box + red-box pairs stacked vertically (= a position tool)."""
    fcfg = cfg["filter"]
    ox, oy = 0, 0
    img = frame_bgr
    roi = fcfg.get("chart_region")  # optional [x, y, w, h] in 1280x720 pixels
    if roi:
        x, y, w, h = roi
        img = frame_bgr[y:y + h, x:x + w]
        ox, oy = x, y

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    greens = find_boxes(color_mask(hsv, fcfg["green_hsv"]), fcfg, (ox, oy))
    reds = find_boxes(color_mask(hsv, fcfg["red_hsv"]), fcfg, (ox, oy))

    pairs = []
    tol = fcfg["pair_gap_px"]
    for g in greens:
        gx, gy, gw, gh = g
        for r in reds:
            rx, ry, rw, rh = r
            overlap = min(gx + gw, rx + rw) - max(gx, rx)
            if overlap < 0.6 * min(gw, rw):
                continue  # not horizontally aligned
            touching = abs((gy + gh) - ry) <= tol or abs((ry + rh) - gy) <= tol
            if not touching:
                continue
            side = "long" if gy < ry else "short"  # green on top = long
            pairs.append({"green": g, "red": r, "side_guess": side,
                          "ratio": round(gh / max(rh, 1), 2)})
    return pairs, greens, reds




# --------------------------------------------------------------------------
# Reading the live stream (frames stay in memory)
# --------------------------------------------------------------------------
def resolve_stream_url(url):
    cmd = [sys.executable, "-m", "yt_dlp", "-g", "--no-warnings",
           # live: use the HLS (m3u8) feed; other live tracks cut out after a few seconds
           # bv* = any format with video (audio optional); widest fallback chain
           "-f", "bv*[height<=720][protocol*=m3u8]/bv*[height<=720]/bv*/b", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(out.stderr.strip()[-300:] or "yt-dlp returned nothing")
    return out.stdout.strip().splitlines()[0]


def frame_source(url, fps, start=None):
    """Yield BGR frames (numpy arrays) from a stream/video. No disk writes."""
    src = url if Path(url).exists() else resolve_stream_url(url)
    cmd = ["ffmpeg", "-loglevel", "error"]
    if not Path(url).exists():
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
    if start:
        cmd += ["-ss", str(start)]
    if CFG.get("capture", {}).get("keyframes_only", True):
        cmd += ["-skip_frame", "nokey"]   # decode keyframes only: far less CPU
    cmd += ["-threads", "2", "-i", src, "-an",
           "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=FRAME_BYTES * 2)
    try:
        while True:
            raw = proc.stdout.read(FRAME_BYTES)
            if len(raw) < FRAME_BYTES:
                break  # stream ended or URL expired -> caller reconnects
            yield np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    finally:
        proc.kill()
        proc.wait()




# --------------------------------------------------------------------------
# YouTube links
# --------------------------------------------------------------------------
YT_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")


def normalize_url(raw):
    """Accepts channel, /live, /streams, watch?v= and youtu.be links.
    Returns (url_to_watch, suggested_name) or raises ValueError."""
    raw = raw.strip()
    if not raw.startswith("http"):
        raw = "https://" + raw
    u = urlparse(raw)
    if u.hostname not in YT_HOSTS:
        raise ValueError("That isn't a YouTube link")
    path = u.path.rstrip("/")
    if u.hostname == "youtu.be" and path:
        vid = path.strip("/")
        return f"https://www.youtube.com/watch?v={vid}", vid
    if path == "/watch" and "v" in parse_qs(u.query):
        vid = parse_qs(u.query)["v"][0]
        return f"https://www.youtube.com/watch?v={vid}", vid
    m = re.match(r"^/live/([\w-]+)$", path)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}", m.group(1)
    m = re.match(r"^/(@[\w.\-]+)", path)
    if m:  # channel: always watch its current live stream
        handle = m.group(1)
        return f"https://www.youtube.com/{handle}/live", handle[1:]
    m = re.match(r"^/(channel/[\w-]+|c/[\w.\-]+|user/[\w.\-]+)", path)
    if m:
        return f"https://www.youtube.com/{m.group(1)}/live", m.group(1).split("/")[-1]
    raise ValueError("Couldn't read that link. Paste a channel or live stream link")


# --------------------------------------------------------------------------
# Position monitor (no AI): box appears -> position opened, box gone -> closed?
# --------------------------------------------------------------------------
class Monitor(threading.Thread):
    def __init__(self, name, url, cfg):
        super().__init__(daemon=True, name=name)
        self.name_, self.url, self.cfg = name, url, cfg
        self.stop_event = threading.Event()
        self.stable = 0          # confirmed number of position boxes on the chart
        self.cand, self.hits = None, 0

    def _st(self):
        return STATE.get(self.name_)  # None once removed

    def set_status(self, status, detail=""):
        with LOCK:
            st = self._st()
            if not st:
                return
            st["status"], st["detail"], st["updated"] = status, detail, time.time()
            if status != "live":
                st["video_id"] = None
        if status == "live":
            threading.Thread(target=self._find_video_id, daemon=True).start()

    def _find_video_id(self):
        try:
            out = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings",
                                  "--print", "id", "--skip-download", self.url],
                                 capture_output=True, text=True, timeout=60)
            vid = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
        except Exception:
            vid = None
        with LOCK:
            st = self._st()
            if st:
                st["video_id"] = vid

    def process(self, frame, now):
        pairs, _, _ = detect_positions(frame, self.cfg)
        count = len(pairs)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        small = cv2.imencode(".jpg", cv2.resize(frame, (640, 360)),
                             [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        d = self.cfg.get("dashboard", {})
        confirm = self.cfg["filter"].get("confirm_frames", 2)
        clear_after = d.get("position_clear_seconds", 30)

        # count must hold for a few frames before it counts (ignores flicker)
        if count == self.cand:
            self.hits += 1
        else:
            self.cand, self.hits = count, 1

        with LOCK:
            st = self._st()
            if not st:
                return
            FRAMES[self.name_] = small
            st["boxes"] = count
            st["updated"] = time.time()
            if count:
                st["last_box_time"] = now
            if self.hits < confirm:
                return
            if count > self.stable:           # a new position box appeared
                sides = {p["side_guess"] for p in pairs}
                side = sides.pop() if len(sides) == 1 else "mixed"
                st["position"] = {"direction": side, "time": fmt_time(now), "ts": now}
                st["position_note"] = ""
                mark_confirmations(st, st["transcript"])   # what was said just before
                if ok:
                    SNAPS[self.name_] = jpg.tobytes()
                ALERTS.appendleft({"time": fmt_time(now), "stream": self.name_,
                                   "text": f"{self.name_}: {side.upper()} position opened"})
                log.info("[%s] %s position opened", self.name_, side.upper())
            self.stable = count
            if st["position"] and not count and now - st["last_box_time"] > clear_after \
                    and not st["position_note"]:
                st["position_note"] = f"Position box removed at {fmt_time(now)} (closed?)"
                log.info("[%s] position box removed", self.name_)

    def run(self):
        fps = self.cfg["capture"]["frames_per_second"]
        stop = self.stop_event
        while not stop.is_set():
            try:
                log.info("[%s] connecting...", self.name_)
                n = 0
                for frame in frame_source(self.url, fps):
                    if stop.is_set():
                        break
                    if n == 0:
                        log.info("[%s] live, watching", self.name_)
                        self.set_status("live")
                    n += 1
                    self.process(frame, time.time())
                if stop.is_set():
                    break
                if n:
                    log.warning("[%s] stream dropped after %d s, reconnecting",
                                self.name_, n // fps)
                    stop.wait(3)
                    continue
                self.set_status("offline", "stream ended")
            except Exception as e:
                msg = str(e).replace("ERROR: ", "")[:160]
                log.warning("[%s] not available: %s", self.name_, msg)
                self.set_status("offline", msg.split(": ")[-1])
            stop.wait(self.cfg["capture"]["retry_seconds"])
        log.info("[%s] stopped", self.name_)


# --------------------------------------------------------------------------
# Live transcript (local Whisper; audio stays in memory)
# --------------------------------------------------------------------------
class Transcriber:
    def __init__(self, cfg):
        d = cfg.get("dashboard", {})
        self.model_name = d.get("whisper_model", "base.en")
        self.language = d.get("language", "en")
        self.chunk = d.get("chunk_seconds", 6)
        self.keywords = [k.lower() for k in d.get("trade_keywords", [])]
        self.model = None
        self.lock = threading.Lock()  # one transcription at a time (keeps CPU sane)

    def load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.error("Transcripts off: run  py -m pip install faster-whisper")
            return False
        log.info("Loading Whisper model '%s' (first run downloads it)...", self.model_name)
        self.model = WhisperModel(self.model_name, device="auto", compute_type="int8")
        log.info("Whisper ready")
        return True

    def start(self, name, url):
        threading.Thread(target=self.run_stream, args=(name, url), daemon=True).start()

    def run_stream(self, name, url):
        sr = 16000
        chunk_bytes = sr * 2 * self.chunk
        while True:
            with LOCK:
                st = STATE.get(name)
                if st is None or st["url"] != url:
                    return  # stream removed
                live = st["status"] == "live"
            if not live:
                time.sleep(5)
                continue
            proc = None
            try:
                out = subprocess.run([sys.executable, "-m", "yt_dlp", "-g", "--no-warnings",
                                      "-f", "ba/b[height<=360]/b", url],
                                     capture_output=True, text=True, timeout=60)
                src = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
                if not src:
                    raise RuntimeError("no audio")
                proc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-reconnect", "1",
                     "-reconnect_streamed", "1", "-i", src, "-vn",
                     "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                while True:
                    raw = proc.stdout.read(chunk_bytes)
                    if len(raw) < chunk_bytes:
                        break
                    with LOCK:
                        if name not in STATE:
                            return
                    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                    with self.lock:
                        segs, _ = self.model.transcribe(
                            audio, language=self.language, beam_size=1,
                            vad_filter=True, condition_on_previous_text=False)
                        text = " ".join(s.text.strip() for s in segs).strip()
                    if text:
                        self.add(name, text)
            except Exception as e:
                log.debug("[%s] transcript: %s", name, e)
            finally:
                if proc:
                    proc.kill()
                    proc.wait()
            time.sleep(3)

    def add(self, name, text):
        low = text.lower()
        hit = any(k in low for k in self.keywords)
        with LOCK:
            st = STATE.get(name)
            if st:
                line = {"t": fmt_time(), "ts": time.time(), "text": text, "hit": hit}
                st["transcript"].append(line)
                mark_confirmations(st, [line])                # what was said just after


def cpu_percent():
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Add / remove streams (session only)
# --------------------------------------------------------------------------
def add_stream(name, url, captions=True):
    with LOCK:
        base, i = name, 2
        while name in STATE:
            name = f"{base} ({i})"
            i += 1
        if any(st["url"] == url for st in STATE.values()):
            raise ValueError("That stream is already on the dashboard")
        STATE[name] = new_state(name, url)
    m = Monitor(name, url, CFG)
    MONITORS[name] = m
    m.start()
    if TRANSCRIBER and captions:
        TRANSCRIBER.start(name, url)
    log.info("[%s] added (%s)", name, url)
    return name


def remove_stream(name):
    with LOCK:
        if name not in STATE:
            return False
        del STATE[name]
        FRAMES.pop(name, None)
        SNAPS.pop(name, None)
    m = MONITORS.pop(name, None)
    if m:
        m.stop_event.set()
    log.info("[%s] removed", name)
    return True


# --------------------------------------------------------------------------
# Web page
# --------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Stream Dashboard</title>
<style>
:root{--bg:#0e1117;--card:#161b24;--line:#262d3a;--text:#e6e9ef;--dim:#8b93a3;--input:#0b0f15;
--chip:#1f2633;--badge:#2a3140;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--blue:#3b82f6;--col:360px}
body.light{--bg:#ffffff;--card:#ffffff;--line:#d0d5dd;--text:#000000;--dim:#555b66;--input:#ffffff;
--chip:#f2f4f7;--badge:#eaecf0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.4 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;flex-direction:column;gap:8px}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
h1{font-size:16px;margin:0 8px 0 0}
input[type=text]{background:var(--input);border:1px solid var(--line);color:var(--text);border-radius:6px;
padding:7px 10px;font:inherit}
#url{flex:1;min-width:260px}#nm{width:150px}
button{background:var(--blue);border:0;color:#fff;border-radius:6px;padding:7px 12px;font:inherit;cursor:pointer}
button.ghost{background:var(--badge);color:var(--text)}button:disabled{opacity:.5}
label.sz{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--dim)}
#cpu{font-size:12px;color:var(--dim);min-width:70px}#cpu.hot{color:var(--red);font-weight:600}
#msg{font-size:12px;color:var(--dim)}#msg.err{color:var(--red)}
#alerts{display:flex;gap:8px;overflow-x:auto;white-space:nowrap;min-height:24px}
.al{background:var(--chip);border:1px solid var(--line);border-radius:6px;padding:3px 8px;font-size:12px}
.al b{color:var(--amber)}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--col),1fr));gap:12px;padding:12px;
grid-auto-flow:row dense}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;
display:grid;grid-template-columns:minmax(0,1fr) clamp(170px,32%,300px);transition:border-color .3s,box-shadow .3s}
.tile.dragging{opacity:.4}.tile.dropover{outline:3px dashed var(--blue);outline-offset:-3px}
.tile.checking{border-color:var(--amber);animation:pulse 1.2s ease-in-out infinite}
.tile.open-long{border:3px solid var(--green);box-shadow:0 0 18px rgba(34,197,94,.45)}
.tile.open-short{border:3px solid var(--red);box-shadow:0 0 18px rgba(239,68,68,.45)}
.tile.open-long .top{background:rgba(34,197,94,.16)}.tile.open-short .top{background:rgba(239,68,68,.16)}
.tile.open-long .pos{background:rgba(34,197,94,.08)}.tile.open-short .pos{background:rgba(239,68,68,.08)}
.tile.fresh{animation:flash .8s ease-in-out 12}
@keyframes pulse{50%{box-shadow:0 0 0 3px rgba(245,158,11,.45)}}
@keyframes flash{50%{box-shadow:0 0 34px 6px rgba(120,120,120,.45)}}
.main{display:flex;flex-direction:column;min-width:0}
.top{display:flex;align-items:center;gap:6px;padding:7px 8px;cursor:grab;user-select:none}
.top:active{cursor:grabbing}
.grip{color:var(--dim);font-size:14px}
.name{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:40px}
.flag{display:none;font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;color:#fff;white-space:nowrap}
.open-long .flag{display:inline;background:var(--green)}.open-short .flag{display:inline;background:var(--red)}
.badge{font-size:11px;padding:2px 8px;border-radius:99px;background:var(--badge);color:var(--dim);
max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge.live{background:#3b1212;color:#ff6b6b}body.light .badge.live{background:#fde2e2;color:#b91c1c}
.sb{font-size:12px;background:none;border:1px solid var(--line);color:var(--dim);border-radius:6px;
padding:1px 7px;cursor:pointer;line-height:18px}
.sb:hover{color:var(--text)}.del:hover{color:var(--red);border-color:var(--red)}
.media{position:relative;aspect-ratio:16/9;background:#000}
.media iframe,.media img{position:absolute;inset:0;width:100%;height:100%;border:0;object-fit:contain}
.media .off{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
color:#9aa3b2;font-size:13px;padding:16px;text-align:center}
.vol{display:flex;align-items:center;gap:6px;padding:5px 8px;border-top:1px solid var(--line);font-size:12px;color:var(--dim)}
.vol input{flex:1;max-width:160px}
.pos{padding:8px 10px;border-top:1px solid var(--line);font-size:13px;min-height:40px}
.pos .long{color:var(--green);font-weight:700}.pos .short{color:var(--red);font-weight:700}
.pos .quote{display:block;margin-top:3px;font-style:italic}
.dim{color:var(--dim)}.pos a{color:var(--blue);text-decoration:none;font-size:12px}
.side{border-left:1px solid var(--line);position:relative;min-height:0}
.side h3{margin:0;padding:9px 10px;font-size:12px;font-weight:600;color:var(--dim);
border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.04em}
.tx{overflow-y:auto;padding:6px 10px;font-size:13px;position:absolute;top:36px;bottom:0;left:0;right:0}
.tx div{margin:3px 0}.tx .t{color:var(--dim);font-size:11px;display:block}
.tx .hit{background:rgba(245,158,11,.15);border-radius:4px;padding:2px 4px}
.tx .confirm{border-radius:4px;padding:3px 5px;font-weight:600}
.tx .confirm.long{background:rgba(34,197,94,.22);border-left:3px solid var(--green)}
.tx .confirm.short{background:rgba(239,68,68,.22);border-left:3px solid var(--red)}
.tx .ck{font-size:10px;font-weight:700;margin-left:4px}
.tx .confirm.long .ck{color:var(--green)}.tx .confirm.short .ck{color:var(--red)}
.jump{position:absolute;bottom:8px;right:10px;font-size:11px;padding:3px 8px;display:none}
.empty{padding:40px;text-align:center;color:var(--dim);grid-column:1/-1}
</style></head><body>
<header>
 <div class="row"><h1>Stream Dashboard</h1>
  <input type="text" id="url" placeholder="Paste a YouTube channel or live stream link, e.g. youtube.com/@TanjaTrades">
  <input type="text" id="nm" placeholder="Name (optional)">
  <button id="add">Add stream</button><span id="msg"></span></div>
 <div class="row">
  <label class="sz">Card size <input type="range" id="col" min="280" max="760" step="20"></label>
  <button class="ghost" id="theme">☀ Light</button>
  <button class="ghost" id="snd" title="Beep when a position opens">🔔 Sound on</button>
  <span id="cpu"></span>
  <div id="alerts" style="flex:1"></div></div>
</header>
<div id="grid"></div>
<script>
const tiles={}, H={'Content-Type':'application/json','X-Dash':'1'};
// ---------- saved preferences (this browser only) ----------
let P={theme:'dark',order:[],span:{},vol:{},col:360,mode:{},sound:true};
try{Object.assign(P,JSON.parse(localStorage.getItem('dash-prefs')||'{}'))}catch(e){}
function save(){try{localStorage.setItem('dash-prefs',JSON.stringify(P))}catch(e){}}
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function msg(t,err){const m=document.getElementById('msg');m.textContent=t;m.className=err?'err':''}
// ---------- theme ----------
const themeBtn=document.getElementById('theme');
function applyTheme(){document.body.classList.toggle('light',P.theme==='light');
  themeBtn.textContent=P.theme==='light'?'☾ Dark':'☀ Light'}
themeBtn.onclick=()=>{P.theme=P.theme==='light'?'dark':'light';save();applyTheme()};applyTheme();
// ---------- card size ----------
const colIn=document.getElementById('col');colIn.value=P.col;
function applyCol(){document.documentElement.style.setProperty('--col',P.col+'px');fitSpans()}
colIn.oninput=()=>{P.col=+colIn.value;save();applyCol()};
function gridCols(){const g=document.getElementById('grid'),w=g.clientWidth-24;
  return Math.max(1,Math.floor((w+12)/(P.col+12)))}
function fitSpans(){const c=gridCols();Object.keys(tiles).forEach(n=>{
  tiles[n].el.style.gridColumn='span '+Math.min(P.span[n]||2,c)})}
window.addEventListener('resize',fitSpans);
function resize(name,d){P.span[name]=Math.max(1,Math.min(4,(P.span[name]||2)+d));save();fitSpans()}
// ---------- drag to reorder (CSS order: videos keep playing) ----------
function applyOrder(){
  Object.keys(tiles).forEach(n=>{if(!P.order.includes(n))P.order.push(n)});
  P.order.forEach((n,i)=>{if(tiles[n])tiles[n].el.style.order=i});
}
let dragName=null;
function setupDrag(el,name){
  const top=el.querySelector('.top');top.draggable=true;
  top.ondragstart=e=>{dragName=name;el.classList.add('dragging');e.dataTransfer.effectAllowed='move';
    e.dataTransfer.setData('text/plain',name)};
  top.ondragend=()=>{el.classList.remove('dragging');document.querySelectorAll('.dropover').forEach(x=>x.classList.remove('dropover'))};
  el.ondragover=e=>{if(dragName&&dragName!==name){e.preventDefault();el.classList.add('dropover')}};
  el.ondragleave=()=>el.classList.remove('dropover');
  el.ondrop=e=>{e.preventDefault();el.classList.remove('dropover');
    if(!dragName||dragName===name)return;
    const o=P.order.filter(n=>n!==dragName),i=o.indexOf(name),
          from=P.order.indexOf(dragName),to=P.order.indexOf(name);
    o.splice(from<to?i+1:i,0,dragName);P.order=o;save();applyOrder();dragName=null};
}
// ---------- volume (YouTube player commands) ----------
function yt(name,func,args){const f=tiles[name]&&tiles[name].el.querySelector('iframe');
  if(f)f.contentWindow.postMessage(JSON.stringify({event:'command',func,args:args||[]}),'*')}
function applyVol(name){const v=P.vol[name]||{muted:true,level:60};
  if(v.muted)yt(name,'mute');else{yt(name,'unMute');yt(name,'setVolume',[v.level])}}
function volUI(name){const t=tiles[name],v=P.vol[name]||{muted:true,level:60};
  t.el.querySelector('.mute').textContent=v.muted?'🔇 Unmute':'🔊 Mute';
  t.el.querySelector('.lvl').value=v.level}
// ---------- sound on new positions ----------
let actx=null;const sb=document.getElementById('snd');
function sndUI(){sb.textContent=P.sound?'🔔 Sound on':'🔕 Sound off'}sndUI();
sb.onclick=()=>{P.sound=!P.sound;save();sndUI();if(P.sound)beep('long')};
function beep(dir){if(!P.sound)return;
  try{actx=actx||new (window.AudioContext||window.webkitAudioContext)();
    [0,0.25].forEach((d,i)=>{const o=actx.createOscillator(),g=actx.createGain();
      o.frequency.value=dir==='short'?(i?440:660):(i?880:660);
      g.gain.setValueAtTime(0.15,actx.currentTime+d);g.gain.exponentialRampToValueAtTime(0.001,actx.currentTime+d+0.2);
      o.connect(g).connect(actx.destination);o.start(actx.currentTime+d);o.stop(actx.currentTime+d+0.22)})}catch(e){}}
document.addEventListener('click',()=>{try{actx=actx||new AudioContext();actx.resume()}catch(e){}},{once:true});
// ---------- add / remove ----------
async function addStream(){
  const url=document.getElementById('url').value.trim(), name=document.getElementById('nm').value.trim();
  if(!url) return msg('Paste a YouTube link first',true);
  const b=document.getElementById('add'); b.disabled=true; msg('Adding…');
  try{const r=await fetch('/add',{method:'POST',headers:H,body:JSON.stringify({url,name})});
    const d=await r.json(); if(!r.ok) throw new Error(d.error||'failed');
    msg('Added '+d.name+'. Connecting…');document.getElementById('url').value='';document.getElementById('nm').value='';
  }catch(e){msg(e.message,true)} b.disabled=false;
}
document.getElementById('add').onclick=addStream;
document.getElementById('url').onkeydown=e=>{if(e.key==='Enter')addStream()};
async function removeStream(name){
  if(!confirm('Remove '+name+' from this session?')) return;
  await fetch('/remove',{method:'POST',headers:H,body:JSON.stringify({name})});
  drop(name);
}
function drop(name){if(tiles[name]){tiles[name].el.remove();delete tiles[name]}}
// ---------- cards ----------
function mk(s){
  const name=s.name, el=document.createElement('div');el.className='tile';
  el.innerHTML=`<div class="main"><div class="top" title="Drag to move">
     <span class="grip">⠿</span><span class="name" title="${esc(s.url)}">${esc(name)}</span><span class="flag"></span>
     <button class="sb" data-a="smaller" title="Smaller">−</button><button class="sb" data-a="bigger" title="Bigger">+</button>
     <button class="sb" data-a="tog" title="YouTube player / snapshot">⇄</button>
     <span class="badge"></span><button class="sb del" data-a="del" title="Remove from this session">✕</button></div>
     <div class="media"></div>
     <div class="vol"><button class="sb mute"></button><input type="range" class="lvl" min="0" max="100" title="Volume"></div>
     <div class="pos"></div></div>
   <div class="side"><h3>Transcript</h3><div class="tx"></div><button class="jump">↓ latest</button></div>`;
  el.querySelector('[data-a=smaller]').onclick=()=>resize(name,-1);
  el.querySelector('[data-a=bigger]').onclick=()=>resize(name,1);
  el.querySelector('[data-a=tog]').onclick=()=>{P.mode[name]=P.mode[name]==='snap'?'video':'snap';save();
    el.dataset.vid='';render(name,tiles[name].last)};
  el.querySelector('[data-a=del]').onclick=()=>removeStream(name);
  el.querySelectorAll('.top button').forEach(b=>b.ondragstart=e=>e.preventDefault());
  el.querySelector('.mute').onclick=()=>{const v=P.vol[name]||{muted:true,level:60};v.muted=!v.muted;
    P.vol[name]=v;save();volUI(name);applyVol(name)};
  el.querySelector('.lvl').oninput=e=>{const v=P.vol[name]||{muted:true,level:60};v.level=+e.target.value;
    if(v.level>0)v.muted=false;P.vol[name]=v;save();volUI(name);applyVol(name)};
  const tx=el.querySelector('.tx'), jump=el.querySelector('.jump');
  tx.onscroll=()=>{jump.style.display=tx.scrollTop+tx.clientHeight>=tx.scrollHeight-10?'none':'block'};
  jump.onclick=()=>{tx.scrollTop=tx.scrollHeight};
  document.getElementById('grid').appendChild(el);
  tiles[name]={el,last:s,txKey:''};
  setupDrag(el,name);volUI(name);applyOrder();fitSpans();
}
function render(name,s){
  const t=tiles[name], el=t.el; t.last=s;
  const b=el.querySelector('.badge');
  b.textContent=s.status==='live'?'LIVE':(s.detail||s.status);b.title=s.detail||'';
  b.className='badge'+(s.status==='live'?' live':'');
  const media=el.querySelector('.media'), m=P.mode[name]||'video';
  const key=s.status==='live'?(m==='video'&&s.video_id?'v:'+s.video_id:'snap'):'off:'+s.detail;
  if(el.dataset.vid!==key){
    el.dataset.vid=key;
    if(key.startsWith('v:')){
      media.innerHTML=`<iframe allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen
        src="https://www.youtube.com/embed/${s.video_id}?autoplay=1&mute=1&enablejsapi=1&origin=${encodeURIComponent(location.origin)}"></iframe>`;
      const f=media.querySelector('iframe');
      f.onload=()=>{f.contentWindow.postMessage(JSON.stringify({event:'listening'}),'*');
        setTimeout(()=>applyVol(name),1500)};
    }
    else if(key==='snap') media.innerHTML=`<img alt="">`;
    else media.innerHTML=`<div class="off">${esc(s.detail||'offline')}</div>`;
    el.querySelector('.vol').style.display=key.startsWith('v:')?'flex':'none';
  }
  const img=media.querySelector('img'); if(img) img.src=`/frame/${encodeURIComponent(name)}?t=${Date.now()}`;
  // position highlight
  const p=s.position, pos=el.querySelector('.pos');
  const isOpen=!!p&&!s.position_note, dir=p?(p.direction||'').toLowerCase():'';
  el.classList.toggle('open-long',isOpen&&dir==='long');
  el.classList.toggle('open-short',isOpen&&dir!=='long');
  el.classList.toggle('checking',!isOpen&&s.boxes>0);
  el.querySelector('.flag').textContent=isOpen?`POSITION OPEN · ${dir.toUpperCase()||'?'}`:'';
  const sig=p?p.time+'|'+p.direction:'';
  if(sig&&t.sig!==undefined&&sig!==t.sig){
    el.classList.remove('fresh');void el.offsetWidth;el.classList.add('fresh');
    setTimeout(()=>el.classList.remove('fresh'),10000);beep(dir);
  }
  t.sig=sig;
  if(p){
    pos.innerHTML=`<span class="${dir}">${esc(dir.toUpperCase()||'?')}</span> position opened at ${p.time}
      &nbsp;<a href="/snap/${encodeURIComponent(name)}?t=${encodeURIComponent(p.time)}" target="_blank">📷 chart at entry</a>`
      +(p.quote?`<span class="quote">✓ Said: “${esc(p.quote)}”</span>`:'');
    if(s.position_note) pos.innerHTML+=`<br><span class="dim">${esc(s.position_note)}</span>`;
  } else pos.innerHTML=`<span class="dim">No position detected${s.boxes?' · box on chart, confirming…':''}</span>`;
  // transcript: redraw only on change, keep the reader's scroll position
  const tx=el.querySelector('.tx'), k=s.transcript.length+'|'+s.transcript.filter(l=>l.confirm).length;
  if(k!==t.txKey){
    const atBottom=tx.scrollTop+tx.clientHeight>=tx.scrollHeight-10;
    tx.innerHTML=s.transcript.map(l=>{
      const c=l.confirm?`confirm ${l.confirm}`:(l.hit?'hit':'');
      return `<div class="${c}"><span class="t">${l.t}${l.confirm?`<span class="ck">✓ confirms ${l.confirm.toUpperCase()}</span>`:''}</span>${esc(l.text)}</div>`
    }).join('')||`<div class="dim">${s.status==='live'?'Listening…':'Transcript appears when the stream is live'}</div>`;
    t.txKey=k;
    if(atBottom) tx.scrollTop=tx.scrollHeight;
  }
}
async function poll(){
  try{
    const r=await fetch('/state');const d=await r.json();
    const names=new Set(d.streams.map(s=>s.name));
    Object.keys(tiles).forEach(n=>{if(!names.has(n))drop(n)});
    d.streams.forEach(s=>{if(!tiles[s.name])mk(s);render(s.name,s)});
    const g=document.getElementById('grid');let e=g.querySelector('.empty');
    if(!d.streams.length){if(!e){e=document.createElement('div');e.className='empty';
      e.textContent='No streams. Paste a YouTube link above to add one.';g.appendChild(e)}}
    else if(e) e.remove();
    const c=document.getElementById('cpu');
    if(d.cpu!=null){c.textContent='CPU '+Math.round(d.cpu)+'%';c.className=d.cpu>85?'hot':''}
    document.getElementById('alerts').innerHTML=d.alerts.slice(0,8)
      .map(a=>`<span class="al"><b>${a.time}</b> ${esc(a.text)}</span>`).join('');
  }catch(e){}
  setTimeout(poll,1000);
}
applyCol();poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        """HTTP Basic auth, only when DASHBOARD_PASSWORD is set (always on Replit)."""
        if not PASSWORD:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(h[6:]).decode("utf-8").partition(":")
                if hmac.compare_digest(pw.encode(), PASSWORD.encode()):
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Stream Dashboard"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Password required")
        return False

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/state":
            with LOCK:
                data = {"streams": [{**{k: v for k, v in st.items() if k != "transcript"},
                                     "transcript": list(st["transcript"])}
                                    for st in STATE.values()],
                        "alerts": list(ALERTS)}
            data["cpu"] = cpu_percent()
            return self.send(200, data)
        if self.path.startswith("/snap/"):
            name = unquote(self.path[6:].split("?")[0])
            with LOCK:
                jpg = SNAPS.get(name)
            if jpg:
                return self.send(200, jpg, "image/jpeg")
        if self.path.startswith("/frame/"):
            name = unquote(self.path[7:].split("?")[0])
            with LOCK:
                jpg = FRAMES.get(name)
            if jpg:
                return self.send(200, jpg, "image/jpeg")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorized():
            return
        # custom header blocks other websites from posting here (browser CORS rule)
        if self.headers.get("X-Dash") != "1":
            return self.send(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send(400, {"error": "bad request"})
        if self.path == "/add":
            try:
                url, suggested = normalize_url(str(body.get("url", "")))
                name = (str(body.get("name") or "").strip() or suggested)[:40]
                name = add_stream(name, url)
            except ValueError as e:
                return self.send(400, {"error": str(e)})
            return self.send(200, {"name": name})
        if self.path == "/remove":
            ok = remove_stream(str(body.get("name", "")))
            return self.send(200 if ok else 404, {"ok": ok})
        self.send(404, {"error": "not found"})


# --------------------------------------------------------------------------
def main():
    global CFG, TRANSCRIBER
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    with open(BASE_DIR / "dashboard.yaml", "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    d = CFG.setdefault("dashboard", {})

    if ON_REPLIT:
        # small cloud machine: transcripts only if the TRANSCRIPTION secret is "on"
        d["transcription"] = os.environ.get("TRANSCRIPTION", "off").lower() == "on"
        d["open_browser"] = False
        if not PASSWORD:
            sys.exit("On Replit, add a DASHBOARD_PASSWORD secret first "
                     "(Tools > Secrets). Without it anyone with the link could use the dashboard.")

    if d.get("transcription", True):
        tr = Transcriber(CFG)
        if tr.load():
            TRANSCRIBER = tr

    for s in CFG.get("streams") or []:
        if s.get("enabled", True):
            try:
                add_stream(s["name"], s["url"], s.get("captions", True))
            except ValueError as e:
                log.warning("[%s] skipped: %s", s["name"], e)

    port = int(os.environ.get("PORT", d.get("port", 8800)))
    host = "0.0.0.0" if ON_REPLIT else d.get("host", "127.0.0.1")
    if host != "127.0.0.1" and not PASSWORD:
        sys.exit("Set a DASHBOARD_PASSWORD before opening the dashboard to the network.")
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://localhost:{port}"
    log.info("Dashboard running at %s  (Ctrl+C to stop)%s", url,
             "  (open it in Replit's Preview / Webview)" if ON_REPLIT else "")
    if d.get("open_browser", True):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
