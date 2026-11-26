#!/usr/bin/env python3
"""
Stream Order Watcher
--------------------
Watches several YouTube live streams of trading charts and sends a phone alert
(via ntfy.sh) when the streamer places a trade drawn with TradingView's
Long/Short Position tool (stacked green take-profit / red stop-loss boxes).

How it works (per stream, each in its own thread):
  1. yt-dlp resolves the live stream URL; ffmpeg decodes it at 720p and
     pipes raw frames straight into memory (nothing is written to disk).
  2. A free local OpenCV filter looks for the green/red position boxes.
  3. Only when a NEW position box appears (or an existing one changes shape a
     lot) is the frame sent to Claude, which reads direction / entry / TP / SL.
  4. If Claude confirms a trade, an alert goes to your phone, the triggering
     frame is saved to ./alerts/, and a line is added to alerts.log.

Usage:
  python watcher.py                       # run with config.yaml
  python watcher.py --test-image shot.png # run filter + Claude on one screenshot
  python watcher.py --calibrate shot.png  # filter only, saves a debug image
  python watcher.py --replay URL --no-claude   # free test on a published video
  python watcher.py --replay URL               # same, with Claude + phone alerts
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

BASE_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 3

log = logging.getLogger("watcher")
alert_lock = threading.Lock()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Load API key from .env-style file or environment
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return cfg


# --------------------------------------------------------------------------
# Local filter: detect TradingView Long/Short position boxes
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
# Claude
# --------------------------------------------------------------------------
PROMPT = """This is a frame from a live trading stream showing a price chart.
Look for a trade the streamer has placed, usually drawn with TradingView's
Long/Short Position tool (a green take-profit zone and a red stop-loss zone
meeting at the entry price), or an open position/order shown in a broker panel.

Reply with ONLY a JSON object, no other text:
{
  "order_detected": true/false,
  "status": "filled" | "pending" | "unclear",
  "symbol": ticker only, e.g. "MNQ", "NQ", "ES", or null,
  "direction": "long" | "short" | null,
  "entry": number or null,
  "take_profit": number or null,
  "stop_loss": number or null,
  "confidence": number from 0 to 1,
  "notes": string, under 20 words
}
Read prices from the labels on the zones or the price axis. Use null for
anything you cannot read clearly. Do not guess numbers."""


def ask_claude(frame_bgr, cfg):
    import anthropic  # imported here so --calibrate works without it

    ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    b64 = base64.standard_b64encode(jpg.tobytes()).decode()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    for attempt in (1, 2):  # retry once if the reply comes back blank or cut off
        msg = client.messages.create(
            model=cfg["claude"]["model"],
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        log.warning("Claude reply unreadable (attempt %d, stop: %s): %s", attempt,
                    getattr(msg, "stop_reason", "?"), text[:120].replace("\n", " "))
    return None


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------
def fmt(v):
    return "?" if v is None else f"{v:.10g}" if isinstance(v, (int, float)) else str(v)


def telegram_text(stream_name, r):
    # e.g. "Candytrades short NQ at price: 21450 (tp: 21480, sl: 21435)"
    parts = [stream_name, (r.get("direction") or "?")]
    if r.get("symbol"):
        parts.append(r["symbol"])
    text = (" ".join(parts) + f" at price: {fmt(r.get('entry'))} "
            f"(tp: {fmt(r.get('take_profit'))}, sl: {fmt(r.get('stop_loss'))})")
    status = r.get("status")
    if status and status != "filled":
        text += f" [{status}]"
    return text


def send_telegram(text, img_path, cfg):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    api = f"https://api.telegram.org/bot{token}"
    try:
        if img_path and cfg["alerts"].get("attach_frame_to_alert", True):
            with open(img_path, "rb") as f:
                r = requests.post(f"{api}/sendPhoto", data={"chat_id": chat, "caption": text},
                                  files={"photo": f}, timeout=20)
        else:
            r = requests.post(f"{api}/sendMessage", data={"chat_id": chat, "text": text},
                              timeout=15)
        if not r.ok:
            log.error("Telegram failed: %s", r.text[:200])
            return False
        return True
    except requests.RequestException as e:
        log.error("Telegram failed: %s", e)
        return False


def telegram_setup():
    """Find your chat ID and send a test message."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("Add TELEGRAM_BOT_TOKEN=... to your .env file first")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15).json()
        chats = {u["message"]["chat"]["id"] for u in r.get("result", []) if "message" in u}
        if not chats:
            sys.exit("No messages found. Open your bot in Telegram, send it 'hi', then rerun.")
        chat = str(next(iter(chats)))
        print(f"Your chat ID is: {chat}\nAdd this line to .env:  TELEGRAM_CHAT_ID={chat}")
        os.environ["TELEGRAM_CHAT_ID"] = chat
    ok = send_telegram("Test: Candytrades short NQ at price: 21450 (tp: 21400, sl: 21475)",
                       None, {"alerts": {}})
    print("Test message sent. Check Telegram." if ok else
          "Test message FAILED. See the error above.")


def send_backup_alert(stream_name, pairs, reason, frame_bgr, cfg):
    """Used when Claude can't be reached (e.g. credits ran out)."""
    sides = {p["side_guess"] for p in pairs}
    side = sides.pop() if len(sides) == 1 else "?"
    text = (f"{stream_name}: possible {side} trade detected "
            f"(prices unavailable: {reason}). Check the chart.")
    img_path = None
    with alert_lock:
        if cfg["alerts"].get("save_trigger_frame", True):
            out_dir = BASE_DIR / "alerts"
            out_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe = "".join(c if c.isalnum() else "_" for c in stream_name)
            img_path = out_dir / f"{ts}_{safe}_backup.jpg"
            cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(BASE_DIR / "alerts.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                "stream": stream_name, "backup": True, "side_guess": side,
                                "reason": reason,
                                "frame": str(img_path) if img_path else None}) + "\n")
    log.warning("BACKUP ALERT %s", text)
    if cfg["alerts"].get("telegram", True):
        send_telegram(text, img_path, cfg)


def claude_error_reason(e):
    msg = str(e).lower()
    if "credit" in msg or "billing" in msg:
        return "Claude credits ran out"
    if "api key" in msg or "authentication" in msg or "401" in msg:
        return "Claude API key problem"
    if "overloaded" in msg or "529" in msg or "rate" in msg:
        return "Claude busy"
    return "Claude unreachable"


def send_alert(stream_name, result, frame_bgr, cfg):
    direction = (result.get("direction") or "?").upper()
    symbol = result.get("symbol") or ""
    title = f"{stream_name}: {direction} {symbol}".strip()
    body = (f"Entry {fmt(result.get('entry'))} | TP {fmt(result.get('take_profit'))} | "
            f"SL {fmt(result.get('stop_loss'))}\n"
            f"Status: {result.get('status', '?')} | "
            f"Confidence: {result.get('confidence', '?')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe = "".join(c if c.isalnum() else "_" for c in stream_name)
    img_path = None
    with alert_lock:
        if cfg["alerts"].get("save_trigger_frame", True):
            out_dir = BASE_DIR / "alerts"
            out_dir.mkdir(exist_ok=True)
            img_path = out_dir / f"{ts}_{safe}.jpg"
            cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(BASE_DIR / "alerts.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                "stream": stream_name, **result,
                                "frame": str(img_path) if img_path else None}) + "\n")

    log.info("ALERT %s | %s", title, body.replace("\n", " | "))

    if cfg["alerts"].get("telegram", True):
        send_telegram(telegram_text(stream_name, result), img_path, cfg)

    topic = cfg["alerts"].get("ntfy_topic")
    if not topic:
        return
    url = f"{cfg['alerts'].get('ntfy_server', 'https://ntfy.sh').rstrip('/')}/{topic}"
    headers = {"Title": title.encode("ascii", "ignore").decode(),
               "Priority": "high", "Tags": "chart_with_upwards_trend"}
    try:
        if img_path and cfg["alerts"].get("attach_frame_to_alert", True):
            headers["Message"] = body.replace("\n", " | ")
            headers["Filename"] = img_path.name
            requests.put(url, data=img_path.read_bytes(), headers=headers, timeout=15)
        else:
            requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=15)
    except requests.RequestException as e:
        log.error("ntfy failed: %s", e)


# --------------------------------------------------------------------------
# Stream reader (frames live in memory only)
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
    cmd += ["-threads", "0", "-i", src, "-an",
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
# Per-stream watcher
# --------------------------------------------------------------------------
class StreamWatcher(threading.Thread):
    def __init__(self, stream_cfg, cfg):
        super().__init__(daemon=True, name=stream_cfg["name"])
        self.name_ = stream_cfg["name"]
        self.url = stream_cfg["url"]
        self.cfg = cfg
        self.stable_count = 0       # pairs seen in the last confirmed state
        self.stable_ratios = []
        self.candidate = None
        self.candidate_hits = 0
        self.last_call = 0.0
        self.calls = deque()        # timestamps of Claude calls (for hourly cap)
        self.recent_alerts = []     # (time, direction, entry) for de-duplication
        self.detections = []        # used by replay --no-claude
        self.stop_event = threading.Event()  # set to stop this watcher (dashboard remove)

    # ---- hooks (used by dashboard.py; do nothing here) -------------------
    def on_frame(self, frame, pairs, now):
        pass

    def on_alert(self, result, now):
        pass

    def on_status(self, status, detail=""):
        pass

    # ---- trigger logic -------------------------------------------------
    def should_ask(self, pairs):
        fcfg = self.cfg["filter"]
        count = len(pairs)
        ratios = sorted(p["ratio"] for p in pairs)
        state = (count, tuple(ratios))

        # require the same state for N consecutive frames (avoids flicker)
        if self.candidate and self.candidate[0] == count and \
                self._ratios_close(self.candidate[1], ratios):
            self.candidate_hits += 1
        else:
            self.candidate, self.candidate_hits = state, 1
        if self.candidate_hits < fcfg["confirm_frames"]:
            return False

        trigger = False
        if count > self.stable_count:
            trigger = True                      # a new position box appeared
        elif count and count == self.stable_count and \
                not self._ratios_close(self.stable_ratios, ratios, fcfg["reshape_tolerance"]):
            trigger = True                      # existing box reshaped (TP/SL moved)
        if count != self.stable_count or trigger:
            self.stable_count, self.stable_ratios = count, ratios
        return trigger

    @staticmethod
    def _ratios_close(a, b, tol=0.15):
        if len(a) != len(b):
            return False
        return all(abs(x - y) <= tol * max(x, y, 0.01) for x, y in zip(a, b))

    def budget_ok(self, now):
        c = self.cfg["claude"]
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()
        if now - self.last_call < c["cooldown_seconds"]:
            return False
        if len(self.calls) >= c["max_calls_per_hour"]:
            log.warning("[%s] hourly Claude cap reached; skipping", self.name_)
            return False
        return True

    # ---- main loop ------------------------------------------------------
    def handle(self, frame, now=None, label=None, use_claude=True):
        """now: clock in seconds (video time in replay mode). label: alert name."""
        now = time.time() if now is None else now
        label = label or self.name_
        pairs, _, _ = detect_positions(frame, self.cfg)
        self.on_frame(frame, pairs, now)
        if not self.should_ask(pairs) or not self.budget_ok(now):
            return
        self.last_call = now
        self.calls.append(now)
        if not use_claude:
            self.detections.append((label, frame.copy(), pairs))
            return
        log.info("[%s] position box detected -> asking Claude", label)
        try:
            result = ask_claude(frame, self.cfg)
        except Exception as e:  # network / API errors shouldn't kill the thread
            log.error("[%s] Claude error: %s", self.name_, str(e)[:160])
            self.backup(label, pairs, claude_error_reason(e), frame, now)
            return
        if result is None:
            self.backup(label, pairs, "Claude reply unreadable", frame, now)
            return
        if not result.get("order_detected"):
            log.info("[%s] Claude: no order (%s)", self.name_,
                     (result or {}).get("notes", "no reply"))
            if self.cfg["alerts"].get("save_rejected_frames", True):
                # keep what Claude saw, so you can check it
                out = BASE_DIR / "alerts" / "rejected"
                out.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe = "".join(c if c.isalnum() else "_" for c in label)
                cv2.imwrite(str(out / f"{ts}_{safe}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            return
        if (result.get("confidence") or 0) < self.cfg["claude"]["min_confidence"]:
            log.info("[%s] low confidence %.2f, skipped", self.name_,
                     result.get("confidence") or 0)
            return
        if self.is_duplicate(result, now):
            log.info("[%s] same trade as a recent alert, not re-sent", self.name_)
            return
        self.recent_alerts.append((now, result.get("direction"), result.get("entry"), "alert"))
        send_alert(label, result, frame, self.cfg)
        self.on_alert(result, now)

    def backup(self, label, pairs, reason, frame, now):
        """Backup alert, at most once per direction per backup window."""
        sides = {p["side_guess"] for p in pairs}
        side = sides.pop() if len(sides) == 1 else "?"
        window = self.cfg["claude"].get("backup_window_minutes", 30) * 60
        for t, d, e, kind in self.recent_alerts:
            if kind == "backup" and d == side and now - t < window:
                log.info("[%s] backup alert already sent recently, not re-sent", self.name_)
                return
        self.recent_alerts.append((now, side, None, "backup"))
        send_backup_alert(label, pairs, reason, frame, self.cfg)
        self.on_alert({"backup": True, "direction": side, "notes": reason}, now)

    def is_duplicate(self, r, now):
        """Same direction and entry within a small tolerance, recently = same trade."""
        c = self.cfg["claude"]
        window = c.get("duplicate_window_minutes", 60) * 60
        tol_pct = c.get("duplicate_price_tolerance_pct", 0.1) / 100
        self.recent_alerts = [a for a in self.recent_alerts if now - a[0] < window]
        d, e = r.get("direction"), r.get("entry")
        for _, d0, e0, kind in self.recent_alerts:
            if kind != "alert" or d != d0:
                continue  # a backup alert never blocks the real one with prices
            if e is None or e0 is None:
                return True
            try:
                if abs(float(e) - float(e0)) <= tol_pct * abs(float(e0)):
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def load_recent_alerts(self):
        """Remember alerts from before a restart, so they aren't sent again."""
        path = BASE_DIR / "alerts.log"
        if not path.exists():
            return
        window = self.cfg["claude"].get("duplicate_window_minutes", 60) * 60
        now = time.time()
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-500:]:
                a = json.loads(line)
                if a.get("stream") != self.name_:
                    continue
                t = datetime.fromisoformat(a["time"]).timestamp()
                if now - t >= window:
                    continue
                if a.get("backup"):
                    self.recent_alerts.append((t, a.get("side_guess"), None, "backup"))
                else:
                    self.recent_alerts.append((t, a.get("direction"), a.get("entry"), "alert"))
        except Exception as e:
            log.debug("could not read alerts.log: %s", e)

    def run(self):
        self.load_recent_alerts()
        fps = self.cfg["capture"]["frames_per_second"]
        stop = self.stop_event
        while not stop.is_set():
            try:
                log.info("[%s] connecting...", self.name_)
                n = 0
                for frame in frame_source(self.url, fps):
                    if stop.is_set():
                        break  # closing the generator also stops ffmpeg
                    if n == 0:
                        log.info("[%s] live, watching", self.name_)
                        self.on_status("live")
                    n += 1
                    self.handle(frame)
                if stop.is_set():
                    break
                if n:
                    # was live: reconnect quickly so a short drop doesn't miss a trade
                    log.warning("[%s] stream dropped after %d s, reconnecting",
                                self.name_, n // fps)
                    stop.wait(3)
                    continue
                log.warning("[%s] stream ended", self.name_)
                self.on_status("offline", "stream ended")
            except Exception as e:
                msg = str(e).replace("ERROR: ", "")[:160]
                log.warning("[%s] not available: %s", self.name_, msg)
                self.on_status("offline", msg.split(": ")[-1])
            stop.wait(self.cfg["capture"]["retry_seconds"])
        log.info("[%s] stopped", self.name_)


# --------------------------------------------------------------------------
# Replay a published video (or local file) as if it were live
# --------------------------------------------------------------------------
def parse_time(t):
    if not t:
        return 0
    secs = 0
    for part in str(t).split(":"):
        secs = secs * 60 + float(part)
    return secs


def hms(secs):
    secs = int(secs)
    return f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}"


def replay(url, cfg, use_claude, start):
    """Runs at full decode speed (much faster than real time). Timing rules
    (cooldown, hourly cap) use the video's own clock, so results match live."""
    fps = cfg["capture"]["frames_per_second"]
    offset = parse_time(start)
    w = StreamWatcher({"name": "Replay", "url": url}, cfg)
    t0, n = time.time(), 0
    log.info("Replaying %s %s", url, "(filter only, free)" if not use_claude else "")

    with tempfile.TemporaryDirectory(prefix="sow_") as tmp:  # auto-deleted at the end
        src = url
        if not Path(url).exists():
            # Streaming straight from YouTube is throttled to ~playback speed.
            # A parallel download to a temp file is many times faster.
            log.info("Downloading video to a temporary file (deleted when done)...")
            tmpl = str(Path(tmp) / "video.%(ext)s")
            cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "-q", "--progress",
                   "-N", "8", "--no-part",
                   "-f", "bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720]/best[height<=720]",
                   "-o", tmpl, url]
            if offset:
                cmd[-1:-1] = ["--download-sections", f"*{int(offset)}-inf"]
            if subprocess.run(cmd).returncode != 0:
                sys.exit("Download failed (see message above)")
            files = list(Path(tmp).glob("video.*"))
            if not files:
                sys.exit("Download produced no file")
            src = str(files[0])
            log.info("Downloaded in %.0f s, scanning...", time.time() - t0)
            start_arg = None  # already trimmed by the download
        else:
            start_arg = offset or None

        for n, frame in enumerate(frame_source(src, fps, start=start_arg), 1):
            vt = offset + n / fps
            if n % (300 * fps) == 0:
                log.info("...at %s in the video", hms(vt))
            w.handle(frame, now=vt, label=f"Replay @ {hms(vt)}", use_claude=use_claude)

    if not use_claude:
        out = BASE_DIR / "replay_detections"
        out.mkdir(exist_ok=True)
        for label, frame, pairs in w.detections:
            for p in pairs:
                for (x, y, bw, bh), col in ((p["green"], (0, 255, 0)), (p["red"], (0, 0, 255))):
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
            name = label.replace("Replay @ ", "").replace(":", "-")
            cv2.imwrite(str(out / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  box detected at {label.replace('Replay @ ', '')}")
        print(f"\n{len(w.detections)} detection(s); frames saved in {out}")
        print(f"A full run with Claude would cost about ${len(w.detections) * 0.0002:.4f}")
    log.info("Done: %s of video in %.0f s", hms(n / fps), time.time() - t0)


# --------------------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return cv2.resize(img, (WIDTH, HEIGHT))


def calibrate(path, cfg):
    img = load_image(path)
    pairs, greens, reds = detect_positions(img, cfg)
    dbg = img.copy()
    for x, y, w, h in greens:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in reds:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for p in pairs:
        x, y, w, h = p["green"]
        cv2.putText(dbg, f"POSITION ({p['side_guess']})", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    out = Path(path).with_name(Path(path).stem + "_debug.jpg")
    cv2.imwrite(str(out), dbg)
    print(f"Green boxes: {len(greens)}  Red boxes: {len(reds)}  Position pairs: {len(pairs)}")
    for p in pairs:
        print("  ", p)
    print(f"Debug image saved: {out}")
    return pairs


def test_image(path, cfg):
    pairs = calibrate(path, cfg)
    print("\nAsking Claude...")
    result = ask_claude(load_image(path), cfg)
    print(json.dumps(result, indent=2))
    if result and result.get("order_detected"):
        print("\nThis would trigger an alert" if pairs else
              "\nClaude saw an order, but the local filter did not; tune the filter")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Watch live trading streams for orders")
    ap.add_argument("--config", default=str(BASE_DIR / "config.yaml"))
    ap.add_argument("--calibrate", metavar="IMAGE", help="filter-only test on a screenshot")
    ap.add_argument("--test-image", metavar="IMAGE", help="filter + Claude test on a screenshot")
    ap.add_argument("--replay", metavar="URL_OR_FILE",
                    help="test on a published YouTube video or local video file")
    ap.add_argument("--no-claude", action="store_true",
                    help="with --replay: filter only (free), saves detected frames")
    ap.add_argument("--telegram-setup", action="store_true",
                    help="find your Telegram chat ID and send a test message")
    ap.add_argument("--start", help="with --replay: start time, e.g. 1:05:30")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)  # hide per-request lines
    cfg = load_config(args.config)

    if args.telegram_setup:
        telegram_setup()
        return
    if args.calibrate:
        calibrate(args.calibrate, cfg)
        return
    if args.test_image:
        test_image(args.test_image, cfg)
        return

    if args.replay:
        if not args.no_claude and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Missing ANTHROPIC_API_KEY (or add --no-claude for a free filter-only run)")
        replay(args.replay, cfg, not args.no_claude, args.start)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY. Put it in a .env file next to watcher.py")

    threads = [StreamWatcher(s, cfg) for s in cfg["streams"] if s.get("enabled", True)]
    if not threads:
        sys.exit("No streams enabled in config.yaml")
    for t in threads:
        t.start()
    log.info("Watching %d stream(s). Ctrl+C to stop.", len(threads))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
