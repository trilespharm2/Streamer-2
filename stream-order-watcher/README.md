# Stream Order Watcher

Watches YouTube live trading streams and pings your phone when the streamer draws a
trade with TradingView's Long/Short Position tool, including the entry, take-profit and stop-loss.

## One-time setup

1. **Install Python 3.10+** (python.org; on Windows tick "Add to PATH").
2. **Install ffmpeg**
   - Windows: `winget install Gyan.FFmpeg`
   - Mac: `brew install ffmpeg`
3. **Install the libraries** (in this folder):
   ```
   pip install -r requirements.txt
   ```
4. **API key:** create one at console.anthropic.com, set a monthly spend limit, then
   rename `.env.example` to `.env` and paste the key in.
5. **Telegram alerts:**
   - In Telegram, message **@BotFather** → send `/newbot` → pick a name → copy the token.
   - Put it in `.env` as `TELEGRAM_BOT_TOKEN=...`
   - Open your new bot, press Start, send it "hi".
   - Run `py watcher.py --telegram-setup`. It prints your chat ID (add it to `.env` as
     `TELEGRAM_CHAT_ID=...`) and sends a test message.
   - Alerts look like: `Candytrades short NQ at price: 21450 (tp: 21400, sl: 21475)` plus the chart frame.
   (Optional) **ntfy alerts:** install the **ntfy** app, subscribe to a topic with a long random
   name (anyone who knows the name can read it), and put that name in `config.yaml`
   under `ntfy_topic`.
6. **Streams:** edit the `streams:` list in `config.yaml`.

## Test before going live

Take a screenshot of a stream showing a position box, then run:
```
python watcher.py --calibrate shot.png    # free: checks the box detector, saves shot_debug.jpg
python watcher.py --test-image shot.png   # also asks Claude to read entry/TP/SL (~$0.0002)
```
If `--calibrate` finds 0 position pairs, the streamer uses custom colours. Send me the
screenshot and I'll adjust the colour ranges.

## Test on a published (non-live) video

Pick a past stream or replay where the streamer took trades:
```
python watcher.py --replay "https://www.youtube.com/watch?v=VIDEO_ID" --no-claude
```
This free run lists the video timestamps where a position box appeared and saves those frames
in `replay_detections/`. Jump to those times in the video to check them.

When the boxes look right, run it with Claude to get real alerts on your phone:
```
python watcher.py --replay "https://www.youtube.com/watch?v=VIDEO_ID"
```
- Replay runs much faster than real time, so a 3-hour video takes a few minutes.
- `--start 1:05:30` skips ahead in long videos.
- A video file on your computer also works in place of the URL.

## Run
```
python watcher.py
```
Leave the window open. Streams that aren't live are retried every 60 seconds.

## Output
- Phone notification with direction, entry, TP, SL and the chart frame attached
- `alerts/` holds only the frame that triggered each alert
- `alerts.log` keeps one line per alert
- No other frames are written to disk.

## Cost controls (config.yaml → claude)
- `max_calls_per_hour`: hard cap per stream (default 60, which is about $0.01/hour at most)
- `cooldown_seconds`: minimum gap between calls


The multi-stream viewer is a separate app: see `../stream-dashboard`.
