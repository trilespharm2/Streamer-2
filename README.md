# Streamer tools

Two separate apps for following YouTube live trading streams:

| Folder | What it does |
|---|---|
| [`stream-order-watcher/`](stream-order-watcher) | Watches streams in the background and sends a Telegram alert (direction, entry, TP, SL read by Claude) when a streamer opens a position. Needs an Anthropic API key. |
| [`stream-dashboard/`](stream-dashboard) | One browser page with all streams in a grid, live transcripts, and free position highlighting (no AI, no API key). |

Both need Python 3.10+ and ffmpeg. See each folder's README for setup.
Secrets go in `stream-order-watcher/.env` (see `.env.example`), which is never committed.

**Replit:** the Run button starts the dashboard (see `stream-dashboard/README.md`, "Running on Replit").
Run the watcher from the Shell: `cd stream-order-watcher && pip install -r requirements.txt && python watcher.py`.
