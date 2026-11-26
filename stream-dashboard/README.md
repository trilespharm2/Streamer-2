# Stream Dashboard

Watch several YouTube live trading streams on one page, each with a live transcript beside
it and a position monitor. Free: no AI service, no API key, no Telegram. Runs on your PC.

This is separate from Stream Order Watcher (the Telegram alert app). Both can run at once.

## Setup (once)
1. Python and ffmpeg are already installed if you set up Stream Order Watcher.
2. In this folder:
   ```
   py -m pip install -r requirements.txt
   ```

## Run
```
py dashboard.py
```
Your browser opens http://localhost:8800. The first run downloads the speech model (~150 MB).

## Using it
- **Add a stream:** paste a YouTube channel or live link in the top bar and press Add stream.
- **Remove a stream:** the x on its card (for this session only).
- **Streams at start:** edit `dashboard.yaml`.
- **Video / snapshot:** switch to the snapshot view if a channel blocks embedding.
- **Position monitor:** when a TradingView Long/Short Position box appears, the card turns
  green (long) or red (short), flashes and beeps. "📷 chart at entry" shows the chart at that
  moment, so you can read the prices. The card shows "closed?" once the box has been gone 30 s.
- **Transcript confirmation:** when a position opens, transcript lines from 90 s before to
  90 s after that say so ("I'm short", "going long", "I'm in"…) are highlighted green/red with
  "✓ confirms LONG/SHORT", and the quote shows under the position.
- **Arrange:** drag a card by its title bar to move it. − / + on a card makes it narrower or wider;
  the "Card size" slider scales all cards. Your layout is remembered in this browser.
- **Light theme:** ☀ Light / ☾ Dark button.
- **Volume:** 🔇 Unmute and the slider under each video (YouTube player view only).
- **Beeps:** toggle with the bell button. Click anywhere on the page once so the browser allows sound.
- **CPU:** the header shows your current CPU use.

## If your PC struggles
Set `whisper_model: "tiny.en"` in `dashboard.yaml`, or add `captions: false` under a stream.
