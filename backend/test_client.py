"""Tiny WebSocket client for smoke-testing the inference backend.

Streams landmark frames at /ws and prints every reply. Use it to confirm the
server buffers, predicts, and gates without needing the frontend or a webcam.

    # server in one shell
    source venv/bin/activate && uvicorn backend.main:app --port 8000

    # client in another
    source venv/bin/activate && python backend/test_client.py

By default it sends a synthetic held "sign" — a fixed pose with slight jitter,
which is enough to exercise buffering, prediction and the stability filter, but
says nothing about accuracy. To send something real, replay a collected sample:

    python backend/test_client.py --sample ml/data/hello/<uuid>.npy

Note the frame counts: 30 frames only fill the window and yield the FIRST
prediction; a word needs 3 consecutive agreeing predictions on top of that, so
33+ frames minimum. Default is 60.

Equivalent one-liner with wscat (npm i -g wscat) — sends a single frame of
zeros and shows the buffering reply:

    wscat -c ws://localhost:8000/ws \
      -x "{\"landmarks\": [$(python -c 'print(",".join(["0.0"]*126))')]}"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

FEATURE_DIM = 126


def synthetic_sign(rng: np.random.Generator) -> np.ndarray:
    """One plausible raw frame: a right hand held mid-frame, wrist + 20 points."""
    pts = np.zeros((21, 3), np.float32)
    pts[0] = (0.50, 0.60, 0.0)      # wrist
    pts[9] = (0.50, 0.42, 0.0)      # middle-finger MCP -> the scale reference
    for base, angle in zip([1, 5, 9, 13, 17], np.linspace(-0.7, 0.7, 5)):
        for k in range(4):
            i = base + k
            if i <= 20:
                pts[i] = pts[0] + np.array(
                    [np.sin(angle) * (0.05 + 0.03 * k),
                     -np.cos(angle) * (0.07 + 0.04 * k), 0.0], np.float32
                )
    pts += rng.normal(0, 0.004, pts.shape)  # small jitter, like a real hold
    frame = np.zeros(FEATURE_DIM, np.float32)
    frame[63:] = pts.reshape(63)            # Right block; Left stays zeroed
    return frame


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Vox backend.")
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=15.0, help="send rate")
    parser.add_argument("--sample", type=Path, help="replay a collected .npy sample")
    args = parser.parse_args()

    if args.sample:
        seq = np.load(args.sample)
        print(f"replaying {args.sample} {seq.shape} (looped)")
        frames = [seq[i % len(seq)] for i in range(args.frames)]
    else:
        rng = np.random.default_rng(0)
        frames = [synthetic_sign(rng) for _ in range(args.frames)]

    delay = 1.0 / args.fps if args.fps > 0 else 0.0
    words = 0

    print(f"connecting to {args.url}")
    sent = 0
    async with connect(args.url) as ws:
        # With no model loaded the server pushes an error and closes straight
        # away. Surface that rather than dying on the first send.
        try:
            greeting = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.3))
            print(f"\nserver refused the connection: {greeting.get('error', greeting)}")
            health = args.url.replace("ws://", "http://").replace("wss://", "https://")
            print(f"check {health.rsplit('/', 1)[0]}/health for what's missing")
            return
        except asyncio.TimeoutError:
            pass  # normal: the server stays quiet until it gets a frame
        except ConnectionClosed:
            print("\nserver closed the connection immediately — see its log")
            return

        try:
            for i, frame in enumerate(frames, start=1):
                await ws.send(json.dumps({"landmarks": [float(v) for v in frame]}))
                reply = json.loads(await ws.recv())
                sent = i

                if "word" in reply:
                    words += 1
                    print(f"  frame {i:>3}  ** WORD: {reply['word']} "
                          f"(confidence {reply['confidence']:.3f}) **")
                elif "error" in reply:
                    print(f"  frame {i:>3}  ERROR: {reply['error']}")
                    break
                elif "top" in reply:
                    print(f"  frame {i:>3}  listening  top={reply['top']:<12} "
                          f"conf={reply['confidence']:.3f}  "
                          f"stable_for={reply['stable_for']}")
                else:
                    print(f"  frame {i:>3}  buffering  "
                          f"{reply['buffered']}/{reply['needed']}")

                if delay:
                    await asyncio.sleep(delay)
        except ConnectionClosed as exc:
            print(f"\nserver closed the connection after {sent} frame(s): {exc}")
            return

    print(f"\nsent {sent} frames, {words} word(s) emitted")


if __name__ == "__main__":
    asyncio.run(main())
