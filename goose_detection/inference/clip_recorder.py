"""
clip_recorder.py
Rolling pre-roll + state-machine video recorder for goose detection events.

Behavior:
    - Always buffers the last N seconds of frames in a deque (pre-roll).
    - When a detection occurs, opens an MP4 and writes the buffered pre-roll,
      then continues writing while detections keep coming.
    - After `post_roll_sec` seconds without any detection, closes the MP4 and
      writes a sibling .json with per-clip metadata.
    - Re-detections inside the post-roll window simply extend the same clip.

Usage:
    recorder = ClipRecorder(output_dir="results/clips", fps=15,
                            pre_roll_sec=10.0, post_roll_sec=3.0)
    recorder.on_frame(annotated_frame, has_detection=True, top_conf=0.81, detection_count=1)
    ...
    recorder.close()  # call in shutdown/finally to finalize any in-progress clip
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ClipRecorder:
    IDLE = "idle"
    RECORDING = "recording"

    def __init__(
        self,
        output_dir,
        fps: float = 15.0,
        pre_roll_sec: float = 10.0,
        post_roll_sec: float = 3.0,
        fourcc: str = "mp4v",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre_roll_sec = pre_roll_sec
        self.post_roll_sec = post_roll_sec
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.ring: deque = deque(maxlen=max(1, int(fps * pre_roll_sec)))
        self.frame_size = None
        self.state = self.IDLE
        self.writer = None
        self.last_detection_time: float = 0.0
        self.clip_path = None
        self.meta: dict = {}

    def on_frame(
        self,
        frame: np.ndarray,
        has_detection: bool,
        top_conf: float = 0.0,
        detection_count: int = 0,
        track_ids=None,
    ) -> None:
        if self.frame_size is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)

        now = time.time()

        if has_detection:
            if self.state == self.IDLE:
                self._start_clip(now)
            self._write_frame(frame)
            self._update_metadata(now, top_conf, detection_count, track_ids)
            self.last_detection_time = now
        else:
            if self.state == self.RECORDING:
                self._write_frame(frame)
                if now - self.last_detection_time >= self.post_roll_sec:
                    self._finalize_clip(now)

        self.ring.append(frame.copy())

    def close(self) -> None:
        if self.state == self.RECORDING:
            self._finalize_clip(time.time())

    def _start_clip(self, now: float) -> None:
        first_frame_ts = now - self.pre_roll_sec
        stem = datetime.fromtimestamp(first_frame_ts).strftime("goose_%Y%m%d_%H%M%S")
        self.clip_path = self.output_dir / f"{stem}.mp4"

        self.writer = cv2.VideoWriter(
            str(self.clip_path), self.fourcc, self.fps, self.frame_size
        )
        if not self.writer.isOpened():
            self.writer = None
            self.clip_path = None
            print(f"[CLIP] ERROR: VideoWriter failed to open. Recording disabled for this event.")
            return

        for f in self.ring:
            self.writer.write(f)

        self.meta = {
            "clip_path": str(self.clip_path),
            "start_time_iso": datetime.fromtimestamp(first_frame_ts).isoformat(),
            "pre_roll_frames": len(self.ring),
            "pre_roll_sec": self.pre_roll_sec,
            "post_roll_sec": self.post_roll_sec,
            "fps": self.fps,
            "frame_size": list(self.frame_size),
            "detection_first_seen_iso": datetime.fromtimestamp(now).isoformat(),
            "detection_last_seen_iso": datetime.fromtimestamp(now).isoformat(),
            "detection_frame_count": 0,
            "total_detections": 0,
            "peak_confidence": 0.0,
            "_sum_confidence": 0.0,
            "_unique_track_ids": set(),
        }
        self.state = self.RECORDING

    def _write_frame(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def _update_metadata(self, now: float, top_conf: float, detection_count: int, track_ids=None) -> None:
        if not self.meta:
            return
        self.meta["detection_last_seen_iso"] = datetime.fromtimestamp(now).isoformat()
        self.meta["detection_frame_count"] += 1
        self.meta["total_detections"] += int(detection_count)
        self.meta["peak_confidence"] = max(self.meta["peak_confidence"], float(top_conf))
        self.meta["_sum_confidence"] += float(top_conf)
        if track_ids:
            self.meta["_unique_track_ids"].update(int(i) for i in track_ids if i >= 0)

    def _finalize_clip(self, now: float) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        end_ts = datetime.fromtimestamp(now)
        if self.meta:
            start_ts = datetime.fromisoformat(self.meta["start_time_iso"])
            self.meta["end_time_iso"] = end_ts.isoformat()
            self.meta["duration_sec"] = (end_ts - start_ts).total_seconds()
            det_n = self.meta["detection_frame_count"]
            self.meta["mean_confidence"] = (
                self.meta["_sum_confidence"] / det_n if det_n > 0 else 0.0
            )
            unique_ids = sorted(self.meta.get("_unique_track_ids", set()))
            self.meta["unique_tracks_seen"] = unique_ids
            self.meta["unique_goose_count"] = len(unique_ids)
            self.meta.pop("_sum_confidence", None)
            self.meta.pop("_unique_track_ids", None)

            if self.clip_path is not None:
                json_path = self.clip_path.with_suffix(".json")
                try:
                    with open(json_path, "w") as f:
                        json.dump(self.meta, f, indent=2)
                except OSError as e:
                    print(f"[CLIP] WARN: failed to write metadata json: {e}")

                print(
                    f"[CLIP] Saved {self.clip_path.name}  "
                    f"duration={self.meta['duration_sec']:.1f}s  "
                    f"det_frames={det_n}  "
                    f"peak_conf={self.meta['peak_confidence']:.2f}"
                )

        self.state = self.IDLE
        self.clip_path = None
        self.meta = {}
