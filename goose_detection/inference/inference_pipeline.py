"""
inference_pipeline.py
Real-time goose detection pipeline for Jetson Orin Nano Super.

Flow:
  Arducam UC-517 (CSI, IMX477) via nvarguscamerasrc
    -> OpenCV frame capture
    -> YOLOv11s goose tracker (Ultralytics, canadian_goose)   ─┐
    -> YOLOv11n stock COCO person-safety check                 ├─ per frame
    -> Annotated preview + "GOOSE" terminal alerts             ┘
    -> Clip recorder (pre-roll + post-roll MP4 + .json)
    -> GPIO laser state: HIGH only if (goose_present AND NOT person_present)

Safety:
    Laser GPIO is forced LOW on any exit path (atexit + SIGINT + SIGTERM) and
    also forced LOW for every frame a person is detected. Laser hardware is
    not yet connected; signalling runs in simulation mode without Jetson.GPIO.

Usage:
    # Live camera (GUI):
    python inference/inference_pipeline.py

    # Live camera (headless, for systemd):
    python inference/inference_pipeline.py --headless

    # Single image (no camera):
    python inference/inference_pipeline.py --test_image test_if_goose/some.jpg
"""

import argparse
import atexit
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from inference.clip_recorder import ClipRecorder


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = "models/goose_yolo/weights/best.pt"
DEFAULT_SAFETY_WEIGHTS = "yolo11n.pt"   # COCO pretrained, downloaded on first use if absent
CONFIDENCE_THRESHOLD = 0.50
SAFETY_CONF_THRESHOLD = 0.35             # lower = more aggressive safety disabling
GOOSE_CLASS_ID = 0                       # in custom model: 0 = canadian_goose
PERSON_CLASS_ID = 0                      # in COCO: 0 = person

ALERT_COOLDOWN_SEC = 1.0
GPIO_PIN = 18
CLIP_OUTPUT_DIR = "results/clips"
CLIP_FPS = 15.0
PRE_ROLL_SEC = 10.0
POST_ROLL_SEC = 3.0


# ── GStreamer pipeline for CSI camera on Jetson ──────────────────────────────

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


# ── GPIO laser control with unconditional failsafe ───────────────────────────

_gpio_handle = None
_laser_state = False
_last_sim_print = 0.0  # throttle the [SIM] print so we don't spam at 18fps


def setup_gpio():
    global _gpio_handle
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
        _gpio_handle = GPIO
        return GPIO
    except Exception as e:
        print(f"[GPIO] unavailable, falling back to simulation: {e}")
        _gpio_handle = None
        return None


def _set_laser(on: bool) -> None:
    global _laser_state, _last_sim_print
    _laser_state = on
    if _gpio_handle is not None:
        _gpio_handle.output(GPIO_PIN, _gpio_handle.HIGH if on else _gpio_handle.LOW)
        return
    now = time.time()
    if now - _last_sim_print > 1.0:
        print(f"[SIM] Laser {'ON' if on else 'OFF'}")
        _last_sim_print = now


def laser_on() -> None:
    if not _laser_state:
        _set_laser(True)


def laser_off() -> None:
    if _laser_state:
        _set_laser(False)


def force_laser_off_and_cleanup(*_args) -> None:
    """Unconditional failsafe. Registered on atexit + SIGINT + SIGTERM."""
    global _laser_state
    try:
        if _gpio_handle is not None:
            _gpio_handle.output(GPIO_PIN, _gpio_handle.LOW)
            _gpio_handle.cleanup()
    except Exception:
        pass
    _laser_state = False


atexit.register(force_laser_off_and_cleanup)
signal.signal(signal.SIGINT, lambda *a: (force_laser_off_and_cleanup(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda *a: (force_laser_off_and_cleanup(), sys.exit(0)))


# ── YOLO Detectors ───────────────────────────────────────────────────────────

class GooseTracker:
    """Goose detector with stable track IDs via Ultralytics ByteTrack."""

    def __init__(self, weights_path: str, conf_threshold: float, device: str = "cuda:0"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf_threshold
        self.device = device

    def track(self, bgr_frame: np.ndarray):
        """
        Returns:
            boxes_xyxy: (N,4) pixel coords
            scores:    (N,)
            track_ids: (N,) int  (-1 if tracker didn't assign one yet)
        """
        results = self.model.track(
            source=bgr_frame,
            conf=self.conf,
            classes=[GOOSE_CLASS_ID],
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(np.int32)
        else:
            ids = np.full(len(boxes), -1, dtype=np.int32)
        return boxes, scores, ids


class PersonSafetyDetector:
    """Stateless per-frame person detector using a stock COCO model."""

    def __init__(self, weights_path: str, conf_threshold: float, device: str = "cuda:0"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf_threshold
        self.device = device

    def detect(self, bgr_frame: np.ndarray):
        results = self.model.predict(
            source=bgr_frame,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return r.boxes.xyxy.cpu().numpy()


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_geese(frame: np.ndarray, boxes_xyxy, scores, track_ids) -> np.ndarray:
    for (x1, y1, x2, y2), score, tid in zip(boxes_xyxy, scores, track_ids):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        label = f"Goose#{int(tid)} {score:.2f}" if tid >= 0 else f"Goose {score:.2f}"
        cv2.putText(frame, label, (p1[0], max(p1[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_people(frame: np.ndarray, boxes_xyxy) -> np.ndarray:
    for (x1, y1, x2, y2) in boxes_xyxy:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
        cv2.putText(frame, "PERSON", (p1[0], max(p1[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def draw_safety_banner(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "! PERSON DETECTED  -  LASER DISABLED",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return frame


# ── Main Loop ────────────────────────────────────────────────────────────────

def run_camera_loop(
    tracker: GooseTracker,
    safety: PersonSafetyDetector,
    save_clips: bool = True,
    clip_fps: float = CLIP_FPS,
    headless: bool = False,
    debug: bool = False,
):
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open camera. Check /dev/video0, CSI ribbon seating, "
            "and `sudo systemctl status nvargus-daemon`."
        )

    setup_gpio()

    debug_dir = None
    if debug:
        import torch
        debug_dir = Path("results/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        cuda_ok = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
        print(f"[DBG] CUDA={cuda_ok}  device={dev_name}  torch={torch.__version__}")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        tracker.model.predict(dummy, conf=0.01, device=tracker.device, verbose=False)
        print(f"[DBG] Goose warmup: {(time.perf_counter()-t0)*1000:.0f}ms")
        t0 = time.perf_counter()
        safety.model.predict(dummy, conf=0.01, device=safety.device, verbose=False)
        print(f"[DBG] Safety warmup: {(time.perf_counter()-t0)*1000:.0f}ms")
        print(f"[DBG] Saving raw frames every 30 frames -> {debug_dir.resolve()}")

    recorder = None
    if save_clips:
        recorder = ClipRecorder(
            output_dir=CLIP_OUTPUT_DIR,
            fps=clip_fps,
            pre_roll_sec=PRE_ROLL_SEC,
            post_roll_sec=POST_ROLL_SEC,
        )
        print(f"Clips -> {Path(CLIP_OUTPUT_DIR).resolve()} (pre={PRE_ROLL_SEC}s, post={POST_ROLL_SEC}s, fps={clip_fps})")

    last_alert_time = 0.0
    frame_count = 0
    t_window = time.perf_counter()
    fps_display = None
    print(f"Camera running ({'headless' if headless else 'GUI'}). "
          f"{'Ctrl+C' if headless else 'q in window or Ctrl+C'} to stop.")

    debug_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Dropped frame")
                continue

            debug_counter += 1

            t0 = time.perf_counter()
            g_boxes, g_scores, g_ids = tracker.track(frame)
            p_boxes = safety.detect(frame)
            latency_ms = (time.perf_counter() - t0) * 1000

            if debug and debug_counter % 15 == 0:
                g_raw = tracker.model.predict(
                    frame, conf=0.01, device=tracker.device, verbose=False
                )[0]
                p_raw = safety.model.predict(
                    frame, conf=0.01, device=safety.device, verbose=False
                )[0]
                g_confs = (
                    g_raw.boxes.conf.cpu().numpy().tolist()
                    if g_raw.boxes is not None and len(g_raw.boxes) > 0
                    else []
                )
                p_confs = (
                    p_raw.boxes.conf.cpu().numpy().tolist()
                    if p_raw.boxes is not None and len(p_raw.boxes) > 0
                    else []
                )
                p_classes = (
                    p_raw.boxes.cls.cpu().numpy().astype(int).tolist()
                    if p_raw.boxes is not None and len(p_raw.boxes) > 0
                    else []
                )
                p_names = getattr(p_raw, "names", {})
                g_top = [f"{s:.2f}" for s in sorted(g_confs, reverse=True)[:5]]
                p_top = [
                    f"{p_names.get(c, c)}={s:.2f}"
                    for c, s in sorted(zip(p_classes, p_confs), key=lambda x: -x[1])[:5]
                ]
                print(
                    f"[DBG f={debug_counter}] goose_raw_confs={g_top} "
                    f"person_raw={p_top}  filt_g={len(g_scores)} filt_p={len(p_boxes)}"
                )

            if debug and debug_counter % 30 == 0 and debug_dir is not None:
                cv2.imwrite(str(debug_dir / f"frame_{debug_counter:06d}.jpg"), frame)

            goose_present = len(g_scores) > 0
            person_present = len(p_boxes) > 0
            top = float(g_scores.max()) if goose_present else 0.0

            # SAFETY-CRITICAL: person gate comes BEFORE laser_on.
            if goose_present and not person_present:
                laser_on()
            else:
                laser_off()

            if goose_present:
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN_SEC:
                    unique_ids = sorted({int(i) for i in g_ids if i >= 0})
                    ids_str = f" ids={unique_ids}" if unique_ids else ""
                    print(f"GOOSE  (n={len(g_scores)}{ids_str}  top_conf={top:.2f}  latency={latency_ms:.1f}ms)")
                    last_alert_time = now

            annotated = draw_geese(frame, g_boxes, g_scores, g_ids)
            annotated = draw_people(annotated, p_boxes)

            frame_count += 1
            elapsed = time.perf_counter() - t_window
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                t_window = time.perf_counter()

            hud = f"Latency: {latency_ms:.1f}ms"
            if fps_display is not None:
                hud += f"  FPS: {fps_display:.1f}"
            hud += f"  Laser: {'ON' if _laser_state else 'OFF'}"
            cv2.putText(annotated, hud, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

            if person_present:
                annotated = draw_safety_banner(annotated)

            if recorder is not None:
                recorder.on_frame(
                    annotated,
                    has_detection=goose_present,
                    top_conf=top,
                    detection_count=len(g_scores),
                    track_ids=[int(i) for i in g_ids if i >= 0],
                )

            if not headless:
                cv2.imshow("Goose Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        laser_off()
        if recorder is not None:
            recorder.close()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def run_test_image(tracker: GooseTracker, safety: PersonSafetyDetector, image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    t0 = time.perf_counter()
    g_boxes, g_scores, g_ids = tracker.track(frame)
    p_boxes = safety.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"Geese: {len(g_scores)}  Persons: {len(p_boxes)}  Latency: {latency_ms:.1f}ms")
    for i, (box, score, tid) in enumerate(zip(g_boxes, g_scores, g_ids)):
        x1, y1, x2, y2 = box
        print(f"  [{i}] id={int(tid)} score={score:.3f}  xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    if len(g_scores) > 0:
        print("GOOSE" + ("  (laser suppressed: person present)" if len(p_boxes) else ""))

    annotated = draw_geese(frame, g_boxes, g_scores, g_ids)
    annotated = draw_people(annotated, p_boxes)
    if len(p_boxes) > 0:
        annotated = draw_safety_banner(annotated)

    out_path = Path(image_path).with_suffix(".detected.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Goose Detector — tracked + person-safety interlock")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"Goose model weights (.pt). Default: {DEFAULT_WEIGHTS}")
    parser.add_argument("--safety-weights", default=DEFAULT_SAFETY_WEIGHTS,
                        help=f"Person-safety model weights (COCO pretrained .pt). Default: {DEFAULT_SAFETY_WEIGHTS}")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Goose confidence threshold (default {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--safety-conf", type=float, default=SAFETY_CONF_THRESHOLD,
                        help=f"Person-detection confidence threshold (default {SAFETY_CONF_THRESHOLD})")
    parser.add_argument("--device", default="cuda:0", help="Inference device (default cuda:0)")
    parser.add_argument("--test_image", help="Single-image mode instead of live camera")
    parser.add_argument("--no-save-clips", action="store_true",
                        help=f"Disable saving detection clips to {CLIP_OUTPUT_DIR}/")
    parser.add_argument("--clip-fps", type=float, default=CLIP_FPS,
                        help=f"Playback FPS for saved clips (default {CLIP_FPS})")
    parser.add_argument("--headless", action="store_true",
                        help="Skip the cv2.imshow preview window (for systemd / SSH runs)")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw detector outputs every 15 frames + dump raw camera frames to results/debug/")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"Goose weights not found: {args.weights}. Run from project root or pass --weights."
        )

    print(f"Goose model:  {args.weights}  (conf={args.conf})")
    print(f"Safety model: {args.safety_weights}  (conf={args.safety_conf})")
    tracker = GooseTracker(args.weights, conf_threshold=args.conf, device=args.device)
    safety = PersonSafetyDetector(args.safety_weights, conf_threshold=args.safety_conf, device=args.device)

    if args.test_image:
        run_test_image(tracker, safety, args.test_image)
    else:
        run_camera_loop(
            tracker, safety,
            save_clips=not args.no_save_clips,
            clip_fps=args.clip_fps,
            headless=args.headless,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
