"""
inference_pipeline.py
Real-time goose detection pipeline for Jetson Orin Nano Super.

Flow:
  Raspberry Pi HQ Camera (CSI)
    → GStreamer/OpenCV frame capture
    → EfficientDetD0 (TensorRT engine or TFLite)
    → Post-processing (confidence threshold + NMS)
    → GPIO signal → Laser deterrent system

Requirements:
    pip install opencv-python-headless numpy tensorrt pycuda Jetson.GPIO

Usage:
    # TensorRT engine (recommended for production):
    python inference/inference_pipeline.py --engine models/goose_trt.engine

    # TFLite fallback:
    python inference/inference_pipeline.py --tflite models/goose.tflite

    # Test on a single image (no camera):
    python inference/inference_pipeline.py --engine models/goose_trt.engine --test_image path/to/image.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

INPUT_SIZE = 512          # EfficientDetD0 input resolution
CONFIDENCE_THRESHOLD = 0.70
NMS_IOU_THRESHOLD = 0.45
TRIGGER_COOLDOWN_SEC = 2.0   # seconds between laser triggers
GPIO_PIN = 18                 # BCM pin number for laser signal output


# ── GStreamer pipeline for Raspberry Pi HQ Camera on Jetson ──────────────────

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=INPUT_SIZE,
    display_height=INPUT_SIZE,
    framerate=30,
    flip_method=0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


# ── GPIO Setup ───────────────────────────────────────────────────────────────

def setup_gpio():
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
        return GPIO
    except ImportError:
        print("[WARN] Jetson.GPIO not available — running in simulation mode (no laser output)")
        return None


def trigger_laser(gpio, duration_ms=100):
    """Send a pulse on the GPIO pin to activate the laser deterrent."""
    if gpio is None:
        print("[SIM] Laser triggered!")
        return
    import Jetson.GPIO as GPIO
    GPIO.output(GPIO_PIN, GPIO.HIGH)
    time.sleep(duration_ms / 1000.0)
    GPIO.output(GPIO_PIN, GPIO.LOW)


# ── TensorRT Inference ───────────────────────────────────────────────────────

class TRTDetector:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.cuda = cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

        self.stream = cuda.Stream()

    def infer(self, image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a pre-processed 512x512 BGR image.
        Returns (boxes, scores, classes) — boxes in [ymin, xmin, ymax, xmax] normalized format.
        """
        preprocessed = preprocess(image_np)
        np.copyto(self.inputs[0]["host"], preprocessed.ravel())

        for inp in self.inputs:
            self.cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        # EfficientDet outputs: [boxes, scores, classes, num_detections]
        num_detections = int(self.outputs[3]["host"][0])
        boxes = self.outputs[0]["host"][:num_detections * 4].reshape(num_detections, 4)
        scores = self.outputs[1]["host"][:num_detections]
        classes = self.outputs[2]["host"][:num_detections].astype(int)
        return boxes, scores, classes


# ── TFLite Inference (fallback) ──────────────────────────────────────────────

class TFLiteDetector:
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preprocessed = preprocess(image_np)
        self.interpreter.set_tensor(self.input_details[0]["index"], preprocessed)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0].astype(int)
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        return boxes, scores, classes


# ── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    """Resize to 512x512, normalize to [0,1], add batch dimension."""
    resized = cv2.resize(bgr_frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


# ── Post-processing & Drawing ─────────────────────────────────────────────────

def filter_detections(boxes, scores, classes, conf_threshold):
    """Keep only high-confidence goose detections (class_id=1)."""
    mask = (scores >= conf_threshold) & (classes == 1)
    return boxes[mask], scores[mask], classes[mask]


def draw_detections(frame: np.ndarray, boxes, scores, orig_h: int, orig_w: int) -> np.ndarray:
    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * orig_w)
        y1 = int(ymin * orig_h)
        x2 = int(xmax * orig_w)
        y2 = int(ymax * orig_h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Goose {score:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


# ── Main Loop ────────────────────────────────────────────────────────────────

def run_camera_loop(detector, gpio):
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open camera. Check CSI connection and nvargus-daemon status."
        )

    last_trigger_time = 0.0
    print("Camera running — press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Dropped frame")
                continue

            orig_h, orig_w = frame.shape[:2]
            t0 = time.perf_counter()

            boxes, scores, classes = detector.infer(frame)
            boxes, scores, classes = filter_detections(boxes, scores, classes, CONFIDENCE_THRESHOLD)

            latency_ms = (time.perf_counter() - t0) * 1000

            if len(scores) > 0:
                now = time.time()
                if now - last_trigger_time > TRIGGER_COOLDOWN_SEC:
                    trigger_laser(gpio)
                    last_trigger_time = now
                    print(f"[DETECT] Goose detected! Confidence={scores[0]:.2f}  Latency={latency_ms:.1f}ms")

            annotated = draw_detections(frame, boxes, scores, orig_h, orig_w)
            cv2.putText(
                annotated,
                f"Latency: {latency_ms:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Goose Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if gpio:
            import Jetson.GPIO as GPIO
            GPIO.cleanup()


def run_test_image(detector, image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_h, orig_w = frame.shape[:2]
    t0 = time.perf_counter()
    boxes, scores, classes = detector.infer(frame)
    boxes, scores, classes = filter_detections(boxes, scores, classes, CONFIDENCE_THRESHOLD)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"Detections: {len(scores)}  |  Latency: {latency_ms:.1f}ms")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  [{i}] score={score:.3f}  box={box}")

    annotated = draw_detections(frame, boxes, scores, orig_h, orig_w)
    out_path = Path(image_path).with_suffix(".detected.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EfficientDetD0 Goose Inference Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine", help="Path to TensorRT .engine file")
    group.add_argument("--tflite", help="Path to TFLite .tflite model")
    parser.add_argument("--test_image", help="Run on a single image instead of camera")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Detection confidence threshold (default {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    if args.engine:
        print(f"Loading TensorRT engine: {args.engine}")
        detector = TRTDetector(args.engine)
    else:
        print(f"Loading TFLite model: {args.tflite}")
        detector = TFLiteDetector(args.tflite)

    if args.test_image:
        run_test_image(detector, args.test_image)
    else:
        gpio = setup_gpio()
        run_camera_loop(detector, gpio)


if __name__ == "__main__":
    main()
