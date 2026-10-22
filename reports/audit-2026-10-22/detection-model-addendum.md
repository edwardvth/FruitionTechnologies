# Addendum — live goose detection model review

## Plain-language guide

The original audit focused on hardware safety, the dataset and the EfficientDet scripts. This addendum looks at the **YOLO goose detector that the live camera program actually uses**: how it was trained, how its score should be read, and how the live code turns detections into a “goose present” decision. It found a debug-mode bug that changes tracking behavior, a confidence setting that disables part of the tracker, and training/validation choices that make the reported 0.715 score less reliable than it looks. None of these mean the detector is bad. They mean its real field performance is not yet measured, and there are cheap ways to make it better and faster.

“Confirmed” means the code, the saved training files, or the exact Ultralytics library source show the behavior. “Review needed” means there is a reason to suspect a problem that has not been measured. Nothing here was retrained or run on the Jetson.

Scope: `edwardvth/FruitionTechnologies` at `e4d516589089f83eea5b46d228c82c47eca204e6`, 2026-10-22. Library behavior was checked against **Ultralytics 8.4.39** source (the version string recorded inside `models/goose_yolo/weights/best.pt`), wheel `ultralytics-8.4.39-py3-none-any.whl`, SHA-256 `df7de9425bafd1a193d631fa38daa3140bc3c91ae03c24f91a19279004cdeccb`. The wheel was unzipped for reading only, not installed.

## Correction to the main audit

`README.md` (Repository structure section) says the live pipeline “loads `yolo26n.pt` as a person-safety model.” At the audited commit `b44d6e6` and at current `main`, `inference_pipeline.py:49` loads **`yolo11n.pt`**. `yolo26n.pt` is tracked in `goose_detection/` but nothing references it. The goose model is fine-tuned from **`yolo11s.pt`** (`models/goose_yolo/args.yaml: model`). The R3 finding is unaffected, but anyone evaluating the person detector should evaluate `yolo11n.pt`, not `yolo26n.pt`.

## Findings

| ID | Finding | What it means in practice | Next action | Evidence / certainty |
|---|---|---|---|---|
| **D1** | `--debug` mode feeds extra frames into the goose tracker | Turning on debug changes what the detector reports and how tracks behave, so debug runs do not reflect normal runs. Debug's “raw” confidence list is not raw. | Use a separate model instance for debug, or skip the extra predict. | Confirmed from library source. |
| **D2** | `conf=0.50` is applied before ByteTrack | The tracker's second-chance matching of weaker detections never runs. A goose whose score dips below 0.5 for one frame drops out, so the “goose present” signal flickers and track IDs break. | Feed the tracker low-confidence boxes (≈0.1) and apply the 0.5 rule to tracks, not raw boxes. | Confirmed from library source. |
| **D3** | One frame with one box is enough to declare a goose | A single false-alarm frame counts the same as a goose seen for two seconds. | Require a confirmed track seen in several recent frames. | Confirmed in code. |
| **D4** | The 0.5 threshold was not chosen from the model's own curve | The saved F1 curve peaks at **0.86 at confidence 0.261**; at 0.5 the curve is roughly 0.81. The threshold trades away recall without a stated reason. | Pick the threshold from test/field data for the actual goal (misses vs. false alarms). | Read from `BoxF1_curve.png`. |
| **D5** | Training used `batch=1` | Validation scores swing wildly between neighboring epochs, which makes “best epoch” partly luck. Training also took ~26 hours. | Retrain with a real batch size and confirm the curve smooths. | Batch confirmed; cause of the swings is likely, not proven. |
| **D6** | 0.715 is the luckiest of 76 noisy validation scores, and the test split was never scored | The headline number is optimistic. The independent test split exists but no saved result evaluates it. | Run `model.val(split="test")` on `best.pt` and report that number instead. | Confirmed from `results.csv` and scripts. |
| **D7** | The person-safety model is the smallest, least accurate one | The model whose misses matter most is the nano model; the goose model is the larger “s” model. | Use a stronger person model and pay for it with TensorRT (D9). | Confirmed in code; COCO figures from Ultralytics docs. |
| **D8** | Camera frames are shrunk to 640 px | Geese far from the camera become a few pixels tall. Small-object recall has not been measured. | Measure recall by goose size; try `imgsz` 960/1280 or tiling if distant geese matter. | Review needed. |
| **D9** | Both models run as plain PyTorch FP32 on the Jetson | Slower than necessary, which also delays every safety decision (main audit R1). | Export both to TensorRT FP16 and measure p95/p99 latency. | Confirmed in code; speedup not measured here. |
| **D10** | Saved clips store annotated frames only | Field mistakes can't easily be turned into new training data, because the boxes and HUD are burned into the pixels. | Also save raw frames or crops for false alarms and near-misses. | Confirmed in code. |

## Technical detail

### D1 — Debug predict calls share the tracker callbacks (High for debug reliability; confirmed)

`inference_pipeline.py:311-317` calls `tracker.model.predict(frame, conf=0.01, ...)` every 15 frames on the **same `YOLO` object** that `GooseTracker.track()` uses with `persist=True`.

In Ultralytics 8.4.39, `Model.track()` calls `register_tracker` (`engine/model.py:574-576`), which adds the tracking callbacks to the model's shared callback dictionary (`trackers/track.py:115-116`). `predict()` reuses the existing predictor (`engine/model.py:528-532`), and the predictor holds that same dictionary (`engine/predictor.py:147`). So **every later `predict()` call also runs `on_predict_postprocess_end`** (`predictor.py:343`), which:

1. calls `tracker.update(...)` with this frame's detections (`track.py:93`). With `conf=0.01`, ByteTrack sees detections in the 0.1–0.25 band that normal frames never pass it, and its Kalman filter advances an extra step on a frame that has already been tracked;
2. replaces the result with tracker output (`track.py:97`). The values printed as `goose_raw_confs` are tracked-box scores, not raw detector scores.

The debug JPEGs and the warm-up call are not affected. The same applies to the safety model only if it were ever tracked; it isn't.

**Fix:** load a second `YOLO(weights)` instance for debug probes, or read raw scores in a way that doesn't go through the tracked model. Add a test that runs `--debug` and normal mode over the same recorded clip and compares track IDs.

### D2 — Confidence threshold defeats ByteTrack (Medium; confirmed)

`GooseTracker.track()` passes `conf=self.conf` (0.50) to `model.track()` (`inference_pipeline.py:162-170`). Ultralytics itself defaults tracking to `conf=0.1` with the comment “ByteTrack-based method needs low confidence predictions as input” (`engine/model.py:577`). The default `bytetrack.yaml` uses `track_high_thresh: 0.25`, `track_low_thresh: 0.1`, `new_track_thresh: 0.25`. With a 0.5 pre-filter, the low-score second association (ByteTrack's main idea) never receives any boxes, and detections between 0.25 and 0.5 can't keep a track alive.

**Fix:** call `track(conf=0.1)` and decide “goose present” from track state: e.g. a track with ID ≥ 0, seen in ≥ N of the last M frames, with a recent score above the operating threshold. Tune `track_buffer` for the real frame rate (the tracker assumes `frame_rate=30`, `track.py` `on_predict_start`; the pipeline runs at about 15–18 FPS according to its own comments).

### D3 — No temporal confirmation (Medium; confirmed)

`goose_present = len(g_scores) > 0` (`inference_pipeline.py:347`) and the laser decision follows on the same frame. Untracked boxes (`id = -1`) count too. Recommend a small state machine: on after a track is confirmed for k frames, off immediately on person or loss (this keeps main-audit R1/R3 fail-closed behavior). This reduces single-frame false alarms from other birds, reflections, or debris without lowering recall much.

### D4 — Operating threshold (Low–Medium; confirmed)

`BoxF1_curve.png` reports **F1 0.86 at confidence 0.261**. The 0.5 threshold sits on the falling side of the curve. The `precision 0.894 / recall 0.826` row in `results.csv` comes from Ultralytics' own max-F1 point, **not** from 0.5, so it does not describe the live setting. The saved confusion matrix (final validation of `best.pt`) shows **179 found, 36 missed, 22 false alarms** out of the 215 validation boxes at the validator's plotting threshold. Choose the threshold on the test split plus field video, and write down why.

### D5 — `batch=1` training (Medium; batch confirmed, effect likely)

`args.yaml` records `batch: 1`, matching `commands.txt:9` (`--batch 1`). Ultralytics compensates for gradient size with accumulation (`trainer.py:281`, `nbs=64`), but BatchNorm statistics are still computed from one image at a time. `results.csv` shows large epoch-to-epoch swings in validation mAP50–95:

| Epochs | mAP50–95 |
|---|---|
| 57 → 58 → 59 | 0.601 → **0.374** → 0.662 |
| 65 → 66 → 67 → 68 | 0.665 → 0.472 → **0.391** → 0.593 |

Validation loss jumps the same way, so this is not a metric glitch. Training ran ~94,000 s (~26 h) for 76 epochs. **Fix:** train with batch 16–32 (cloud GPU or a larger machine if the Jetson/laptop can't fit it), or `batch=-1` for auto-sizing. Then check that the curve is smooth.

### D6 — Headline number is optimistic; no test-split score (Medium; confirmed)

Early stopping (`patience=15`) stopped at epoch 76 because the peak was epoch 61. `best.pt` is the maximum of a noisy series on the same 707-image validation set used to select it, so 0.715 includes selection luck. The last epoch scored 0.647. No script or saved artifact evaluates the 708-image test split (`test_yolo.py --test_split` only draws pictures). **Fix:** `YOLO(best.pt).val(data=..., split="test", conf=0.001)` once, record the number with the commit hash, and treat that as the reported score. With only ~212 positive test images, report a confidence interval.

### D7 — Weakest model guards people (Medium; confirmed)

Goose: YOLO11s. Person: YOLO11n (`inference_pipeline.py:49`). Ultralytics' published COCO mAP50–95 is **39.5 for YOLO11n vs 47.0 for YOLO11s** (all 80 classes, 640 px; [docs](https://docs.ultralytics.com/models/yolo11/)). That is not a person-recall figure, but it shows the safety check runs on the least capable option. With TensorRT (D9) a larger person model is affordable. A single combined goose + person model would halve inference but tie person safety to the goose fine-tune, so keep them separate unless person recall is measured afterward.

### D8 — Input resolution (Review needed)

The camera delivers 1280×720 (`gstreamer_pipeline`), and both models run at their trained 640 px, so everything shrinks by 2×. Training also used 640. Nothing measures recall for small or distant geese. Slice the test/field results by box size; if distant geese are missed, try `imgsz=960/1280` (train and infer) or tiled inference.

### D9 — No TensorRT / FP16 (Low–Medium; confirmed)

Both models load `.pt` files and run in PyTorch with no `half=True` and no exported engine. On Jetson, Ultralytics recommends TensorRT export (`model.export(format="engine", half=True)`). Lower latency also shortens the window in main-audit R1 where an old decision stays applied. Measure before/after p95/p99 on the device, and verify the exported model gives the same detections on the test split.

### D10 — No feedback loop from the field (Improvement)

`recorder.on_frame` receives `annotated` frames (`inference_pipeline.py:385-392`), with boxes, HUD and banner drawn on them. Save raw frames (or raw crops plus box JSON) for false alarms, low-confidence detections and person/goose overlaps. That turns every field day into labeled hard examples, which is the most effective way to improve this model.

## Smaller notes

- Module and script docstrings call the goose model YOLOv11s; that matches `args.yaml`. `GOOSE_CLASS_ID = 0` matches the single-class config (`canadian_goose`).
- `args.yaml` records `data: /tmp/tmpiljhsfct.yaml` (a temp file deleted after training) and a machine-specific `save_dir`, so the exact data config can't be recovered from the run. Write the data YAML into the run folder.
- Debug mode runs two extra full-frame inferences synchronously inside the safety loop every 15 frames, causing periodic latency spikes. Another reason to separate debug probes (D1).

## Suggested order

1. D1 + D2 + D3 together (same ~40 lines of `GooseTracker` / decision code), with a recorded-clip regression test.
2. D6: score `best.pt` on the test split now; that becomes the baseline.
3. D5 + D8: retrain with a proper batch size, compare against the baseline, slice by goose size.
4. D9 + D7: TensorRT export, then upgrade the person model and measure person recall separately.
5. D4 + D10: set the threshold from data and start collecting field hard examples.
