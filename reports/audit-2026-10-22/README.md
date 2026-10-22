# Fruition Technologies repository and model audit

**Audit date:** 2026-10-22. **Repository:** [edwardvth/FruitionTechnologies](https://github.com/edwardvth/FruitionTechnologies). **Reviewed commit:** `b44d6e69708eff6057818973a453b18e1dfd5db5` (`main` at inspection). Three parallel reviewers examined data, runtime/deployment, and model evaluation/software trust. Application code and weights were not changed.

## Summary — what we found and what it means

**The project contains training and live-camera software with saved training results, but this audit found important problems to address before trusting the results or allowing the software to control connected laser hardware.** The most urgent problem is that the software can leave its previous “laser on” command in place when the camera stops providing images. We reproduced that software behavior with a simulated output; we did not connect or test a real laser.

Three reviewers worked in parallel. We examined the project files, checked the dataset, inspected the saved results, and ran small controlled tests of the code. **This was an audit, not a repair or a fresh accuracy test.** Publishing these reports to GitHub main does not change the application or fix any finding. The next work is to make the corrections below and test them.

The project contains two goose-detection approaches: **YOLO**, which is used by the live camera program, and **EfficientDet**, which has separate training and image-testing scripts. Several serious training/reporting problems apply specifically to EfficientDet; they do not automatically mean the live YOLO detector has those same problems. The live program also uses a separate model to look for people and suppress the laser command.

### The most important takeaways

- **Address hardware controls first.** Losing the camera feed can preserve an old “on” command. Hardware can also become active automatically when the GPIO library works, and the program does not verify that the chosen safety model actually identifies people in the expected way. These are confirmed code weaknesses; the actual electrical behavior and real-world risk require testing on the intended hardware.
- **Do not treat the EfficientDet “best” score as accuracy.** It rewards how confident the model sounds, even when its answer is wrong. Our controlled example gave it a completely misplaced detection and it still returned a score of 0.99. Its image-testing script also draws boxes with horizontal and vertical coordinates mixed up.
- **A fresh copy of the repository is missing files that EfficientDet training needs.** All 6,675 referenced processed images are absent from Git, and the trainer silently replaces missing images with blank samples. The YOLO images are present. This does not prove the existing saved model was trained on blank images; its original computer may have had the missing files.
- **There are useful positive results.** We checked 7,115 YOLO images and found no identical image files shared between training, validation and test groups, no missing matching label files, and no malformed or out-of-bounds box labels. This does not check whether every label is visually correct or whether similar pictures of the same scene occur in different groups.
- **The saved YOLO results are promising evidence to investigate, not a deployment guarantee.** Its best recorded validation mAP50–95 is 0.71481. That is an object-detection benchmark score, **not “71.5% accurate”** and not a measure of person safety. We read that historical result; we did not rerun it or measure the current system outdoors.
- **Addendum: the live YOLO detection logic has its own problems.** Debug mode quietly changes tracking, the 0.5 confidence filter disables part of the tracker, a single frame counts as a goose, and the 0.715 score is the best of a very noisy training run that used batch size 1. The test split was never scored. Details and fixes in the [detection model addendum](detection-model-addendum.md) (D1–D10).

### Every finding in everyday language

“Confirmed” means the code or a controlled test supports the finding. “Review needed” means there is a reason to investigate, but the suspected real-world error has not been demonstrated. Importance indicates when to address it, not how likely an attack is.

| Finding | What this means in practice | Importance and next action | Evidence / certainty |
|---|---|---|---|
| **R1: An old laser command can remain on** | The camera can stop supplying pictures while the software keeps the last “on” decision. A slow or stuck prediction can also delay a new safety decision. | **Before hardware use:** turn off on lost/stale images and independently expire every on-command. Test the actual hardware's default-off behavior. | Confirmed using simulated GPIO; real laser behavior untested. |
| **R2: Hardware mode is automatic** | A command used for a demonstration can start controlling a connected device simply because the hardware library is available. | **Before hardware use:** default to simulation and require deliberate arming after checks. | Confirmed in code. |
| **R3: The person detector is trusted without identity checks** | Selecting the wrong compatible model could cause “no person detected” to mean “the wrong category was checked.” | **Before hardware use:** verify the approved model file and its category names before enabling output. | Missing checks confirmed; no evidence that the current default model is mislabelled. |
| **DATA-01: Training inputs are missing in a fresh download** | EfficientDet can appear to train while receiving blank placeholders instead of the intended photos. | **Before EfficientDet training:** restore verified images and stop with a clear error if any are missing. | All 6,675 required processed-image paths absent from Git; fallback confirmed in code. |
| **M1: Confidence is mistaken for correctness** | A confidently wrong EfficientDet model can be chosen as “best.” | **Before trusting its results:** compare predictions to human labels and measure missed objects and false alarms. | Reproduced: a wrong-only example scored 0.99. |
| **M2: Detection boxes are drawn in the wrong place** | The EfficientDet test script mixes up horizontal and vertical positions, making its output misleading. | **Before interpreting these images:** fix coordinate order and test with an off-center object. | Reproduced against the pinned library's documented output. |
| **M3: “Resume” does not request a true continuation** | YOLO can start a new training session from saved weights instead of restoring the interrupted training state. | **Before relying on resume:** use the library's resume option and verify the restored training step and optimizer. | Code and official library contract confirm the missing option; training was not rerun. |
| **M4: Software/model provenance is not enforced** | Different computers can install different dependency versions; a replaced or untrusted model file may be unsafe to load. Incomplete weights can also load without a strict check. | **Before a release:** record exact supported versions, verify trusted file hashes, check actual installed versions for vulnerabilities and reject incompatible weights. | Missing protections confirmed. Exploitation and the Jetson's installed vulnerability status are unknown. |
| **S1: Downloaded datasets can name unapproved files** | Data downloads use unencrypted HTTP, and a crafted manifest can point outside the expected image folder. Importing it could copy unintended local files into generated data. | **Before importing new data:** verify downloads and restrict every file path to the approved folder. | Confirmed code trust boundary; requires malicious/intercepted input, and no attack was executed. |
| **DATA-02: “No goose” labels need review** | One generator labels every source image as background without checking whether geese are present. Wrong negatives would teach the wrong lesson. | **Before retraining:** review likely bird images, label any actual geese and keep a review record. | Missing filtering confirmed; number of wrongly labelled images unknown. |
| **DATA-03: Some boxes cover the whole picture** | A box may describe the scene instead of the goose, teaching or scoring the wrong location. Some full-picture boxes may be legitimate. | **Before retraining/evaluation:** visually review all 28 whole-frame box candidates and correct those that need it. | 28 candidates measured; not 28 proven errors. |
| **DATA-04: Image-preparation errors remove answers** | If an image transformation fails, EfficientDet keeps the image but throws away its object labels without reporting the problem. | **Before retraining:** stop visibly or use a fallback that correctly transforms both image and labels. | Confirmed in code; occurrence rate not measured. |
| **DATA-05: Rebuilding data can preserve mistakes** | Reused folders and incomplete duplicate handling can leave old files, repeated labels or unreviewed negatives in a future rebuild. | **For reliable rebuilds:** generate fresh output from one verified manifest and validate it before use. | Future code risk; current exact-duplicate checks passed. |
| **R4: Recording can fail or shutdown can get stuck** | A video-writing error can stop the worker that saves clips; a full queue can prevent clean shutdown. | **For unattended use:** report recorder health, handle errors and bound every shutdown wait. | Exception and queue behavior reproduced with mocks. |
| **R5: Recordings and logs can fill the disk** | Continuous operation has no built-in limit on stored clips, debug pictures or appended logs. | **For unattended use:** add retention limits, disk-space checks and log rotation. | Limits absent from repository; actual disk exhaustion not tested. |
| **R6: Clip timestamps can overstate coverage** | A clip can claim seconds of footage from before the event even when that footage was not captured. Variable speed also changes timing assumptions. | **For dependable incident records:** timestamp captured frames and report actual coverage. | Confirmed by code calculation; camera timing unmeasured. |
| **D1–D3: Live detection decision is fragile** | Debug mode feeds extra frames into the tracker; the 0.5 filter blocks ByteTrack's weak-box matching; one frame is enough to say "goose". | **Before field tests:** separate debug model, track at conf 0.1, require a confirmed track over several frames. | Confirmed from code and Ultralytics 8.4.39 source. See [addendum](detection-model-addendum.md). |
| **D4–D6: The 0.715 score overstates what we know** | Training at batch 1 made validation scores swing ±0.3; best.pt is the luckiest epoch; the test split was never scored; 0.5 isn't the curve's best threshold (0.261). | **Before quoting accuracy:** score best.pt on the test split, retrain with a real batch size, pick the threshold from data. | Confirmed from results.csv, args.yaml, F1 curve. |
| **D7–D10: Speed and improvement loop** | Person safety runs on the smallest model; frames are shrunk 2×; no TensorRT; clips can't feed retraining. | **For deployment:** TensorRT FP16, stronger person model, size-sliced recall, save raw field frames. | Code confirmed; gains not measured. |

### Smaller improvements worth doing

The EfficientDet saved “best score” can lag behind its actual newest result; update the saved metadata in the correct order. The YOLO trainer hardcodes GPU 0 despite a comment claiming automatic CPU fallback; choose a supported device explicitly. Old TensorFlow configuration files do not control the current PyTorch trainer; label historical files clearly so operators know which settings apply. Roboflow API keys supplied on the command line can enter shell history; use a protected credential source. These are maintenance and reliability improvements, with details in the model report.

The repository would also benefit from a tested installation guide, automated regression checks, unique training-run records and a field evaluation of the exact models intended for deployment. The person detector needs its own evaluation: good goose detection says nothing by itself about how often a person is missed.

### Terms used in the rest of the reports

| Term | Plain-language meaning |
|---|---|
| Model / weights / checkpoint | A model is the program's learned image-recognition behavior. Weights are its learned numbers; a checkpoint is a saved file containing those numbers and sometimes training state. |
| Training / inference | Training adjusts the learned behavior using examples. Inference uses the saved model to make a prediction about a new image. |
| YOLO / EfficientDet | Two different object-detection model families. Here YOLO powers live camera detection; EfficientDet has separate scripts. |
| Dataset / label / annotation / bounding box | A dataset is a collection of examples. Labels or annotations are the expected answers, including rectangles around objects called bounding boxes. |
| Negative image | A training example labelled as containing no target goose. It must actually have no target goose to be a correct negative. |
| Training / validation / test split | Separate groups of examples: one teaches the model, one guides training choices, and one is reserved for a final independent check. |
| Leakage / duplicate | Leakage happens when information from an evaluation group also influences training. Identical or closely related images across groups can make results look better than performance on new scenes. |
| Confidence | The model's score for a prediction. A high score does not prove the prediction is right. |
| False positive / false negative | A false positive is a false alarm, such as calling another object a goose. A false negative is a missed real object, such as failing to detect a person. |
| Precision / recall | Precision asks how many reported detections are correct; recall asks how many real objects were found. The tradeoff depends on the chosen confidence threshold. |
| IoU / AP / mAP50–95 | IoU measures overlap between a predicted box and the correct box. AP summarizes detection quality across confidence thresholds. mAP50–95 averages AP across categories and several required overlap levels. It is not the percentage of all images answered correctly. |
| GPIO / actuator / interlock / fail closed | GPIO is a hardware control pin. An actuator carries out a physical action. An interlock prevents action under unsafe conditions. Fail closed here means denying the laser-enable command when safety information is missing or invalid. |
| Watchdog / stale decision | A watchdog independently checks that the system is still responsive. A stale decision was based on information too old to trust. |
| Hash / manifest / provenance | A hash is a file fingerprint for checking identity. A manifest lists required files and metadata. Provenance records where files came from and how they were created. A hash alone does not prove a file is trustworthy. |
| Dependency / lockfile / CVE | A dependency is another software package this project uses. A lockfile records exact versions. A CVE identifies a publicly reported vulnerability; it only applies if the relevant version and conditions match. |
| Deserialization / pickle | Turning a saved file back into program objects. Some Python model formats can execute code while loading, so the file source and loader protections matter. |
| Mock / synthetic probe | A controlled test with simulated hardware or made-up predictions. It checks a code path without proving how the complete real system performs. |
| p95 / p99 latency | Response times that 95% or 99% of observations stay below. These expose occasional delays that an average can hide. |
| Confidence interval | A range expressing uncertainty in a measured result. Testing only a few examples cannot establish a precise failure rate. |

## Technical detail and evidence

The following sections retain file locations, measured counts and reproduction instructions for developers. The three linked detailed reports begin with plain-language guides as well.

## Repository structure and actual execution paths

| Location | Role / important relationship |
|---|---|
| Root PDFs, presentation, demonstration video | Project background and demonstrations; not executable model specifications |
| `goose_detection/data/raw/` | Roboflow, images.cv, Gabe YOLO and negative source data |
| `goose_detection/data/processed/*_annotations.json` | COCO split manifests consumed by EfficientDet; referenced processed images are ignored/untracked |
| `goose_detection/data/yolo/images/` and `labels/` | Tracked train/val/test data used by the active YOLO training path |
| `goose_detection/scripts/` | Download/conversion/merge, EfficientDet and YOLO training, visual smoke-test scripts, service installers |
| `goose_detection/configs/` | YOLO YAML plus legacy EfficientDet/TF configuration; not all configuration is consumed |
| `goose_detection/models/` | EfficientDet checkpoints, YOLO checkpoints and training curves/results |
| `goose_detection/inference/inference_pipeline.py` | Live YOLO goose tracking + separate person detector + Jetson GPIO decision loop |
| `goose_detection/inference/clip_recorder.py` | Background event recording, pre-roll and metadata |
| `goose_detection/results/`, `test_if_goose/` | Example output and manual qualitative smoke-test images |
| `reports/` | Audit reports and reproducible probes |

Training has two separate branches: COCO manifests → `train.py` → EfficientDet `.pth` → `test_model.py`; YOLO labels → `train_yolo.py` → `models/goose_yolo/weights/best.pt` → `test_yolo.py` / live inference. The live pipeline targets NVIDIA Jetson and also loads `yolo11n.pt` as a person-safety model (corrected in the [detection addendum](detection-model-addendum.md); `yolo26n.pt` is tracked but unused). Changes to EfficientDet do not automatically repair the YOLO runtime. Eight commits were reachable. No tracked automated CI workflow, unit-test suite, dependency lockfile or main README was present; `test_model.py` and `test_yolo.py` generate visual examples, not automated quantitative regression tests.

## Priority findings

| Priority | Finding | Evidence / detail |
|---|---|---|
| P0 before physical actuation | GPIO HIGH survives failed camera reads and blocking inference | R1, reproduced with mock output state |
| P0 before physical actuation | Hardware activates implicitly; safety artifact/class mapping not verified | R2–R3, static code evidence |
| P1 before EfficientDet retraining | 6,675 processed image files absent from Git; missing input silently becomes blank samples | DATA-01 |
| P1 before relying on EfficientDet results | Best checkpoint selected by confidence; wrong-only predictions score 0.99 | M1, synthetic extracted-function probe |
| P1 before interpreting EfficientDet boxes | Prediction coordinates are transposed | M2, exact dependency contract + reproduced fixture |
| P1 before a reproducible release | Unlocked dependencies, unrestricted artifact provenance and version-dependent loading risk | M4; installed-device CVE status unknown |
| P1 data hardening | HTTP downloads and unconfined manifest paths | S1, attacker-controlled input prerequisite |
| P2 training/data quality | Negative review gaps, whole-frame targets, silently discarded labels, ineffective resume option | DATA-02–04, M3 |
| P2 unattended operation | Recorder failure/shutdown and unbounded storage/logging | R4–R5 |
| P3 maintainability | Clip timing, stale configs, best-score metadata, rebuild hygiene and documentation | R6, DATA-05, maintenance observations |

Priority is remediation order, not a claim that every item is remotely exploitable. Runtime severity depends on actual connected hardware; no hardware deployment was inspected.

## What existing metrics do and do not show

The committed `models/goose_yolo/results.csv` contains **76 epochs**. Its highest validation mAP50–95 is at **epoch 61**: precision **0.89426**, recall **0.82607**, mAP50 **0.89808**, mAP50–95 **0.71481**. Final epoch 76 reports mAP50–95 **0.64660**. These values were read from the historical CSV, not reproduced. They describe goose validation, not person recall, final test performance, field robustness or actuator safety. The CSV alone cannot establish which checkpoint was deployed or cryptographically bind its results to current data.

The audit checked **7,115 YOLO images** across train/val/test (5,700 / 707 / 708) and found **zero exact-content cross-split overlaps**, malformed labels, out-of-bounds boxes or missing label partners. There are **28 whole-frame box review candidates** and approximately 70% negative images. A full dataset table and caveats are in the data report. Near-duplicate/scene leakage and semantic labels need additional review.

## Recommended sequence and acceptance criteria

1. **Make actuation fail closed.** Require explicit arming and verified safety-model metadata, expire decisions independently of the frame loop, and validate physical default-off/watchdog behavior. Acceptance: dropped/stalled frames, stalled inference, failed safety checks and process failure cannot leave an indefinite enable permission. Test hardware behavior separately from mocks.
2. **Repair reproducibility and model correctness.** Enforce complete image manifests; implement IoU-based evaluation; fix coordinate order and genuine resume; fail visibly on augmentation/weight incompatibility. Acceptance: an all-false-positive synthetic detector has zero AP and missing inputs abort before training.
3. **Review labels and preserve data lineage.** Review the 28 whole-frame boxes and negative source candidates, use approved category mappings and source/scene-group splits, rebuild atomically, and preserve hashes. Acceptance: integrity checks pass with a documented semantic-review record.
4. **Publish a versioned evaluation run.** Evaluate the exact goose checkpoint on untouched test images and field videos; separately evaluate the person detector. Report AP, precision/recall, false alarms per camera-hour, missed events, person misses with confidence intervals, distance/light/weather slices and p95/p99 latency on Jetson. Thresholds should follow the intended operating conditions; no acceptable safety miss rate is assumed here.
5. **Harden packaging and unattended operations.** Lock a supported Jetson environment, verify artifact hashes/provenance, scan actual resolved versions, add bounded recording/storage and meaningful automated regression checks. Record deployment configuration and rollback procedure.

## Reports and reproduction

- [Dataset findings](data-findings.md), [machine-readable data evidence](data-evidence.json)
- [Runtime and deployment findings](runtime_audit.md), [runtime probe output](runtime_probe_output.txt)
- [Model and software trust findings](model-security-findings.md), [model probe evidence](model-evidence.json)
- [Live YOLO detection model addendum](detection-model-addendum.md) (added 2026-10-22, D1–D10)
- [Checkpoint inventory](checkpoint-inventory.json), [effdet contract reference](effdet-upstream-reference.json), [limited credential scan](security-evidence.json)

From the repository root:

```sh
python3 reports/audit-2026-10-22/data_probe.py
python3 reports/audit-2026-10-22/runtime_probes.py
python3 reports/audit-2026-10-22/model_probe.py
python3 reports/audit-2026-10-22/security_probe.py
```

Data/runtime/security probes use Python standard-library techniques; the model probe additionally requires torch and supplies synthetic tensors. The probes read source/data from Git HEAD, so rerunning after source changes audits that new HEAD; compare commit identity before interpreting results. They do not import the application, deserialize model weights, access GPIO/camera, install services or retrain. The effdet wheel was downloaded solely for ZIP source inspection, not installed. Checkpoint inspection used ZIP/pickle opcode parsing, not `torch.load`.

**Limitations:** no actual checkpoint inference benchmark, Jetson access, deployed dependency inventory, adversarial-image testing, physical safety certification, exhaustive secret scan, image-decode sweep, visual annotation review, or full near-duplicate analysis. Official dependency advisories were checked, but no comprehensive resolved-environment vulnerability scan was possible without the deployment environment. No finding establishes a malicious artifact or an exploited system.
