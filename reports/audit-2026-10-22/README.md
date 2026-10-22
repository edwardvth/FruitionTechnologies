# Fruition Technologies repository and model audit

**Audit date:** 2026-10-22. **Repository:** [edwardvth/FruitionTechnologies](https://github.com/edwardvth/FruitionTechnologies). **Reviewed commit:** `b44d6e69708eff6057818973a453b18e1dfd5db5` (`main` at inspection). Three parallel reviewers examined data, runtime/deployment, and model evaluation/software trust. Application code and weights were not changed.

The largest confirmed concerns are stale actuator commands after camera failures, weak safety-model/configuration validation, and misleading EfficientDet evaluation. The YOLO dataset passed exact duplicate and label-geometry checks, and its committed training log contains useful validation metrics. These checks do not establish safe physical deployment or held-out field accuracy.

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

Training has two separate branches: COCO manifests → `train.py` → EfficientDet `.pth` → `test_model.py`; YOLO labels → `train_yolo.py` → `models/goose_yolo/weights/best.pt` → `test_yolo.py` / live inference. The live pipeline targets NVIDIA Jetson and also loads `yolo26n.pt` as a person-safety model. Changes to EfficientDet do not automatically repair the YOLO runtime. Eight commits were reachable. No tracked automated CI workflow, unit-test suite, dependency lockfile or main README was present; `test_model.py` and `test_yolo.py` generate visual examples, not automated quantitative regression tests.

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
