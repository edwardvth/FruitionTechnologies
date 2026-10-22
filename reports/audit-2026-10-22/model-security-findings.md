# Model evaluation and software trust audit

Scope: `edwardvth/FruitionTechnologies`, commit `b44d6e69708eff6057818973a453b18e1dfd5db5`, 2026-10-22. Source locations below refer to that commit. These findings complement the data and runtime reports. Severity reflects impact and required preconditions, not confirmed exploitation.

## M1 — High, confirmed: EfficientDet selects checkpoints by confidence, not correctness

`goose_detection/scripts/train.py:224-247,301-315` averages prediction confidence above 0.3 and saves a best checkpoint when that mean increases. Although targets are passed to the model, this metric does not compare predictions to labels, match boxes by IoU, penalize missed objects, or count false positives. An overconfident wrong detector can beat an accurate calibrated one.

`model_probe.py` extracts the actual validation function and supplies a single 0.99-confidence box with zero overlap with the ground truth: the returned validation score is 0.99. No real checkpoint is involved. Replace checkpoint selection with a ground-truth evaluator (COCO AP plus deployment-relevant precision/recall), record all configuration/data/artifact hashes, and reserve a final held-out test set. Synthetic false positives and empty detections should fail meaningful evaluation regression tests.

## M2 — Medium, confirmed: EfficientDet visualization swaps X and Y

`goose_detection/scripts/test_model.py:86-98` interprets predicted coordinates as YXYX. The exact pinned dependency, effdet 0.4.1, emits XYXY: its wheel's `effdet/anchors.py:128-130` explicitly decodes with `output_xyxy=True`. The versioned artifact URL and SHA-256 are recorded in `effdet-upstream-reference.json`; the [upstream repository](https://github.com/rwightman/efficientdet-pytorch) also documents this contract.

The extracted function probe inputs XYXY `[10,20,110,220]` with horizontal scale 2 and vertical scale 1. Correct output is `[20,20,220,220]`; actual output is `[40,10,440,110]`. Correct unpacking and add an asymmetric, non-square fixture. This affects the EfficientDet inspection script; the active YOLO runtime uses a different path.

## M3 — Medium, confirmed: YOLO resume flag starts from weights without requesting trainer resume

`goose_detection/scripts/train_yolo.py:44-64` loads `last.pt` for `--resume`, but calls `model.train(...)` without `resume=True`. This does not explicitly restore the interrupted optimizer, scheduler and epoch progression. The [Ultralytics resume documentation](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings) requires `model.train(resume=True)` for that behavior. Pass the supported resume argument and verify restored epoch/optimizer metadata in a short interruption test. `exist_ok=True` also reuses the same run directory, obscuring experiment history; create uniquely identified runs unless deliberately resuming.

## M4 — Medium: artifact and dependency trust are not enforced

`requirements.txt:1-24` leaves torch/torchvision to a separate unversioned installation command and mostly uses open lower bounds, including `ultralytics>=8.3.0`. There is no lockfile, deployed environment inventory, artifact signature/checksum verification, or tested Jetson dependency matrix. `test_model.py:52` and `train.py:286` call `torch.load` without an explicit restricted-loading policy. `test_model.py:61` also ignores missing/unexpected state keys through `strict=False`, permitting incomplete checkpoint loading without inspection.

This is a conditional attack surface: an attacker must replace a model/dependency or persuade an operator to load it. No malicious checkpoint or installed vulnerable Jetson version was demonstrated. Modern torch defaults differ from older releases, so the bare call is not universally proof of unrestricted pickle execution. [PyTorch documents the default change in 2.6 and restricted loader limitations](https://docs.pytorch.org/docs/stable/notes/serialization). Official advisories identify [CVE-2025-32434 affecting <=2.5.1](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6), fixed in 2.6.0, and [GHSA-63cw-57p8-fm3p affecting <=2.9.1](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p), patched in >=2.10.0. These are version-conditional exposures allowed by the unspecified torch policy, not an inventory-based CVE finding. Check all advisories for the exact chosen build; merely moving to the minimum patch cited here is not a complete dependency audit.

The checkpoint inventory used ZIP parsing and `pickletools`, never unpickling. It records three artifacts' hashes and referenced globals. The YOLO best checkpoint contains an `8.4.39` string and the person checkpoint an `8.3.222` string; these are metadata candidates, not proof of the deployed runtime version. YOLO artifacts reference Python model classes, so blindly forcing restricted loading may break compatibility. Verify trusted artifacts, use an isolated conversion/export workflow where necessary, deploy a reviewed inference format, explicitly restrict state-dict loading on a patched compatible torch, and assert all expected keys. Inventory and scan the actual Jetson environment before claiming dependency clearance.

## S1 — Medium: downloaded data and annotation paths cross unvalidated trust boundaries

`download_negatives.py:18-21` uses HTTP for both annotation and image resources with no digest verification. It trusts annotation `file_name` directly in the destination and URL (`80-89`); parent/absolute paths are not rejected. `merge_datasets.py:71-98` accepts existing absolute/CWD-relative image paths and copies them without proving they lie inside the intended source directory or verifying that they are images. A crafted dataset manifest can import an unintended readable local file into generated data. Exposure requires processing attacker-controlled/intercepted input; this is not a network-accessible server endpoint, and no exploit was run.

Use authenticated transport plus pinned manifests/checksums; validate schema, maximum sizes and actual image types; resolve paths and require containment in the selected root including symlink handling; use atomic temporary downloads and timeouts. Reject mixed/unrecognized categories: `merge_datasets.py:64-66,118-125` maps every category to goose, which can silently poison labels if a new multiclass source is introduced. Explicitly map approved source categories only.

## Lower-priority maintenance observations

- `train.py:307-313` constructs saved checkpoint metadata before updating `best_conf`, so the saved value lags the new best; restoring it can weaken best-checkpoint tracking. Store the updated best value before both saves.
- `train_yolo.py:64` hardcodes `device=0` despite a comment claiming automatic CPU fallback. Expose/select a valid device explicitly and test the supported hardware path.
- `configs/goose_config.yaml` describes a legacy TensorFlow/TFRecord/SGD/COCO-evaluation setup that the PyTorch trainer does not read. `generate_tfrecords.py` is also legacy. Mark these historical or connect actual configuration to code; current comments are not proof those settings or metrics apply.
- Roboflow API keys are accepted on the command line (`download_roboflow.py:18`), which can leave them in shell history/process arguments. Prefer an environment or protected credential source.
- A conservative regex scan of eight reachable commits / 31 unique non-data code/config blobs found no credential-shaped matches. `security_probe.py` records scope; this is not exhaustive secret detection. Media, datasets and checkpoint internals were excluded from this scan.
