# Dataset audit — 2026-10-22

Scope: committed tree `b44d6e69708eff6057818973a453b18e1dfd5db5`. Reproduce with `python3 reports/audit-2026-10-22/data_probe.py`. The probe reads Git blobs and JSON/text only, so checkout timing does not affect results. It does not execute training or deserialize checkpoints. Machine-readable counts are in `data-evidence.json`.

## Measured data inventory

| Dataset | Split | Images | Positive images | Negative images | Boxes | Whole-frame boxes |
|---|---|---:|---:|---:|---:|---:|
| Processed COCO | train | 5,340 | 1,358 | 3,982 | 1,597 | 22 |
| Processed COCO | val | 667 | 145 | 522 | 152 | 1 |
| Processed COCO | test | 668 | 172 | 496 | 206 | 5 |
| YOLO | train | 5,700 | 1,718 | 3,982 | 2,412 | 22 |
| YOLO | val | 707 | 185 | 522 | 215 | 1 |
| YOLO | test | 708 | 212 | 496 | 332 | 5 |

The YOLO additions are 360/40/40 Gabe images. Negatives are 69.9%/73.8%/70.1% of YOLO train/val/test. This is a property of the dataset, not automatically a defect; deployment-specific false-positive and recall measurement is needed.

**Checks that passed:** No exact image-content overlap across any pair of YOLO splits (Git image blob identity); no missing YOLO label partner, malformed/nonfinite label, unexpected class, nonpositive box dimension, box edge outside the image, or exact duplicated label row. Processed COCO annotations have no orphan image reference, nonpositive/nonfinite box, out-of-image box, or duplicated class/box annotation. Exact identity checks do not rule out resized/cropped near-duplicates, adjacent video frames, or common-source leakage. Image decode correctness and semantic label correctness were not tested.

## Findings

### DATA-01 — High: clean-clone EfficientDet training silently consumes placeholder images

All **6,675** processed annotation records reference image basenames whose `data/processed/images/` file is absent from the committed tree. That directory is explicitly ignored (`goose_detection/.gitignore:3`). The corresponding YOLO images are tracked, but `scripts/train.py:57-65` checks only the annotation path and processed-images directory, then returns `_empty_sample()` for missing/unreadable files. Thus an ordinary clean-clone EfficientDet training invocation without data reconstruction uses black placeholder images with ignored targets instead of the intended data, while appearing to train. `scripts/convert_coco_to_yolo.py:57-61` silently skips these same missing sources.

Recommendation: resolve a versioned data manifest before starting either trainer, fail if any required source is missing, and explicitly reconstruct processed images from a verified source (the tracked YOLO split images could provide the current basenames). Add a preflight count/hash check and document the reconstruction command. This finding does not imply the existing saved model was trained on placeholders: its original training environment may have contained the ignored directory.

### DATA-02 — Medium: negative dataset creation does not enforce its stated filtering

`create_negatives_annotations.py:4` promises bird filtering, but lines 20-39 scan every JPG and lines 47-52 assign zero annotations to all of them without reading COCO instance labels. The committed negative manifest contains **5,000 images and zero annotations**, and the processed split negative counts sum to exactly 5,000. This leaves semantic correctness unverified and can assign background targets to geese present in source imagery. We did not visually review the 5,000 images and do not claim a confirmed count of false-negative labels. Bird images without geese are legitimate hard negatives and should be retained after species review.

Recommendation: review source bird-labelled images for actual geese, annotate any geese, retain reviewed non-goose birds as hard negatives, and store source IDs plus review provenance. Make filtering/review requirements executable instead of relying on the docstring.

### DATA-03 — Medium: classification-to-detection conversion fabricates localization targets

`convert_images_cv_to_coco.py:5-7` states that whole-frame boxes are used because the source has no boxes; lines 82-90 construct `[0,0,width,height]`. The raw images.cv manifest has 1,532 images/annotations. The committed processed and YOLO splits contain **28 whole-frame boxes total**, including **six validation/test boxes**. A frame-filling goose may legitimately require such a box, so these 28 are review candidates, not 28 proven annotation errors. The current processed manifests do not preserve sufficient source provenance to assert that every raw whole-frame box flowed into this snapshot unchanged.

Recommendation: localize each goose with actual object boxes, review all whole-frame labels, record annotation provenance, and reserve classification-only data for a separate image-presence objective. Otherwise detector box regression and IoU metrics can reward scene-sized boxes rather than goose localization.

### DATA-04 — Medium: augmentation failures silently discard positive targets

`train.py:81-89` catches every transform exception, resizes the image, and deletes all boxes/labels. A positive image can therefore be presented without its positive targets and no error counter. Frequency was not measured because model/training dependencies were not run.

Recommendation: validate inputs before training; fail with image ID and transform details when augmentation breaks. If recovery is required, use a deterministic resize that also transforms boxes and emit a counted warning with a strict failure threshold.

### DATA-05 — Low: rebuild pipeline lacks defenses against future duplicates and stale data

`merge_datasets.py:91-95` maps duplicate images to an existing image ID, but lines 115-132 append their annotations without checking duplicate boxes or reconciling conflicting annotations. `convert_coco_to_yolo.py:89-109` then adds Gabe images without cross-dataset content deduplication; a missing source label becomes an empty negative label. Existing copied images/labels are not consistently refreshed, and output directories are not cleared when the split changes. The current snapshot passed exact duplicate/label checks, so these are reproducible code risks for subsequent rebuilds, not observed current split leakage.

Recommendation: rebuild into a fresh staged directory from a single manifest, deduplicate image identities across every source before splitting, reconcile labels, reject absent labels unless explicitly declared negatives, and validate output hashes/counts before publishing. Group splits by scene/capture/source where available and add perceptual duplicate review.

## Suggested validation after corrections

Rebuild the data with immutable provenance and a held-out field-video test set, then retrain and evaluate using precision/recall, IoU-based AP, false alarms per camera-hour, missed-goose events, and slices for small/distant birds, darkness/weather, non-goose birds, and crowded scenes. No deployment robustness conclusion can be drawn from annotation syntax and exact-hash checks alone.
