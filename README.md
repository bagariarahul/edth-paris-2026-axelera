# EDTH Paris 2026 — Axelera Challenge: RGB + Thermal Detection & Tracking

Team submission for the Axelera challenge *"Day RGB + Thermal IR ISR Detection & Multi-Stream Tracking"*
(listed as challenge 07 on the EDTH site, "challenge-2" inside the data pack).

---

## 1. The problem in plain words

A surveillance drone carries two cameras:

- a normal **daylight camera (RGB)**: sees colours, like your phone;
- a **thermal camera (IR)**: sees heat, so it works at night, but the picture is grey blobs.

We have to build **one** AI model that, on both kinds of image, finds three things:
`armored_vehicle`, `vehicle` (trucks, pickups) and `person`. Then it has to **follow each object
across video frames** and give it a stable ID ("this is tank #3"). The model also has to run on a
small **Axelera Metis** AI chip plugged into a Raspberry Pi, not on a big GPU.

The hard part is that most AI models learn from normal photos, so they are good at RGB and bad at
thermal. The organisers call this "modality collapse".

## 2. How we are scored

| Weight | What | Measured by |
|---|---|---|
| 45% | Detection on **both** camera types, kept in balance | `rgb_mAP50` and `thermal_mAP50`, reported separately |
| 35% | Tracking across video | `MOTA`, `IDF1`, `track_continuity` |
| 20% | Running on the Metis chip | compiled model, INT8 accuracy kept, speed, memory |

Final scores come from a **hidden test set** the organisers keep. Everything we measured ourselves
is on the public `valid` split.

## 3. What we built: three parts

### Part 1: the detector (the "eyes")
- **Model:** RF-DETR Nano at 384×384 pixels, the model the organisers recommend.
- **Training:** 25 planned epochs (one epoch = one pass over all 712 training images) on a desktop
  CPU (Intel i5-14600K, no usable GPU), about 10 minutes per epoch.
- **Best model:** epoch 12. After that the model started memorising the small training set
  ("overfitting") and got worse on new images. The trainer keeps the best epoch automatically
  (`checkpoint_best_ema.pth`).
- **Export:** converted to **ONNX**, a standard file format that other tools, including the Axelera
  compiler, can read.

### Part 2: the tracker (the "memory"): `track_bytetrack.py`
The organisers' baseline tracker forgets an object the moment it misses one frame, so every flicker
creates a brand-new ID. Ours fixes that:
- **Track memory:** a lost object is remembered for up to 15 frames (half a second).
- **ByteTrack trick:** detections the model is unsure about (low confidence) can't start a new track,
  but they *can* continue an existing one. That recovers objects the detector half-sees.
- **Motion prediction:** remembers each object's speed and direction, so it can be found again after a gap.
- **Confirmation:** a track only counts after 3 hits in a row, which removes one-frame false alarms.
- **Gap filling:** if an object disappears and comes back, the missing frames are filled in.
- **Auto-tuning (`--sweep`):** tries 108 setting combinations and keeps the best. All top-5
  results agreed on the key settings, so the result is not a lucky fluke.

### Part 3: the edge chip (the "body"): in progress
- Hardware: Raspberry Pi 5 (8 GB) + db-tronic NVMe HAT + Axelera Metis M.2 card.
- Status: the Pi **does not see the Metis card**. `dmesg` reports
  `brcm-pcie 1000110000.pcie: link down`.
  - Already tried: `dtparam=pciex1` under `[all]`, PCIe gen 3 and gen 2, two different ribbon cables.
  - Not tried: the HAT's 2-pin 5V auxiliary power input (no cable for it). The Metis draws 3.5–9 W.
- Ready for compilation once the card is visible: full and backbone ONNX exports,
  `preprocess.py` (the calibration transform the compiler requires) and the 200-image calibration
  set (100 RGB + 100 thermal).

## 4. Results (public `valid` split)

| Metric | Organisers' baseline | **Ours** | Change |
|---|---|---|---|
| mAP50 (all) | 0.349 | **0.414** | +19% |
| mAP50:95 | 0.249 | **0.324** | +30% |
| RGB mAP50 | 0.526 | **0.547** | +4% |
| Thermal mAP50 | 0.180 | **0.157** | −13% ⚠️ |
| MOTA | −0.076 | **0.290** | from negative to positive |
| IDF1 | 0.235 | **0.511** | ×2.2 |
| Track continuity | 0.206 | **0.337** | +64% |
| ID switches | 9 | **0** | |

How the tracking gain splits up (same streams):

| Step | MOTA | IDF1 |
|---|---|---|
| Organisers' baseline | −0.076 | 0.235 |
| Our detector (epoch 9) + baseline tracker | 0.143 | 0.314 |
| + our ByteTrack tracker | 0.236 | 0.474 |
| Our best detector (epoch 12) + our tracker | **0.290** | **0.511** |

**Honest weak spot:** thermal detection did not improve. Longer training made the model better on
RGB and slightly worse on thermal, which is exactly the "modality collapse" the brief warns about.
`make_aug_dataset.py` is the prepared fix (see section 7).

## 5. What is in this repo

| File / folder | What it is |
|---|---|
| `track_bytetrack.py` | **Ours.** The improved tracker, with a settings sweep. |
| `make_aug_dataset.py` | **Ours.** Builds a thermal-focused training set (prepared, not yet trained on). |
| `preprocess.py` | **Ours.** Calibration transform for the Axelera compiler (`axcompile --transform`). |
| `best_tracker_params.json` | The tracker settings the sweep picked. |
| `c2_preds_test.json` | **Submission:** detections on the hidden **test** images (COCO format). |
| `c2_preds_valid.json` | Detections on the `valid` images (for our own scoring). |
| `pred_mot_final/` | **Submission:** tracking results for all 8 video streams (MOTChallenge format). |
| `train_rfdetr.py`, `export_rfdetr_axelera.py`, `predict_onnx_baseline.py`, `evaluate_submissions.py` | Organisers' starter-kit scripts. |

**Big files** (too large for normal git) are attached to the GitHub **Release** instead:
`rfdetr_nano_full.onnx` (complete detector), `rfdetr_nano_backbone.onnx` (backbone only, for the
chip) and `checkpoint_best_ema.pth` (trained weights).

Not in the repo: the dataset, calibration images and videos (organisers' data pack) and the Python
environment.

## 6. How to reproduce

Run everything from the starter-kit folder, inside the Python environment, with `ffmpeg` installed.

```powershell
# train (~10 min/epoch on CPU)
python train_rfdetr.py --dataset-dir ..\dataset --model nano --epochs 25 --output-dir runs\c2_nano

# export the best epoch to ONNX
python export_rfdetr_axelera.py --model nano --dataset-dir ..\dataset --weights runs\c2_nano\checkpoint_best_ema.pth --calibration-dir ..\calibration

# score detection (prints rgb_mAP50 and thermal_mAP50)
python predict_onnx_baseline.py detect --onnx exports\rfdetr\full\inference_model.onnx --dataset-dir ..\dataset --split valid --out c2_preds_valid.json
python evaluate_submissions.py detect --gt ..\dataset\valid\_annotations.coco.json --preds c2_preds_valid.json

# submission detections on test
python predict_onnx_baseline.py detect --onnx exports\rfdetr\full\inference_model.onnx --dataset-dir ..\dataset --split test --out c2_preds_test.json

# tracking: tune on valid, write all 8 streams
python track_bytetrack.py --onnx exports\rfdetr\full\inference_model.onnx --eval-dir ..\streams_eval --sweep --out-dir pred_mot_final
python evaluate_submissions.py mot --gt-dir ..\streams_eval --pred-dir pred_mot_final --split valid
```

Note: `track_bytetrack.py` saves detections in `cache_dets/`. **Delete that folder whenever you
change the model**, or it will reuse the old model's detections.

## 7. Next steps

1. **Metis bring-up (with the Axelera reps):** fix the PCIe `link down` (most likely the HAT's
   5V aux power), install the Voyager SDK, then compile:
   ```bash
   axcompile -i rfdetr_nano_full.onnx --imageset calibration --transform preprocess.py --input-shape 1,3,384,384 -o compiled_full/
   ```
   If the full model won't compile (transformer layers), compile `rfdetr_nano_backbone.onnx`
   instead, and run the rest of the model on the Pi's CPU.
2. **Fix thermal:** `python make_aug_dataset.py --src ..\dataset --dst ..\dataset_aug` adds inverted
   and contrast-boosted thermal copies, grayscale RGB copies and extra copies of rare cases. Then
   fine-tune from the current best weights.
3. **Pitch line:** "We doubled tracking quality (IDF1 0.235 → 0.511) and removed every ID switch,
   improved overall detection by 19%, and identified thermal collapse as the open problem, with
   the fix already built."

## 8. Glossary

- **mAP50:** detection accuracy from 0 to 1. A detection counts as correct if its box overlaps the
  real object by at least 50%. Higher is better.
- **mAP50:95:** a stricter version that also rewards tight boxes.
- **MOTA:** overall tracking accuracy. It penalises missed objects, false alarms and ID mix-ups, and
  can be negative if the tracker does more harm than good. Higher is better.
- **IDF1:** how consistently each real object keeps the same ID over time. Higher is better.
- **ID switch:** the tracker suddenly gives an object a different ID. Lower is better.
- **Epoch:** one full pass of training over the training images.
- **Overfitting:** the model memorises the training images instead of learning general patterns,
  so it gets worse on new images.
- **ONNX:** a standard file format for AI models, readable by many tools.
- **INT8 / quantization:** shrinking the model's numbers from 32-bit to 8-bit so it runs fast on the
  chip. The **calibration set** is used to do this without losing much accuracy.
- **EMA:** a smoothed average of the model's weights during training; usually a bit more accurate.
