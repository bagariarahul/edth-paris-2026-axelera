"""
track_bytetrack.py - ByteTrack-style tracker for Challenge 2 MOT streams.

Fixes the baseline IoU tracker's main weakness (a track dies the first frame it is not detected):
  * track memory: lost tracks survive up to --max-age frames
  * ByteTrack two-stage association: low-confidence detections may only CONTINUE existing tracks
  * constant-velocity motion prediction so a track can be re-found after a gap
  * --min-hits confirmation (kills one-frame false-positive tracks) with back-fill of the
    confirmation frames, so recall is not lost
  * offline gap interpolation (--interp-gap) inside each track
  * class-agnostic association (the official MOT scorer is class-agnostic)

Reuses the starter kit's own preprocessing / RF-DETR decoding, so boxes match predict_onnx_baseline.py.

Usage (from starter-kit/, venv active, ffmpeg on PATH):
  # 1. tune on the 4 valid streams (detections are cached, so re-runs take seconds)
  python track_bytetrack.py --onnx exports_interim\\rfdetr\\full\\inference_model.onnx --eval-dir ..\\streams_eval --sweep --out-dir pred_mot_bt
  # 2. or run one fixed configuration
  python track_bytetrack.py --onnx <model.onnx> --eval-dir ..\\streams_eval --out-dir pred_mot_bt --high-thr 0.4 --min-hits 2
Then score:  python evaluate_submissions.py mot --gt-dir ..\\streams_eval --pred-dir pred_mot_bt --split valid
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------- detection cache


def build_cache(onnx_path: Path, eval_dir: Path, cache_dir: Path, img_size: int, num_classes: int,
                cache_thr: float, top_k: int) -> None:
    """Run the detector once per stream and store all detections >= cache_thr."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in eval_dir.glob("seq_*") if (p / "video.mp4").exists())
    todo = [s for s in seqs if not (cache_dir / f"{s.name}.json").exists()]
    if not todo:
        return
    import onnxruntime as ort
    from PIL import Image

    from predict_onnx_baseline import _decode_rfdetr_topk, _nms_dets, _preprocess_imagenet

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    for seq_dir in todo:
        target = cache_dir / f"{seq_dir.name}.json"
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(seq_dir / "video.mp4"),
                            "-fps_mode", "passthrough", str(Path(td) / "f_%06d.jpg")], check=True)
            frames = []
            for f1, fpath in enumerate(sorted(Path(td).glob("f_*.jpg")), 1):
                with Image.open(fpath) as im:
                    w_px, h_px = im.size
                    inp = _preprocess_imagenet(im, img_size)
                outs = dict(zip(out_names, sess.run(out_names, {in_name: inp}), strict=True))
                dets = _nms_dets(_decode_rfdetr_topk(outs["dets"][0], outs["labels"][0], w_px, h_px,
                                                     num_classes, top_k=top_k, score_thr=cache_thr),
                                 iou_thr=0.5)
                frames.append({"f": f1, "d": [[c, s, *b] for c, s, b in dets]})
        target.write_text(json.dumps({"num_frames": len(frames), "frames": frames}), encoding="utf-8")
        print(f"cached {seq_dir.name}: {len(frames)} frames")


# ----------------------------------------------------------------- tracker


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0:1] + a[:, 2:3], a[:, 1:2] + a[:, 3:4]
    bx2, by2 = (b[:, 0] + b[:, 2])[None], (b[:, 1] + b[:, 3])[None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0:1], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1:2], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


class Track:
    def __init__(self, tid: int, box: np.ndarray, score: float, cls: int, frame: int):
        self.id = tid
        self.box = box.astype(np.float64)
        self.vel = np.zeros(4)
        self.hits = 1
        self.last_frame = frame
        self.cls_votes: dict[int, float] = {cls: score}
        self.history: list[tuple[int, np.ndarray, float]] = [(frame, self.box.copy(), score)]

    def predicted(self, frame: int) -> np.ndarray:
        p = self.box + self.vel * (frame - self.last_frame)
        p[2:] = np.maximum(p[2:], 1.0)
        return p

    def update(self, box: np.ndarray, score: float, cls: int, frame: int) -> None:
        dt = max(1, frame - self.last_frame)
        self.vel = 0.6 * self.vel + 0.4 * (box - self.box) / dt
        self.box = box.astype(np.float64)
        self.last_frame = frame
        self.hits += 1
        self.cls_votes[cls] = self.cls_votes.get(cls, 0.0) + score
        self.history.append((frame, self.box.copy(), score))

    @property
    def cls(self) -> int:
        return max(self.cls_votes, key=self.cls_votes.get)


def _match(tracks: list[Track], dets: list, frame: int, iou_thr: float):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    tb = np.stack([t.predicted(frame) for t in tracks])
    db = np.asarray([d[2:] for d in dets], dtype=np.float64)
    ious = iou_xywh(tb, db)
    ri, ci = linear_sum_assignment(np.where(ious >= iou_thr, 1.0 - ious, 1e6))
    pairs = [(r, c) for r, c in zip(ri, ci) if ious[r, c] >= iou_thr]
    mt, md = {r for r, _ in pairs}, {c for _, c in pairs}
    return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]


def run_tracker(seq: dict, p: dict) -> list[str]:
    tracks: list[Track] = []
    finished: list[Track] = []
    next_id = 1
    for fr in seq["frames"]:
        f = fr["f"]
        dets = fr["d"]
        high = [d for d in dets if d[1] >= p["high_thr"]]
        low = [d for d in dets if p["low_thr"] <= d[1] < p["high_thr"]]

        # stage 1: every live track vs high-confidence detections
        pairs, un_t, un_d = _match(tracks, high, f, p["match_iou"])
        for r, c in pairs:
            d = high[c]
            tracks[r].update(np.asarray(d[2:]), d[1], d[0], f)
        # stage 2: leftover CONFIRMED tracks vs low-confidence detections (ByteTrack)
        rest = [tracks[i] for i in un_t if tracks[i].hits >= p["min_hits"]]
        pairs2, _, _ = _match(rest, low, f, p["low_iou"])
        for r, c in pairs2:
            d = low[c]
            rest[r].update(np.asarray(d[2:]), d[1], d[0], f)
        # new tracks only from unmatched high-confidence detections
        for j in un_d:
            d = high[j]
            if d[1] >= p["new_thr"]:
                tracks.append(Track(next_id, np.asarray(d[2:]), d[1], d[0], f))
                next_id += 1
        # retire stale tracks (unconfirmed tracks get no grace period)
        alive = []
        for t in tracks:
            age = f - t.last_frame
            if (t.hits < p["min_hits"] and age > 0) or age > p["max_age"]:
                finished.append(t)
            else:
                alive.append(t)
        tracks = alive
    finished.extend(tracks)

    lines = []
    out_id = 1
    n_frames = seq["num_frames"]
    for t in finished:
        if t.hits < p["min_hits"]:
            continue  # never confirmed -> treated as a false positive
        rows = list(t.history)
        if p["interp_gap"] > 0:
            filled = []
            for (f0, b0, s0), (f1, b1, s1) in zip(rows, rows[1:]):
                filled.append((f0, b0, s0))
                gap = f1 - f0
                if 1 < gap <= p["interp_gap"]:
                    for k in range(1, gap):
                        a = k / gap
                        filled.append((f0 + k, b0 * (1 - a) + b1 * a, min(s0, s1)))
            filled.append(rows[-1])
            rows = filled
        cls1 = t.cls + 1
        for f, b, s in rows:
            if not (1 <= f <= n_frames) or b[2] <= 0 or b[3] <= 0:
                continue
            lines.append((f, out_id, f"{f},{out_id},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},{s:.4f},{cls1},1.0"))
        out_id += 1
    lines.sort()
    return [l[2] for l in lines]


def write_preds(cache_dir: Path, out_dir: Path, p: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cf in sorted(cache_dir.glob("seq_*.json")):
        seq = json.loads(cf.read_text(encoding="utf-8"))
        lines = run_tracker(seq, p)
        (out_dir / f"{cf.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ----------------------------------------------------------------- main

DEFAULTS = dict(high_thr=0.4, low_thr=0.1, new_thr=0.4, match_iou=0.25, low_iou=0.4,
                max_age=30, min_hits=2, interp_gap=10)

GRID = dict(high_thr=[0.2, 0.25, 0.3], match_iou=[0.1, 0.15, 0.2], max_age=[15, 30],
            min_hits=[2, 3], interp_gap=[20, 30, 45])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--cache-thr", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--sweep", action="store_true", help="grid-search on the valid split, write the best")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    cache_dir = a.cache_dir or Path("cache_dets") / (a.onnx.parent.parent.parent.name + "_" + a.onnx.stem)
    build_cache(a.onnx, a.eval_dir, cache_dir, a.img_size, a.num_classes, a.cache_thr, a.top_k)

    params = {k: getattr(a, k) for k in DEFAULTS}
    if a.sweep:
        from evaluate_submissions import evaluate_mot_tracking

        keys = list(GRID)
        results = []
        tmp = Path(tempfile.mkdtemp(prefix="bt_sweep_"))
        for combo in itertools.product(*(GRID[k] for k in keys)):
            p = dict(params, **dict(zip(keys, combo)))
            p["new_thr"] = p["high_thr"]
            write_preds(cache_dir, tmp, p)
            m = evaluate_mot_tracking(a.eval_dir, tmp, split="valid")
            results.append((m["MOTA"] + m["IDF1"], m, p))
        shutil.rmtree(tmp, ignore_errors=True)
        results.sort(key=lambda r: -r[0])
        print("top 5 (ranked by MOTA + IDF1):")
        for _, m, p in results[:5]:
            print(json.dumps({k: m[k] for k in ("MOTA", "IDF1", "track_continuity", "id_switches")}),
                  json.dumps({k: p[k] for k in keys}))
        params = results[0][2]
        Path("best_tracker_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        print("best params saved to best_tracker_params.json")

    write_preds(cache_dir, a.out_dir, params)
    print(f"wrote {a.out_dir} with params {json.dumps(params)}")


if __name__ == "__main__":
    main()