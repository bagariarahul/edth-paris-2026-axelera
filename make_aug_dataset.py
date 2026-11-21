"""
make_aug_dataset.py - build dataset_aug/ from dataset/ to close the RGB->thermal gap.

Adds to TRAIN only (valid/test are copied unchanged so scores stay comparable):
  thermal images -> inverted copy + CLAHE copy
  rgb images     -> grayscale copy (pushes the model toward a shared, colour-free representation)
  rare tails     -> extra horizontally-flipped copies of RGB images containing `vehicle`
                    and thermal images containing `person`

Usage (from starter-kit/):
  python make_aug_dataset.py --src ..\\dataset --dst ..\\dataset_aug
  python train_rfdetr.py --dataset-dir ..\\dataset_aug --model nano --epochs 25 --output-dir runs\\c2_aug
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ANN = "_annotations.coco.json"


def clahe3(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def gray3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../dataset")
    ap.add_argument("--dst", default="../dataset_aug")
    ap.add_argument("--rare-repeat", type=int, default=2, help="flipped copies per rare-class image")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    for split in ("valid", "test"):
        if (src / split).exists():
            shutil.copytree(src / split, dst / split, dirs_exist_ok=True)

    coco = json.load(open(src / "train" / ANN, encoding="utf-8"))
    out_dir = dst / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_id = {c["name"]: c["id"] for c in coco["categories"]}
    vehicle_id, person_id = cat_id.get("vehicle"), cat_id.get("person")

    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    images, annotations = [], []
    next_img = max(i["id"] for i in coco["images"]) + 1
    next_ann = max((a["id"] for a in coco["annotations"]), default=0) + 1
    counts = {"orig": 0, "inv": 0, "clahe": 0, "gray": 0, "rare": 0, "skipped": 0}

    for img in coco["images"]:
        path = src / "train" / img["file_name"]
        shutil.copy2(path, out_dir / img["file_name"])
        images.append(img)
        anns = anns_by_img.get(img["id"], [])
        annotations.extend(anns)
        counts["orig"] += 1

        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            counts["skipped"] += 1
            continue
        W = im.shape[1]
        cats = {a["category_id"] for a in anns}
        mod = img.get("modality")

        variants = []  # (tag, image, flipped?)
        if mod == "thermal":
            variants += [("inv", 255 - im, False), ("clahe", clahe3(im), False)]
            if person_id in cats:
                variants += [(f"rare{k}", cv2.flip(im, 1), True) for k in range(args.rare_repeat)]
        elif mod == "rgb":
            variants += [("gray", gray3(im), False)]
            if vehicle_id in cats:
                variants += [(f"rare{k}", cv2.flip(im, 1), True) for k in range(args.rare_repeat)]

        stem = Path(img["file_name"]).stem
        for tag, vim, flipped in variants:
            fname = f"{stem}__{tag}.jpg"
            cv2.imwrite(str(out_dir / fname), vim, [cv2.IMWRITE_JPEG_QUALITY, 95])
            new_img = dict(img, id=next_img, file_name=fname)
            images.append(new_img)
            for a in anns:
                x, y, w, h = a["bbox"]
                bbox = [W - x - w, y, w, h] if flipped else [x, y, w, h]
                annotations.append(dict(a, id=next_ann, image_id=next_img, bbox=bbox))
                next_ann += 1
            next_img += 1
            counts["rare" if tag.startswith("rare") else tag] += 1

    coco["images"], coco["annotations"] = images, annotations
    json.dump(coco, open(out_dir / ANN, "w", encoding="utf-8"))
    print(f"train images: {len(images)}  annotations: {len(annotations)}  breakdown: {counts}")


if __name__ == "__main__":
    main()
