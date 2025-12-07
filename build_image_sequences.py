import os
import json
from pathlib import Path

import cv2
import torch
import numpy as np

DATA_ROOT = Path("..") / "data"
FRAMES_ROOT = DATA_ROOT / "frames"
SEQS_ROOT = DATA_ROOT / "sequences_img"
INDEX_PATH = DATA_ROOT / "videos_index.json"

SEQ_LEN = 30
IMG_SIZE = 64

SEQS_ROOT.mkdir(parents=True, exist_ok=True)


def build_sequences_for_video(video_id: str, label: int):
    video_frames_dir = FRAMES_ROOT / video_id
    if not video_frames_dir.is_dir():
        return 0

    frame_files = sorted(video_frames_dir.glob("frame_*.jpg"))
    if len(frame_files) < SEQ_LEN:
        return 0

    num_seqs = 0

    for start in range(0, len(frame_files) - SEQ_LEN + 1, SEQ_LEN):
        chunk = frame_files[start:start + SEQ_LEN]
        imgs = []

        for fp in chunk:
            img = cv2.imread(str(fp))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = img[None, :, :]
            imgs.append(img)

        if len(imgs) != SEQ_LEN:
            continue

        imgs_np = np.stack(imgs, axis=0).astype("float32")
        seq_tensor = torch.from_numpy(imgs_np)  

        out_name = f"{video_id}_seq{num_seqs:04d}.pt"
        out_path = SEQS_ROOT / out_name

        torch.save(
            {
                "video": seq_tensor,
                "label": int(label), 
                "video_id": video_id,
            },
            out_path
        )

        num_seqs += 1

    return num_seqs


def main():
    if not INDEX_PATH.is_file():
        return

    with open(INDEX_PATH, "r") as f:
        index = json.load(f)
    total_seqs = 0

    for entry in index:
        video_id = entry["video_id"] 
        label = entry["label"]       

        print(f"Gerando sequências para {video_id} (label={label}) ...")
        n = build_sequences_for_video(video_id, label)
        print(f"-> {n} sequências geradas.")
        total_seqs += n



if __name__ == "__main__":
    main()
