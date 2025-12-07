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
IMG_SIZE = 64   # 64x64

SEQS_ROOT.mkdir(parents=True, exist_ok=True)


def build_sequences_for_video(video_id: str, label: int):
    """
    Lê os frames de frames/<video_id> e gera janelas de SEQ_LEN,
    salvando em sequences_img/ como tensores (SEQ_LEN, 1, 64, 64).
    """
    video_frames_dir = FRAMES_ROOT / video_id
    if not video_frames_dir.is_dir():
        print(f"[AVISO] Pasta de frames não encontrada: {video_frames_dir}")
        return 0

    frame_files = sorted(video_frames_dir.glob("frame_*.jpg"))
    if len(frame_files) < SEQ_LEN:
        print(f"[AVISO] {video_id} tem só {len(frame_files)} frames (< {SEQ_LEN}), ignorando.")
        return 0

    num_seqs = 0

    # aqui andamos de SEQ_LEN em SEQ_LEN; se quiser overlap, mude o passo
    for start in range(0, len(frame_files) - SEQ_LEN + 1, SEQ_LEN):
        chunk = frame_files[start:start + SEQ_LEN]
        imgs = []

        for fp in chunk:
            img = cv2.imread(str(fp))
            if img is None:
                continue

            # BGR -> Gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # resize para 64x64
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # normaliza [0,1]
            img = img.astype("float32") / 255.0

            # adiciona canal: (1, H, W)
            img = img[None, :, :]  # shape (1, 64, 64)
            imgs.append(img)

        if len(imgs) != SEQ_LEN:
            # algum frame falhou, ignora essa sequência
            continue

        # imgs é lista de arrays (1, 64, 64)
        # junta em um único array (SEQ_LEN, 1, 64, 64)
        imgs_np = np.stack(imgs, axis=0).astype("float32")

        # converte para tensor de forma eficiente
        seq_tensor = torch.from_numpy(imgs_np)  # (SEQ_LEN, 1, 64, 64)

        out_name = f"{video_id}_seq{num_seqs:04d}.pt"
        out_path = SEQS_ROOT / out_name

        torch.save(
            {
                "video": seq_tensor,       # (SEQ_LEN, 1, 64, 64)
                "label": int(label),       # 0 ou 1
                "video_id": video_id,
            },
            out_path
        )

        num_seqs += 1

    return num_seqs


def main():
    if not INDEX_PATH.is_file():
        print(f"[ERRO] Index {INDEX_PATH} não existe. Rode primeiro extract_frames.py")
        return

    with open(INDEX_PATH, "r") as f:
        index = json.load(f)

    total_seqs = 0

    for entry in index:
        video_id = entry["video_id"]   # ex: Fold1_part1_001_0
        label = entry["label"]         # 0 = alerta, 1 = sonolento

        print(f"Gerando sequências para {video_id} (label={label}) ...")
        n = build_sequences_for_video(video_id, label)
        print(f"  -> {n} sequências geradas.")
        total_seqs += n

    print(f"\n[OK] Total de sequências geradas: {total_seqs}")
    print(f"Salvas em: {SEQS_ROOT}")


if __name__ == "__main__":
    main()
