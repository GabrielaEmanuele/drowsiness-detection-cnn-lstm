import os
import cv2
import json

VIDEO_ROOT = os.path.join("..", "videos")
FRAMES_ROOT = os.path.join("..", "data", "frames")
os.makedirs(FRAMES_ROOT, exist_ok=True)

VALID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MOV")


def class_from_filename(fname: str):
    """Mapeia nome do arquivo para rótulo binário.
       0.mov  -> 0 (alerta)
       10.MOV -> 1 (sonolento)
       5.mov  -> None (ignoramos por enquanto)
    """
    base = os.path.splitext(fname)[0]
    if base == "0":
        return 0
    if base.lower() == "10":
        return 1
    return None  # 5 = baixa vigilância (fora da prática)


def extract_frames_from_video(video_path: str, out_dir: str,
                              fps_target: int = 10) -> int:
    """Extrai frames com fps reduzido. Retorna número de frames salvos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não abriu {video_path}")
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(int(orig_fps // fps_target), 1)
    frame_idx = 0
    saved = 0

    os.makedirs(out_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            out_name = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    index = []

    for fold_name in ["Fold1_part1", "Fold1_part2"]:
        fold_path = os.path.join(VIDEO_ROOT, fold_name)
        if not os.path.isdir(fold_path):
            print(f"[AVISO] Pasta não encontrada: {fold_path}")
            continue

        for participant_id in sorted(os.listdir(fold_path)):
            part_path = os.path.join(fold_path, participant_id)
            if not os.path.isdir(part_path):
                continue

            for fname in sorted(os.listdir(part_path)):
                if not fname.endswith(VALID_EXT):
                    continue

                label = class_from_filename(fname)
                if label is None:
                    # ignorando baixa vigilância por enquanto
                    continue

                video_path = os.path.join(part_path, fname)
                video_id = f"{fold_name}_{participant_id}_{os.path.splitext(fname)[0]}"

                print(f"Extraindo frames de {video_id} ...")
                out_dir = os.path.join(FRAMES_ROOT, video_id)
                n_frames = extract_frames_from_video(video_path, out_dir)

                if n_frames == 0:
                    print("  >> nenhum frame salvo.")
                    continue

                index.append({
                    "video_id": video_id,
                    "fold": fold_name,
                    "participant": participant_id,
                    "file": fname,
                    "label": int(label),
                    "n_frames": int(n_frames)
                })

    index_path = os.path.join("..", "data", "videos_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[OK] Index salvo em {index_path}")
    print(f"Total de vídeos processados: {len(index)}")


if __name__ == "__main__":
    main()

