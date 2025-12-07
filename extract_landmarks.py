import os
import json
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

FRAMES_ROOT = os.path.join("..", "data", "frames")
LANDMARKS_ROOT = os.path.join("..", "data", "landmarks")
os.makedirs(LANDMARKS_ROOT, exist_ok=True)

def process_video_frames(video_id: str, n_frames: int):
    video_dir = os.path.join(FRAMES_ROOT, video_id)
    if not os.path.isdir(video_dir):
        print(f"Pasta de frames não encontrada: {video_dir}")
        return None

    landmarks_list = []    print("\n[OK] Landmarks salvos em", LANDMARKS_ROOT)


    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:

        for i in range(n_frames):
            frame_path = os.path.join(video_dir, f"frame_{i:06d}.jpg")
            if not os.path.exists(frame_path):
                break

            img = cv2.imread(frame_path)
            if img is None:
                landmarks_list.append(None)
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            if not res.multi_face_landmarks:
                landmarks_list.append(None)
                continue

            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = img.shape
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            landmarks_list.append(pts)

    return landmarks_list


def main():
    index_path = os.path.join("..", "data", "videos_index.json")
    if not os.path.exists(index_path):
        print("Rodar primeiro: extract_frames.py")
        return

    with open(index_path, "r") as f:
        index = json.load(f)

    for entry in index:
        vid = entry["video_id"]
        n_frames = entry["n_frames"]
        out_path = os.path.join(LANDMARKS_ROOT, vid + ".npy")

        if os.path.exists(out_path):
            print(f"[PULO] Já existe {out_path}")
            continue

        print(f"Extraindo landmarks de {vid} ...")
        lm_list = process_video_frames(vid, n_frames)
        if lm_list is None:
            continue

        # vamos guardar como array de objetos (alguns frames podem ser None)
        np.save(out_path, np.array(lm_list, dtype=object))


if __name__ == "__main__":
    main()

