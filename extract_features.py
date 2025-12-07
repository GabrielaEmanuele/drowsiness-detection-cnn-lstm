import cv2
import mediapipe as mp
import numpy as np
import os
import json

mp_face_mesh = mp.solutions.face_mesh

# índices do olho esquerdo e boca no FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 146, 91, 181, 84, 17]

def calc_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return float((A + B) / (2.0 * C + 1e-6))

def calc_MAR(mouth):
    A = np.linalg.norm(mouth[1] - mouth[3])
    B = np.linalg.norm(mouth[2] - mouth[4])
    C = np.linalg.norm(mouth[0] - mouth[5])
    return float((A + B) / (2.0 * C + 1e-6))

def process_video(path, fps_target=10):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir {path}")
        return []

    sequence = []

    # para reduzir fps (os vídeos vêm até 30fps)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(int(orig_fps // fps_target), 1)
    frame_idx = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            if not res.multi_face_landmarks:
                # sem rosto nesse frame
                continue

            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            points = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

            eye_pts = points[LEFT_EYE]
            mouth_pts = points[MOUTH]

            ear = calc_EAR(eye_pts)
            mar = calc_MAR(mouth_pts)

            sequence.append([ear, mar])

    cap.release()
    return sequence

def class_from_filename(fname):
    """
    Converte nome do arquivo (0.mov, 5.mov, 10.MOV)
    em rótulo numérico. Aqui vamos fazer binário:
      0 -> alerta
      10 -> sonolento
    e ignorar 5 (baixa vigilância) por enquanto.
    """
    name = os.path.splitext(fname)[0]  # '0', '5', '10'
    if name == "0":
        return 0          # alerta
    elif name.lower() == "10":
        return 1          # sonolento
    else:
        return None       # ignorar 5 por enquanto

def main():
    # raiz dos vídeos (dentro da pasta do tcc)
    videos_root = os.path.join("..", "videos")
    output_root = os.path.join("..", "data", "features")
    os.makedirs(output_root, exist_ok=True)

    index = []

    # vamos percorrer Fold1_part1 e Fold1_part2
    for fold_name in ["Fold1_part1", "Fold1_part2"]:
        fold_path = os.path.join(videos_root, fold_name)
        if not os.path.isdir(fold_path):
            print(f"[AVISO] Pasta não encontrada: {fold_path}")
            continue

        # pastas 01, 02, ..., 12
        for participant_id in sorted(os.listdir(fold_path)):
            participant_path = os.path.join(fold_path, participant_id)
            if not os.path.isdir(participant_path):
                continue

            for fname in os.listdir(participant_path):
                label = class_from_filename(fname)
                if label is None:
                    # por enquanto ignoramos '5.mov'
                    continue

                video_path = os.path.join(participant_path, fname)
                print(f"Processando participante {participant_id} | arquivo {fname}")

                seq = process_video(video_path)

                if len(seq) == 0:
                    print("  >> nenhum frame válido, pulando.")
                    continue

                seq = np.array(seq, dtype=np.float32)

                # nome do arquivo de saída, ex: 01_0.npy ou 01_10.npy
                base_name = f"{participant_id}_{os.path.splitext(fname)[0]}"
                out_path = os.path.join(output_root, base_name + ".npy")
                np.save(out_path, seq)

                index.append({
                    "participant": participant_id,
                    "fold": fold_name,
                    "file": fname,
                    "label": int(label),
                    "features_file": out_path
                })

    # salvar índice com metadados
    index_path = os.path.join("..", "data", "features_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[OK] Extração concluída.")
    print(f"Arquivos de features em: {output_root}")
    print(f"Índice salvo em: {index_path}")

if __name__ == "__main__":
    main()
