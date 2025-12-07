import os
import json
import numpy as np

PastaDados = os.path.join("..", "data")
PastaLandmarks = os.path.join(PastaDados, "landmarks")
PastaAtributos = os.path.join(PastaDados, "features")

os.makedirs(PastaAtributos, exist_ok=True)

ArquivoIndiceLandmarks = os.path.join(PastaDados, "landmarks_index.json")
ArquivoIndiceVideos = os.path.join(PastaDados, "videos_index.json")  # usado como reserva, se precisar

# Olho esquerdo
IndicesOlhoEsquerdo = [33, 160, 158, 133, 153, 144]

# Olho direito 
IndicesOlhoDireito = [362, 385, 387, 263, 373, 380]

# Boca (6 pontos principais)
IndicesBoca = [61, 146, 91, 181, 84, 17]

# Ponta aproximada do nariz
IndicePontaNariz = 1


def calcular_ear(pontos_olho: np.ndarray) -> float:

    ponto1, ponto2, ponto3, ponto4, ponto5, ponto6 = pontos_olho

    # Distâncias verticais
    distancia_vertical_1 = np.linalg.norm(ponto2 - ponto6)
    distancia_vertical_2 = np.linalg.norm(ponto3 - ponto5)

    # Distância horizontal
    distancia_horizontal = np.linalg.norm(ponto1 - ponto4)

    ear = (distancia_vertical_1 + distancia_vertical_2) / (2.0 * distancia_horizontal + 1e-6)
    return ear


def calcular_mar(pontos_boca: np.ndarray) -> float:
    """
    Calcula o Mouth Aspect Ratio (MAR) de forma simplificada.

    Usamos:
      - A (61): canto esquerdo
      - B (91): canto direito aproximado
      - E (146): ponto superior
      - F (84): ponto inferior

    MAR = abertura vertical da boca / largura horizontal.
    Valores altos indicam boca bem aberta (por exemplo, bocejo).
    """
    ponto_A = pontos_boca[0]  # índice 61
    ponto_E = pontos_boca[1]  # índice 146
    ponto_B = pontos_boca[2]  # índice 91
    ponto_F = pontos_boca[4]  # índice 84

    largura_boca = np.linalg.norm(ponto_B - ponto_A)
    abertura_boca = np.linalg.norm(ponto_F - ponto_E)

    mar = abertura_boca / (largura_boca + 1e-6)
    return mar


def calcular_inclinacao_roll(centro_olho_esquerdo: np.ndarray,
                             centro_olho_direito: np.ndarray) -> float:
    """
    Calcula a inclinação lateral da cabeça (head roll), em radianos.

    É o ângulo da linha que liga o centro do olho esquerdo ao centro do olho direito.
    Quando a cabeça está reta, o valor tende a ficar próximo de zero.
    """
    diferenca_x = centro_olho_direito[0] - centro_olho_esquerdo[0]
    diferenca_y = centro_olho_direito[1] - centro_olho_esquerdo[1]

    angulo_roll = np.arctan2(diferenca_y, diferenca_x)
    return angulo_roll


def calcular_inclinacao_pitch(ponto_nariz: np.ndarray,
                              ponto_meio_olhos: np.ndarray) -> float:
    """
    Calcula uma aproximação da inclinação para frente/trás da cabeça (head pitch).

    Usa a diferença vertical entre a ponta do nariz e o ponto médio entre os olhos,
    normalizada pela distância entre esses pontos, para ficar independente da escala.
    """
    diferenca_vertical = ponto_nariz[1] - ponto_meio_olhos[1]
    distancia_nariz_olhos = np.linalg.norm(ponto_nariz - ponto_meio_olhos) + 1e-6

    pitch = diferenca_vertical / distancia_nariz_olhos
    return pitch


def calcular_perclos_local(valores_ear: np.ndarray,
                           limite_olho_fechado: float = 0.21,
                           tamanho_janela: int = 30) -> np.ndarray:
    """
    Calcula o PERCLOS "local" para cada frame.

    - valores_ear: vetor [T] com o EAR de cada frame
    - limite_olho_fechado: abaixo deste valor, consideramos olho "fechado"
    - tamanho_janela: quantos frames considerar na janela (ex.: 30 frames ~ 1s em 30 fps)

    Para cada frame t, olhamos uma janela [t - tamanho_janela + 1, ..., t]
    e calculamos qual porcentagem desses frames estava com os olhos fechados.
    """
    quantidade_frames = len(valores_ear)

    mascara_fechado = (valores_ear < limite_olho_fechado).astype(np.float32)
    perclos_por_frame = np.zeros(quantidade_frames, dtype=np.float32)

    for indice_frame in range(quantidade_frames):
        inicio_janela = max(0, indice_frame - tamanho_janela + 1)
        janela_fechado = mascara_fechado[inicio_janela:indice_frame + 1]
        perclos_por_frame[indice_frame] = janela_fechado.mean()

    return perclos_por_frame


# ==============================
# Cálculo de atributos para um vídeo
# ==============================

def calcular_atributos_por_video(identificador_video: str,
                                 caminho_landmarks: str) -> np.ndarray:
    """
    Carrega os landmarks de um vídeo (arquivo .npy) e gera,
    para cada frame, um vetor de atributos com:

      [EAR, MAR, PERCLOS_local, head_roll, head_pitch]
    """

    # 1) Carrega o arquivo permitindo pickle,
    #    pois alguns foram salvos como array de objetos
    dados_brutos = np.load(caminho_landmarks, allow_pickle=True)

    # 2) Se for um array de objetos, filtramos apenas os frames válidos
    if isinstance(dados_brutos, np.ndarray) and dados_brutos.dtype == object:
        frames_validos = []

        for frame in dados_brutos:
            # ignora frames vazios ou None
            if frame is None:
                continue

            frame_array = np.array(frame)

            # esperamos algo como [468, 2] (FaceMesh completo)
            if (
                frame_array.ndim == 2
                and frame_array.shape[0] >= 400
                and frame_array.shape[1] == 2
            ):
                frames_validos.append(frame_array)

        if len(frames_validos) == 0:
            print(f"[AVISO] Nenhum frame válido em {caminho_landmarks}")
            return None

        try:
            matriz_landmarks = np.stack(frames_validos, axis=0)
        except Exception as erro:
            print(f"[ERRO] Não foi possível empilhar os landmarks filtrados de {caminho_landmarks}: {erro}")
            return None

    else:
        # Caso já esteja num formato numérico comum (ex.: [T, 468, 2])
        matriz_landmarks = np.array(dados_brutos)

    # Verificação básica de formato
    if matriz_landmarks.ndim != 3 or matriz_landmarks.shape[1] < 400:
        print(f"[AVISO] Formato inesperado em {caminho_landmarks}: {matriz_landmarks.shape}")
        return None

    quantidade_frames = matriz_landmarks.shape[0]

    # Vetores para armazenar as séries temporais
    serie_ear = np.zeros(quantidade_frames, dtype=np.float32)
    serie_mar = np.zeros(quantidade_frames, dtype=np.float32)
    serie_roll = np.zeros(quantidade_frames, dtype=np.float32)
    serie_pitch = np.zeros(quantidade_frames, dtype=np.float32)

    for indice_frame in range(quantidade_frames):
        # landmarks do frame atual: [468, 2]
        pontos_face = matriz_landmarks[indice_frame]

        # Pontos dos olhos
        pontos_olho_esquerdo = pontos_face[IndicesOlhoEsquerdo]
        pontos_olho_direito = pontos_face[IndicesOlhoDireito]

        # Pontos da boca
        pontos_boca = pontos_face[IndicesBoca]

        # Ponta do nariz e centro dos olhos
        ponto_nariz = pontos_face[IndicePontaNariz]
        centro_olho_esquerdo = pontos_olho_esquerdo.mean(axis=0)
        centro_olho_direito = pontos_olho_direito.mean(axis=0)
        ponto_meio_olhos = 0.5 * (centro_olho_esquerdo + centro_olho_direito)

        # EAR: média dos dois olhos (mais robusto)
        ear_esquerdo = calcular_ear(pontos_olho_esquerdo)
        ear_direito = calcular_ear(pontos_olho_direito)
        ear_medio = 0.5 * (ear_esquerdo + ear_direito)

        # MAR (abertura da boca)
        mar_atual = calcular_mar(pontos_boca)

        # head_roll e head_pitch
        valor_roll = calcular_inclinacao_roll(centro_olho_esquerdo, centro_olho_direito)
        valor_pitch = calcular_inclinacao_pitch(ponto_nariz, ponto_meio_olhos)

        serie_ear[indice_frame] = ear_medio
        serie_mar[indice_frame] = mar_atual
        serie_roll[indice_frame] = valor_roll
        serie_pitch[indice_frame] = valor_pitch

    # PERCLOS local calculado a partir do EAR
    serie_perclos = calcular_perclos_local(
        serie_ear,
        limite_olho_fechado=0.21,
        tamanho_janela=30
    )

    # Monta a matriz final de atributos: [T, 5]
    matriz_atributos = np.stack(
        [serie_ear, serie_mar, serie_perclos, serie_roll, serie_pitch],
        axis=1
    ).astype(np.float32)

    return matriz_atributos


# ==============================
# Carregamento do índice de landmarks
# ==============================

def carregar_indice_landmarks():
    """
    Carrega o índice de arquivos de landmarks.

    Preferência:
      1. landmarks_index.json (gerado pelo script de landmarks)
      2. videos_index.json (como alternativa, se o primeiro não existir)
    """
    if os.path.exists(ArquivoIndiceLandmarks):
        with open(ArquivoIndiceLandmarks, "r") as arquivo:
            return json.load(arquivo)

    if os.path.exists(ArquivoIndiceVideos):
        with open(ArquivoIndiceVideos, "r") as arquivo:
            lista_videos = json.load(arquivo)

        lista_indice = []
        for objeto_video in lista_videos:
            identificador_video = objeto_video["video_id"]
            caminho_landmarks = os.path.join(PastaLandmarks, f"{identificador_video}.npy")

            lista_indice.append({
                "video_id": identificador_video,
                "landmarks_path": caminho_landmarks,
                "label": objeto_video.get("label", None),
                "n_frames": objeto_video.get("n_frames", None)
            })

        return lista_indice

    print("[ERRO] Nem landmarks_index.json nem videos_index.json foram encontrados.")
    return []

def main():
    indice_landmarks = carregar_indice_landmarks()
    if not indice_landmarks:
        print("[ERRO] Nenhum índice de landmarks foi carregado.")
        return

    indice_atributos = []

    for item in indice_landmarks:
        identificador_video = item["video_id"]
        caminho_landmarks = item.get("landmarks_path") or os.path.join(
            PastaLandmarks,
            f"{identificador_video}.npy"
        )

        if not os.path.exists(caminho_landmarks):
            print(f"Arquivo de landmarks não encontrado para {identificador_video}: {caminho_landmarks}")
            continue

        print(f"Calculando atributos (EAR, MAR, PERCLOS, inclinações) para {identificador_video} ...")

        matriz_atributos = calcular_atributos_por_video(identificador_video, caminho_landmarks)
        if matriz_atributos is None or len(matriz_atributos) == 0:
            print(f"  >> Nenhum atributo válido para {identificador_video}, pulando.")
            continue

        caminho_saida_atributos = os.path.join(PastaAtributos, f"{identificador_video}.npy")
        np.save(caminho_saida_atributos, matriz_atributos)

        indice_atributos.append({
            "video_id": identificador_video,
            "features_path": caminho_saida_atributos,
            "label": item.get("label", None),
            "n_frames": int(matriz_atributos.shape[0]),
            "n_features": int(matriz_atributos.shape[1])
        })

    # Salva o índice dos arquivos de atributos
    caminho_arquivo_indice_atributos = os.path.join(PastaDados, "features_index.json")
    with open(caminho_arquivo_indice_atributos, "w") as arquivo:
        json.dump(indice_atributos, arquivo, indent=2)

    print(f"\n[OK] Atributos salvos em {PastaAtributos}")
    print(f"[OK] Índice de atributos salvo em {caminho_arquivo_indice_atributos}")
    print(f"Total de vídeos processados: {len(indice_atributos)}")


if __name__ == "__main__":
    main()
