import cv2
import numpy as np

# Variável para definir se o modelo é YOLOv3-tiny ou YOLOv3 completo
TINY = False # Use True para YOLOv3-tiny, False para YOLOv3 completo

# Configurações do modelo YOLOv3 ou YOLOv3-tiny
ARQUIVO_CFG = "detector-mala/yolov3{}.cfg".format("-tiny" if TINY else "")
ARQUIVO_PESOS = "detector-mala/yolov3{}.weights".format("-tiny" if TINY else "")
ARQUIVO_CLASSES = "detector-mala/coco.names"

# Carregar os nomes das classes
with open(ARQUIVO_CLASSES, "r") as arquivo:
    CLASSES = [linha.strip() for linha in arquivo.readlines()]

# Índices das classes "handbag" e "suitcase" no COCO dataset
idx_classes_malas = [26, 28]  # Atualize estes índices conforme necessário

# Gerar cores diferentes para cada classe
CORES = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def carregar_modelo_pretreinado():
    """
    Carrega o modelo YOLOv3-tiny ou YOLOv3 completo pré-treinado e configurações associadas ao OpenCV.
    """
    modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
    modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if modelo.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de objetos.")
    print("Modelo carregado com sucesso.")
    return modelo

def processar_frame_completo(frame):
    """
    Pré-processa e processa o frame para detecção: redimensiona, normaliza e aplica filtros de imagem.
    """
    frame = cv2.resize(frame, (320, 240))
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    
    # Processamento de imagem adicional
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_threshold = cv2.adaptiveThreshold(img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    img_blur = cv2.medianBlur(img_threshold, 5)
    kernel = np.ones((3, 3), np.int8)
    img_dil = cv2.dilate(img_blur, kernel)

    return blob, img_dil

def detectar_objetos(frame, modelo):
    """
    Detecta objetos no frame usando o modelo carregado.
    """
    blob, _ = processar_frame_completo(frame)
    modelo.setInput(blob)
    nomes_camadas = modelo.getLayerNames()
    camadas_saida = [nomes_camadas[i - 1] for i in modelo.getUnconnectedOutLayers()]
    saidas = modelo.forward(camadas_saida)
    return saidas

def desenhar_deteccoes(frame, deteccoes, limiar=0.5):
    """
    Desenha retângulos ao redor dos objetos detectados com confiança acima do limiar.
    Filtra apenas as detecções de malas.
    """
    (altura, largura) = frame.shape[:2]
    caixas = []
    confiancas = []
    ids_classes = []

    for saida in deteccoes:
        for deteccao in saida:
            pontuacoes = deteccao[5:]
            id_classe = np.argmax(pontuacoes)
            confianca = pontuacoes[id_classe]
            if confianca > limiar and id_classe in idx_classes_malas:
                caixa = deteccao[0:4] * np.array([largura, altura, largura, altura])
                (centroX, centroY, largura_caixa, altura_caixa) = caixa.astype("int")
                x = int(centroX - (largura_caixa / 2))
                y = int(centroY - (altura_caixa / 2))

                caixas.append([x, y, int(largura_caixa), int(altura_caixa)])
                confiancas.append(float(confianca))
                ids_classes.append(id_classe)

    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar, limiar - 0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (caixas[i][0], caixas[i][1])
            (largura_caixa, altura_caixa) = (caixas[i][2], caixas[i][3])
            cor = [int(c) for c in CORES[ids_classes[i]]]
            cv2.rectangle(frame, (x, y), (x + largura_caixa, y + altura_caixa), cor, 2)
            texto = f"{CLASSES[ids_classes[i]]}: {confiancas[i]:.2f}"
            cv2.putText(frame, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            print(f"Detecção: {texto} - Caixa: {x, y, largura_caixa, altura_caixa}")

def main(): 
    
    """
    Executa a detecção de objetos em múltiplos vídeos.
    """
    print("Inicializando o detector de objetos...")
    modelo = carregar_modelo_pretreinado()
    caminhos_dos_videos = [
        'detector-mala/1012162400-preview.mp4',
        'detector-mala/28647520-preview.mp4',
        'detector-mala/1889074-preview.mp4'
        
    ]  # Adicione os caminhos dos seus vídeos aqui

    for caminho_do_video in caminhos_dos_videos:
        captura_video = cv2.VideoCapture(caminho_do_video)
        if not captura_video.isOpened():
            print(f"Não foi possível abrir o vídeo: {caminho_do_video}")
            continue

        limiar_confianca = 0.4  # valor inicial do limiar de confiança. 
        # caso queira detectar uma variedade maior de itens, seria melhor um limiar maior, para evitar a detecção de objetos de forma errada.

        cv2.namedWindow('Detecta Objetos')

        try:
            while True:
                ret, frame = captura_video.read()
                if not ret:
                    break

                frame_copia = frame.copy()

                deteccoes = detectar_objetos(frame_copia, modelo)
                desenhar_deteccoes(frame, deteccoes, limiar_confianca)

                _, frame_processado = processar_frame_completo(frame_copia)

                # Redimensionar o frame processado para ter a mesma altura que o frame original
                frame_processado_redimensionado = cv2.resize(frame_processado, (frame.shape[1], frame.shape[0]))

                # Mostrar os dois frames lado a lado
                frame_concat = np.hstack((frame, cv2.cvtColor(frame_processado_redimensionado, cv2.COLOR_GRAY2BGR)))

                cv2.imshow('Detecta Objetos', frame_concat)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            captura_video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()