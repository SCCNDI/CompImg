import cv2
import imagehash
from PIL import Image
import sys
import numpy as np


def comparar_imagenes(ruta_img1, ruta_img2):
    # --- 1. Análisis Espectral / Perceptual (ImageHash) ---
    # Usamos pHash (basado en DCT - Transformada de Coseno Discreta)
    # Esto es excelente para detectar si la imagen es estructuralmente la misma
    # sin importar el tamaño.
    hash1 = imagehash.phash(Image.open(ruta_img1))
    hash2 = imagehash.phash(Image.open(ruta_img2))

    # La diferencia es la distancia de Hamming entre los hashes
    # 0 = idénticas, >10 = muy diferentes
    diferencia_hash = hash1 - hash2
    similitud_hash = max(0.0, (1 - diferencia_hash / 64.0)) * 100  # Aproximación a %

    # --- 2. Análisis de Histogramas (OpenCV) ---
    img1 = cv2.imread(ruta_img1)
    img2 = cv2.imread(ruta_img2)

    # Convertir a HSV (Hue Saturation Value) es mejor que RGB para comparar color
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Calcular histogramas (H y S)
    # Normalizamos (0 a 1) para que el TAMAÑO de la imagen no importe
    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)

    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Comparar usando Correlación (1.0 = idéntico, 0 = nada que ver)
    similitud_hist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return {
        "Diferencia_Estructural_pHash": diferencia_hash,
        "Similitud_Estructural_pct": f"{similitud_hash:.2f}%",
        "Similitud_Color_Histograma": f"{similitud_hist * 100:.2f}%"
    }

# Uso


imagen1 = sys.argv[1]
imagen2 = sys.argv[2]

resultado = comparar_imagenes(imagen1,imagen2)
print(resultado)