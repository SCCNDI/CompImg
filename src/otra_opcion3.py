"""
Compara dos imágenes utilizando detección y descripción de características con OpenCV.
Este método es robusto a cambios de escala, rotación y ciertas variaciones de iluminación.
Usa el algoritmo SIFT para detectar puntos clave y describirlos, seguido de un emparejamiento
de características para evaluar la similitud.
"""
import sys
import cv2
import numpy as np

def comparar_features_opencv(ruta_img1, ruta_img2):
    # 1. Cargar imágenes en escala de grises
    img1 = cv2.imread(ruta_img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(ruta_img2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return "Error al cargar imágenes"

    # 2. Inicializar el detector SIFT
    # SIFT es excelente porque es invariante a escala, rotación e iluminación.
    sift = cv2.SIFT_create()

    # 3. Encontrar los Keypoints y Descriptores
    # Keypoint: Punto de interés (esquina, borde).
    # Descriptor: Huella digital matemática de ese punto.
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Si no hay características suficientes, abortar
    if des1 is None or des2 is None:
        return "No se encontraron características suficientes."

    # 4. Emparejamiento (Matching)
    # Usamos FLANN para velocidad, o BFMatcher para precisión absoluta.
    # Aquí usamos BFMatcher (Fuerza Bruta) con normalización L2 (típica para SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # knnMatch devuelve los 2 mejores matches para cada punto (k=2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 5. Aplicar el "Ratio Test" de David Lowe
    # Esta es la CLAVE de la calidad. Descartamos falsos positivos.
    # Un match es bueno solo si es mucho mejor que el segundo mejor match.
    buenos_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            buenos_matches.append(m)

    # 6. Calcular métrica de similitud
    # No es un porcentaje directo, pero podemos inferirlo basado en la cantidad de matches
    # relativos a la cantidad de puntos encontrados.
    numero_puntos_clave = min(len(kp1), len(kp2))
    porcentaje_match = (len(buenos_matches) / numero_puntos_clave) * 100

    # Opcional: Dibujar los matches para verlos visualmente
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, buenos_matches, None)
    # cv2.imshow("Matches", img_matches); cv2.waitKey(0)

    return {
        "Puntos_Clave_Img1": len(kp1),
        "Puntos_Clave_Img2": len(kp2),
        "Matches_De_Alta_Calidad": len(buenos_matches),
        "Score_Similitud": f"{porcentaje_match:.2f}%", # > 5-10% suele ser muy alto en Feature Matching
        "Conclusion": "Coincidencia Detectada" if len(buenos_matches) > 10 else "Poca o nula coincidencia"
    }

# USO
imagen1 = sys.argv[1]
imagen2 = sys.argv[2]

resultado = comparar_features_opencv(imagen1,imagen2)
print(resultado)
