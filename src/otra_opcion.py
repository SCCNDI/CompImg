import sys
import numpy as np
from skimage import io, color, transform, metrics, img_as_float


def comparar_similitud(ruta_img1, ruta_img2, tamaño_analisis=(300, 300)):
    """
    Compara dos imágenes usando scikit-image normalizando su escala.

    Args:
        ruta_img1, ruta_img2: Rutas de los archivos.
        tamaño_analisis: Tupla (ancho, alto) a la que se redimensionarán ambas
                         imágenes para hacerlas comparables sin importar su tamaño original.
    """
    try:
        # 1. Cargar imágenes
        img1 = io.imread(ruta_img1)
        img2 = io.imread(ruta_img2)

        # 2. Preprocesamiento para Invariancia de Tamaño y Color
        # Convertimos a escala de grises para analizar estructura (luminancia)
        # Si las imágenes tienen canal alfa (transparencia), lo eliminamos tomando solo RGB
        img1_gray = color.rgb2gray(img1[:, :, :3]) if img1.ndim == 3 else img1
        img2_gray = color.rgb2gray(img2[:, :, :3]) if img2.ndim == 3 else img2

        # Redimensionamos a un tamaño común estándar.
        # Esto elimina la sensibilidad al tamaño original.
        img1_norm = transform.resize(img1_gray, tamaño_analisis, anti_aliasing=True)
        img2_norm = transform.resize(img2_gray, tamaño_analisis, anti_aliasing=True)

        # Convertir a float para asegurar precisión en métricas
        img1_norm = img_as_float(img1_norm)
        img2_norm = img_as_float(img2_norm)

        # 3. Cálculo de Similitud Estructural (SSIM)
        # data_range=1.0 porque img_as_float normaliza de 0 a 1.
        # SSIM devuelve un valor entre -1 y 1 (siendo 1 idéntico).
        score_ssim = metrics.structural_similarity(img1_norm, img2_norm, data_range=1.0)

        # 4. Comparación de Histogramas (Similitud de distribución de luz/espectro)
        # Calculamos el histograma de cada imagen normalizada
        hist1, _ = np.histogram(img1_norm, bins=256, range=(0, 1))
        hist2, _ = np.histogram(img2_norm, bins=256, range=(0, 1))

        # Normalizamos los histogramas para que sumen 1 (distribución de probabilidad)
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Usamos la intersección de histogramas o distancia euclidiana
        # Aquí calculamos qué tan parecidas son las distribuciones
        diferencia_hist = np.sum(np.abs(hist1 - hist2))  # 0 es idéntico
        similitud_hist = max(0, 1 - (diferencia_hist / 2))  # Aproximación a 0-1

        return {
            "Indice_Estructural_SSIM": round(score_ssim, 4),  # 1.0 es perfecto
            "Similitud_Histograma": round(similitud_hist, 4),  # 1.0 es perfecto
            "Conclusion": "Alta Similitud" if score_ssim > 0.8 else "Baja Similitud"
        }

    except Exception as e:
        return f"Error procesando imágenes: {str(e)}"

# Ejemplo de uso:


imagen1 = sys.argv[1]
imagen2 = sys.argv[2]

resultado = comparar_similitud(imagen1,imagen2)
print(resultado)