import numpy as np
from skimage import io, color, transform, metrics, img_as_float

def comparar_similitud_total(ruta_img1, ruta_img2, tamaño_norm=(300, 300)):
    """
    Compara dos imágenes siendo insensible a:
    1. Escala (Tamaño)
    2. Rotación (0, 90, 180, 270 grados)
    """
    try:
        # --- 1. Carga y Normalización de Tamaño ---
        img1 = io.imread(ruta_img1)
        img2 = io.imread(ruta_img2)

        # Convertir a escala de grises para análisis estructural
        # (Si tiene canal alfa, lo ignoramos)
        img1_gray = color.rgb2gray(img1[:,:,:3]) if img1.ndim == 3 else img1
        img2_gray = color.rgb2gray(img2[:,:,:3]) if img2.ndim == 3 else img2

        # Redimensionar ambas al mismo cuadrado para eliminar el factor "tamaño"
        img1_norm = transform.resize(img1_gray, tamaño_norm, anti_aliasing=True)
        img2_norm = transform.resize(img2_gray, tamaño_norm, anti_aliasing=True)
        
        # Convertir a float
        img1_norm = img_as_float(img1_norm)
        img2_norm = img_as_float(img2_norm)

        # --- 2. Análisis de Histograma (Espectral) ---
        # El histograma NO cambia con la rotación, así que lo calculamos una sola vez.
        hist1, _ = np.histogram(img1_norm, bins=256, range=(0, 1))
        hist2, _ = np.histogram(img2_norm, bins=256, range=(0, 1))
        
        # Normalización estadística (suma = 1)
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Similitud de histograma (Intersección)
        diferencia_hist = np.sum(np.abs(hist1 - hist2))
        score_histograma = max(0, 1 - (diferencia_hist / 2))

        # --- 3. Análisis Estructural con Rotación (SSIM) ---
        best_ssim = -1
        best_angle = 0
        
        # Probamos 4 rotaciones cardinales: 0, 90, 180, 270
        # Usamos np.rot90 que es más preciso y rápido que transform.rotate para ángulos rectos
        for k in range(4): # k=0 (0°), k=1 (90°), k=2 (180°), k=3 (270°)
            img2_rotated = np.rot90(img2_norm, k=k)
            
            # Calculamos SSIM para esta rotación específica
            current_ssim = metrics.structural_similarity(
                img1_norm, 
                img2_rotated, 
                data_range=1.0
            )
            
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                best_angle = k * 90

        # --- 4. Interpretación de Resultados ---
        # Combinamos ambos scores (Estructura + Histograma)
        # Damos más peso a la estructura (0.7) que al color (0.3)
        score_final = (best_ssim * 0.7) + (score_histograma * 0.3)

        return {
            "Similitud_Global": f"{score_final*100:.2f}%",
            "Detalles": {
                "Mejor_SSIM_Estructural": round(best_ssim, 4),
                "Similitud_Histograma": round(score_histograma, 4),
                "Rotacion_Detectada": f"{best_angle} grados"
            },
            "Veredicto": "Es la misma imagen (posiblemente rotada)" if score_final > 0.85 else "Son diferentes"
        }

    except Exception as e:
        return f"Error: {str(e)}"

# Ejemplo ficticio:
# print(comparar_similitud_total("original.jpg", "copia_rotada_90.jpg"))