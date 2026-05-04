import cv2
import easyocr
import os
import numpy as np
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont

def traducir_con_ia(texto):
    if not texto or not str(texto).strip(): 
        return "" # Retorna string vacío si no hay texto
    
    try:
        traduccion = GoogleTranslator(source='auto', target='es').translate(texto)
        return traduccion if traduccion is not None else texto
    except Exception as e:
        print(f"⚠️ Error en Traducción: {e}")
        return str(texto) # Si falla, devuelve al menos el texto original

print("🚀 Cargando modelos de IA con optimización...")
# Al cargar el modelo, EasyOCR ya sabe que debe buscar caracteres complejos
reader = easyocr.Reader(['ja', 'en'], gpu=False) # Pon True si tienes NVIDIA CUDA

# --- CONFIGURACIÓN DE RUTAS ---
carpeta_entrada = 'entrada'
carpeta_salida = 'salida'

# Crear carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)
    print(f"✅ Carpeta '{carpeta_salida}' creada.")

# Inicializar EasyOCR fuera del bucle (para que cargue solo una vez y sea más rápido)
print("🚀 Cargando modelos de IA...")
reader = easyocr.Reader(['ja', 'en'])

# Obtener lista de archivos y ordenarlos (001, 002...)
archivos = sorted([f for f in os.listdir(carpeta_entrada) if f.endswith(('.webp', '.jpg', '.png'))])

if not archivos:
    print(f"❌ No hay imágenes en la carpeta '{carpeta_entrada}'")
else:
    print(f"📂 Encontradas {len(archivos)} imágenes. Iniciando proceso...")

    for nombre_archivo in archivos:
        path_entrada = os.path.join(carpeta_entrada, nombre_archivo)
        path_salida = os.path.join(carpeta_salida, nombre_archivo)
        
        print(f"\n🖼️ Procesando: {nombre_archivo}")
        
        # 1. Leer imagen
        image = cv2.imread(path_entrada)
        if image is None: continue
        image_limpia = image.copy()
        
        # 2. Pre-procesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # 3. OCR (He quitado paragraph=True para evitar el error de desempaquetado anterior)
        results = reader.readtext(processed_img)

        # 4. Dibujo con PIL
        pil_img = Image.fromarray(cv2.cvtColor(image_limpia, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 5. Traducir y limpiar globos
        for (bbox, text, prob) in results:
            if prob > 0.20:  # Ajusta este umbral según necesites
                texto_es = traducir_con_ia(text)
                
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                # Dibujar rectángulo blanco y texto
                draw.rectangle([top_left, bottom_right], fill=(255, 255, 255))
                draw.text(top_left, texto_es, font=font, fill=(0, 0, 0))

        # 6. Guardar resultado
        final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_salida, final_img)
        print(f"💾 Guardado en: {path_salida}")

    print("\n✨ ¡Proceso completado! Todas las imágenes están en la carpeta 'salida'.")