import cv2
import easyocr
import os
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont # Para escribir bonito

# Configura tu API Key real aquí
client = OpenAI(api_key="TU_KEY_AQUI")

def traducir_con_ia(texto):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un traductor experto de manga. Traduce a español neutro de forma concisa para que quepa en un globo de texto."},
                {"role": "user", "content": f"Traduce: {texto}"}
            ]
        )
        return response.choices[0].message.content
    except:
        return texto # Si falla la API, devuelve el texto original

if not os.path.exists('test.jpg'):
    print("❌ No encuentro 'test.jpg'")
else:
    print("✅ Iniciando IA...")
    reader = easyocr.Reader(['es', 'en'])
    image = cv2.imread('test.jpg')
    image_limpia = image.copy()

    results = reader.readtext(image)

    # Convertimos la imagen de OpenCV (BGR) a formato PIL (RGB) para escribir
    pil_img = Image.fromarray(cv2.cvtColor(image_limpia, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Intentamos cargar una fuente estándar (puedes descargar una fuente de manga .ttf)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for (bbox, text, prob) in results:
        if prob > 0.20:
            # 1. TRADUCIR (Llamamos a la función ahora sí)
            print(f"📝 Original: {text}")
            texto_es = traducir_con_ia(text)
            print(f"🌍 Traducido: {texto_es}")

            # 2. COORDENADAS
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # 3. LIMPIAR EL GLOBO (Pintar rectángulo blanco en PIL)
            draw.rectangle([top_left, bottom_right], fill=(255, 255, 255))

            # 4. ESCRIBIR TEXTO NUEVO
            # Lo ponemos en color negro (0,0,0)
            draw.text(top_left, texto_es, font=font, fill=(0, 0, 0))

    # Convertimos de vuelta a OpenCV para mostrar/guardar
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imwrite('resultado_final.jpg', final_img)
    print("💾 ¡Proyecto terminado! Mira 'resultado_final.jpg'")
    
    cv2.imshow('Manga Traducido', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()