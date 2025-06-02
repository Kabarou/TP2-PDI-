import os
import cv2
import numpy as np


def ordenar_puntos(puntos):
    """
    Recibe un array de 4 puntos (x, y) en orden arbitrario y devuelve
    un array con los mismos 4 puntos en el orden: [tl, tr, br, bl].
    """
    s = puntos.sum(axis=1)
    diff = np.diff(puntos, axis=1)
    tl = puntos[np.argmin(s)]
    br = puntos[np.argmax(s)]
    tr = puntos[np.argmin(diff)]
    bl = puntos[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

# Directorio donde están las imágenes originales
carpeta_entrada = "Resistencias"
carpeta_salida = "Resistencias_out"

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)

# Rango HSV para azules
lower_blue = np.array([ 90,  80,  40]) 
upper_blue = np.array([150, 255, 255])

# kernel para procesamiento morfologico 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))


# EJERCICIO A) Y B)

for i in range(1, 11):           
    for j in ['a', 'b', 'c', 'd']:
        nombre = f"R{i}_{j}.jpg"
        ruta = os.path.join(carpeta_entrada, nombre)
        img_bgr = cv2.imread(ruta)

        # Convertimos a HSV y aplicamos mascara azul
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

        # Aplicamos morfología - Apertura y clausura
        mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_morph = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel, iterations=1)

        # Encontramos contornos y nos quedamos con el mayor
        contornos, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_max = max(contornos, key=cv2.contourArea)

        # Usamos descriptor topologico convexHull 
        hull = cv2.convexHull(cont_max)

        # Buscamos 4 vertices del polígono
        vertices = None
        epsilon = 0.01 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            vertices = approx.reshape(4, 2).astype("float32")
            
        else: # Si no conseguimos 4 vertices, usamos boundingRect como último recurso
            print("No se consiguio un poligono de 4 vertices")
            x, y, w, h = cv2.boundingRect(hull)
            vertices = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
            ], dtype="float32")

        # 5) Ordenar puntos y calcular tamaño destino
        puntos_ordenados = ordenar_puntos(vertices)
        wA = np.linalg.norm(puntos_ordenados[2] - puntos_ordenados[3])
        wB = np.linalg.norm(puntos_ordenados[1] - puntos_ordenados[0])
        hA = np.linalg.norm(puntos_ordenados[1] - puntos_ordenados[2])
        hB = np.linalg.norm(puntos_ordenados[0] - puntos_ordenados[3])
        width  = int(max(wA, wB))
        height = int(max(hA, hB))
        puntos_destino = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Calculamos homografía y recortamos
        M, mask = cv2.findHomography(puntos_ordenados, puntos_destino, cv2.RANSAC, 3.0)
        img_recortada = cv2.warpPerspective(img_bgr, M, (width, height))

        # Guardamos en carpeta nueva con sufijo "_out"
        nombre_salida = f"R{i}_{j}_out.jpg"
        path_salida = os.path.join(carpeta_salida, nombre_salida)
        cv2.imwrite(path_salida, img_recortada)



# EJERICIO C), D) y E)

def detectar_bandas(img):
    """ El objetivo de esta funcion es que devuelva las 3 bandas recortadas de la imagen"""
    return 
def detectar_color_banda(img):
    """ El objetivo de esta funcion es que devuelva el color correspondiente de la banda"""
    return 

color_a_valor = {
    "Negro":   0,
    "Marrón":  1,
    "Rojo":    2,
    "Naranja": 3,
    "Amarillo":4,
    "Verde":   5,
    "Azul":    6,
    "Violeta": 7,
    "Gris":    8,
    "Blanco":  9
}

def calcular_resistencia(banda1: str, banda2: str, banda3: str) -> int:
    """
    Calcula el valor de una resistencia en ohmios a partir de las tres bandas:
      - banda1 y banda2 representan los dígitos significativos (0–9).
      - banda3 es el multiplicador (10^n).

    Parámetros:
        banda1 (str): color de la primera banda (dígito decenas).
        banda2 (str): color de la segunda banda (dígito unidades).
        banda3 (str): color de la tercera banda (multiplicador).

    Retorna:
        int: valor de la resistencia en ohmios.
    """
    # Obtener los dígitos correspondientes
    d1 = color_a_valor[banda1]
    d2 = color_a_valor[banda2]
    # El exponente para el multiplicador es el mismo valor numérico del color
    exponente = color_a_valor[banda3]

    # Construir el número base de dos dígitos y luego aplicar el multiplicador
    valor_base = d1 * 10 + d2
    resistencia_ohm = valor_base * (10 ** exponente)
    return resistencia_ohm


for i in range(1, 11):
    nombre = f"R{i}_a_out.jpg"
    ruta = os.path.join(carpeta_salida, nombre)
    img_bgr = cv2.imread(ruta)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_not(mask) # invertimos

    kernel_clausura = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mascara_clausura = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clausura, iterations=2)
    mascara_apertura = cv2.morphologyEx(mascara_clausura, cv2.MORPH_OPEN, kernel_apertura, iterations=1)

    contornos, _ = cv2.findContours(mascara_apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_max = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cont_max)

    padding = 25
    x = x + padding
    y = y + padding
    w = w - 2 * padding
    h = h - 2 * padding

    resistor = img_bgr[y:y+h, x:x+w]
    resistor_recortado = cv2.resize(resistor, (200, (h / w)))

    img_banda1, img_banda2, img_banda3 = detectar_bandas(resistor_recortado)
    banda1 = detectar_color_banda(img_banda1)
    banda2 = detectar_color_banda(img_banda2)
    banda3 = detectar_color_banda(img_banda3)

    valor = calcular_resistencia(banda1, banda2, banda3)
  
    print("El color de las bandas de la resistencia {nombre} es:")
    print("banda 1: {banda1}")
    print("banda 2: {banda1}")
    print("banda 3: {banda1}")
    print(f" valor resistencia: {valor} Ω")
    
