import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
# ----ITEMS A) y B)--------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def ordenar_puntos(puntos):
    """
    Recibe un array con 4 puntos (x,y) en orden arbitrario
    y devuelve [tl, tr, br, bl] (top-left, top-right…).
    """
    s = puntos.sum(axis=1)
    diff = np.diff(puntos, axis=1)
    tl = puntos[np.argmin(s)]
    br = puntos[np.argmax(s)]
    tr = puntos[np.argmin(diff)]
    bl = puntos[np.argmax(diff)]
    return np.array([tl, tr, br, bl])

def crear_mascara_azul(img):
    """Construye y limpia con morfología la máscara del rectángulo azul."""
    # Umbrales para azules en HSV
    azul_bajo = np.array([ 90,  80,  40]) 
    azul_alto = np.array([150, 255, 255])

    # Kernel para morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    # Convertimos la imagen a HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Aplicamos mascara azul
    mask = cv2.inRange(img_hsv, azul_bajo, azul_alto)
    # Aplicamos morfología
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def extraer_rectangulo_azul(mask):
    """
    Encuentra el contorno más grande en la máscara,
    intenta aproximar a polígono de 4 vértices y los devuelve.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se encontraron contornos")   ############
    
    # Seleccionamos el contorno de mayor área 
    main_contour = max(contours, key=cv2.contourArea)

    # Encontramos el polígono convexo mínimo de 4 vertices que contiene todos los puntos del contorno
    hull = cv2.convexHull(main_contour)
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) != 4:
        raise RuntimeError(f"No se pudo aproximar a 4 vértices (encontrados: {len(approx)})")

    pts = approx.reshape(4, 2)
    
    return ordenar_puntos(pts)

def convertir_a_vista_superior(img, src_pts):
    """
    Dada la imagen original y 4 puntos ordenados,
    retorna la imagen con perspectiva corregida ('vista superior')
    utilizando Homografía.
    """
    # Calculamos dimensiones destino
    (tl, tr, br, bl) = src_pts
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    dst_w = int(max(widthA, widthB))
    dst_h = int(max(heightA, heightB))

    dst_pts = np.array([
        [0,       0],
        [dst_w-1, 0],
        [dst_w-1, dst_h-1],
        [0,       dst_h-1]
    ], dtype="float32")

    M = cv2.findHomography(src_pts, dst_pts)[0]
    return cv2.warpPerspective(img, M, (dst_w, dst_h))

def procesar_imagen(ruta):
    """Pipeline de mascara, detectar rectángulo, corregir perspectiva"""
    img_bgr = cv2.imread(str(ruta))
    mask = crear_mascara_azul(img_bgr)
    src_pts = extraer_rectangulo_azul(mask)
    top_view = convertir_a_vista_superior(img_bgr, src_pts)
    return top_view

# -------------------------------------------------------------------------
# Directorio donde están las imágenes originales
carpeta_entrada = "Resistencias"
carpeta_salida = "Resistencias_out"

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)

for i in range(1, 11):           
    for j in ['a', 'b', 'c', 'd']:
        nombre = f"R{i}_{j}.jpg"
        ruta = os.path.join(carpeta_entrada, nombre)
        print("Leyendo:", ruta)
        if not os.path.exists(ruta):
            print("¡No existe!, compruebe el directorio de trabajo", ruta)
            continue
        imagen_lista = procesar_imagen(ruta)

        # Guardamos en carpeta nueva con sufijo "_out"
        nombre_salida = f"R{i}_{j}_out.jpg"
        path_salida = os.path.join(carpeta_salida, nombre_salida)
        cv2.imwrite(path_salida, imagen_lista)

# -------------------------------------------------------------------------------------------------
# ------ITEMS C), D) Y E)--------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
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
