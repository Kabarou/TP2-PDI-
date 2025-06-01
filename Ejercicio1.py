import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import os

img_color = cv2.imread('TUIA-PDI-TP2/placa.png')
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

placa = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# plt.imshow(placa, cmap='gray')
# plt.show()

def detectar_bordes(img : np.ndarray, mascara: Tuple[int, int], th1 : int, th2: int) -> Tuple:
    """
    Detecta los bordes de la imagen utilizando el operador Canny.
    A los bordes encontrados, se le segmenta a la imagen según los contornos
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
        mascara (Tuple[int, int]): Tamaño de la mascara para el filtro de suavizado.
        th1 (int): Umbral inferior para el operador Canny.
        th2 (int): Umbral superior para el operador Canny.
    Returns:
        Tuple de listas de contornos (capacitadores, resistencias, chips)
    """
    # Aplicamos un filtro de suavizado para reducir el ruido
    img_median = cv2.medianBlur(img, 5)

    # Aplicamos el detector de bordes Canny
    img_canny = cv2.Canny(img_median, th1, th2)

    # Aplicamos una dilatación para mejorar la detección de bordes
    kernel = np.ones(mascara, np.uint8)
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)
    img_closed = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel)

    return img_closed

def visualizar_todos_los_contornos(img: np.ndarray, binaria: np.ndarray) -> None:
    """
    Visualiza todos los contornos detectados por connectedComponentsWithStats.
    
    Args:
        img (np.ndarray): Imagen original (en escala de grises o color).
        binaria (np.ndarray): Imagen binaria para calcular los componentes conectados.
    """
    # Asegurarse de trabajar con una copia en color
    if len(img.shape) == 2:  # imagen en gris
        img_vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    else:
        img_vis = img.copy()

    # Obtener los componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

    # Dibujar todos los rectángulos (ignorando el primero que es el fondo)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar la imagen con los contornos
    plt.figure(figsize=(10, 10))
    plt.imshow(img_vis)
    plt.axis('off')
    plt.show()

def detectar_chips(img: np.ndarray, contornos: np.ndarray) -> list:
    """
    Detecta los chips en la imagen.
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
    Returns:
        List[np.ndarray]: Lista de imágenes de los chips detectados.
    """

    img_copy = img.copy()

    connectivity = 8
    _,_, stats, _ = cv2.connectedComponentsWithStats(contornos, connectivity, cv2.CV_32S) 

    chips = []

    for st in stats:
        x, y, w, h, _ = st
        if (285 <= w <= 315) and (590<= h <= 610):
            chips.append((x,y,w,h, "Chip"))

    # Dibujar rectángulos alrededor de los chips detectados
    for (x, y, w, h, titulo) in chips:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(img_copy)
    plt.title("Chips detectados")
    plt.show()

    return chips

def deteccion_capacitadores(img: np.ndarray, contornos: np.ndarray) -> Tuple[list, list, list, list]:
    """
    Detecta los capacitadores en la imagen.
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
    Returns:
        Tuple[list, list, list, list]: Listas de capacitadores grandes, medianos, pequeños y muy pequeños detectados.
    """
    img_copy = img.copy()

    connectivity = 8
    _,_, stats, _ = cv2.connectedComponentsWithStats(contornos, connectivity, cv2.CV_32S) 

    capacitadores_grandes = []
    capacitadores_medianos = []
    capacitadores_pequenos = []
    capacitadores_muy_pequenos = []

    for st in stats:
        x, y, w, h, area = st
        if (490 <= w <= 520) and (600 <= h <= 650):
            capacitadores_grandes.append((x, y, w, h, "Capacitadores grandes"))
        if (330 <= w <= 360) and (300 <= h <= 340):
            capacitadores_medianos.append((x, y, w, h, "Capacitadores medianos"))
        if (290 <= w <= 330) and (400 <= h <= 450):
            capacitadores_medianos.append((x, y, w, h, "Capacitadores medianos"))
        if (250 <= w <= 266) and (150 <= h <= 210):
            capacitadores_muy_pequenos.append((x, y, w, h, "Capacitadores muy pequenos"))
        if (150 <= w <= 170) and (155 <= h <= 210):
            capacitadores_muy_pequenos.append((x, y, w, h, "Capacitadores muy pequenos"))
        if (180 <= w <= 210) and (200 <= h <= 240):
            capacitadores_pequenos.append((x, y, w, h, "Capacitadores pequenos"))
        if (260 <= w <= 270) and (235 <= h <= 245):
            capacitadores_pequenos.append((x, y, w, h, "Capacitadores pequenos"))

    # Dibujar rectángulos alrededor de los capacitadores detectados
    for (x, y, w, h, titulo) in capacitadores_grandes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (100, 255, 0), 2)
    for (x, y, w, h, titulo) in capacitadores_medianos:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 100, 255), 2)
    for (x, y, w, h, titulo) in capacitadores_pequenos:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 100), 2)
    for (x, y, w, h, titulo) in capacitadores_muy_pequenos:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (100, 255, 255), 2)
    plt.imshow(img_copy)
    plt.show()
    return capacitadores_grandes, capacitadores_medianos, capacitadores_pequenos, capacitadores_muy_pequenos

def detectar_resistencia(img: np.ndarray, contornos: np.ndarray) -> list:
    """
    Detecta las resistencias en la imagen.
    Args:
        img (numpy.ndarray): Imagen de entrada en escala de grises.
    Returns:
        List[np.ndarray]: Lista de imágenes de las resistencias detectadas.
    """
    img_copy = img.copy()

    connectivity = 8
    _,_, stats, _ = cv2.connectedComponentsWithStats(contornos, connectivity, cv2.CV_32S) 

    resistencias = []

    for st in stats:
        x, y, w, h, _ = st
        if (
            ((55 <= h <= 90) or (100 <= h <= 110)) and
            ((140 <= w <= 263) or (275 <= w <= 277) or (300 <= w <= 330))
            and (y >= 800)
        ) or (
            ((190 <= h <= 230) or (300 <= h <= 330)) and (50 <= w <= 90)
        ):
            resistencias.append((x, y, w, h, "Resistencia"))

    # Dibujar rectángulos alrededor de las resistencias detectadas
    for (x, y, w, h, titulo) in resistencias:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(img_copy)
    plt.title("Resistencias detectadas")
    plt.show()

    return resistencias

def generar_imagen(img: np.ndarray, tipo_componente : List[Tuple[int, int, int, int, str]]) -> np.ndarray:
    """
    Genera una imagen con los componentes detectados.
    
    Args:
        img (np.ndarray): Imagen de entrada en escala de grises.
        tipo_componente (List[Tuple[int, int, int, int, str]]): Lista de los tipos de compones, con sus coordenadas.
    Returns:
        np.ndarray: Imagen con los componentes dibujados.
    """
    img_copy = img.copy()
    for (x, y, w, h, titulo) in tipo_componente:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, titulo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)
    return img_copy

mascara = (3,3)
th1 = 0.1* 255
th2 = 0.8 * 255
contornos = detectar_bordes(placa,mascara, th1, th2)

visualizar_todos_los_contornos(img_color, contornos)

chips = detectar_chips(img_color,contornos)
capacitadores_grandes, capacitadores_medianos, capacitadores_pequenos, capacitadores_muy_pequenos = deteccion_capacitadores(img_color,contornos)
resistencia = detectar_resistencia(img_color,contornos)

os.system('cls')

print("-------------------------------")

print("Total de capacitadores grandes:", len(capacitadores_grandes))
print("Total de capacitadores medianos:", len(capacitadores_medianos))
print("Total de capacitadores pequeños:", len(capacitadores_pequenos))
print("Total de capacitadores muy pequeños:", len(capacitadores_muy_pequenos))

print("-------------------------------")
print("Total de resistencias:", len(resistencia))

img_clasificada = generar_imagen(img_color, capacitadores_grandes + capacitadores_medianos + capacitadores_pequenos + capacitadores_muy_pequenos + resistencia + chips)

plt.figure(figsize=(10, 10))
plt.imshow(img_clasificada)
plt.axis('off')
plt.title("Componentes detectados")
plt.show()