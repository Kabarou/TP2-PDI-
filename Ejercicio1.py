import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

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
            chips.append((x,y,w,h))

    # Dibujar rectángulos alrededor de los chips detectados
    for (x, y, w, h) in chips:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(img_copy)
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

    capacitadores_grades = []
    capacitadores_medianos = []
    capacitadores_pequenos = []
    capacitadores_muy_pequenos = []

    for st in stats:
        x, y, w, h, area = st
        if (490 <= w <= 520) and (600 <= h <= 650):
            capacitadores_grades.append((x, y, w, h))
        if (330 <= w <= 360) and (300 <= h <= 340):
            capacitadores_medianos.append((x, y, w, h))
        if (300 <= w <= 330) and (360 <= h <= 440):
            print("Entro")
            capacitadores_medianos.append((x, y, w, h))
        if (250 <= w <= 266) and (150 <= h <= 210):
            capacitadores_muy_pequenos.append((x, y, w, h))
        if (150 <= w <= 170) and (155 <= h <= 210):
            capacitadores_muy_pequenos.append((x, y, w, h))
        if (180 <= w <= 210) and (200 <= h <= 240):
            capacitadores_pequenos.append((x, y, w, h))
        if (260 <= w <= 270) and (235 <= h <= 245):
            capacitadores_pequenos.append((x, y, w, h))

    # Dibujar rectángulos alrededor de los capacitadores detectados
    for (x, y, w, h) in capacitadores_grades:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in capacitadores_medianos:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in capacitadores_pequenos:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in capacitadores_muy_pequenos:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
    plt.imshow(img_copy, cmap='gray')
    plt.show()
    return capacitadores_grades, capacitadores_medianos, capacitadores_pequenos, capacitadores_muy_pequenos

mascara = (3,3)
th1 = 0.12* 255
th2 = 0.8 * 255
contornos = detectar_bordes(placa,mascara, th1, th2)

chips = detectar_chips(placa,contornos)
capacitadores_grandes, capacitadores_medianos, capacitadores_pequenos, capacitadores_muy_pequenos = deteccion_capacitadores(placa,contornos)

print(f"Capacitadores grandes: {len(capacitadores_grandes)}")
print(f"Capacitadores medianos: {len(capacitadores_medianos)}")
print(f"Capacitadores pequeños: {len(capacitadores_pequenos)}")
print(f"Capacitadores muy pequeños: {len(capacitadores_muy_pequenos)}")

plt.imshow(contornos, cmap='gray')
plt.show()