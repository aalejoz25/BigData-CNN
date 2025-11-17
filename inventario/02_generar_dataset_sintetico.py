#!/usr/bin/env python3
"""
Generador de Dataset Sintético con Múltiples Objetos por Imagen
Técnica: Copy-Paste con variaciones aleatorias

Este script resuelve el problema de que el modelo solo detecta 1 objeto por imagen.
Genera imágenes sintéticas con 2-8 objetos por imagen, con variaciones de:
- Posición, escala, rotación
- Iluminación y color
- Solapes parciales
- Fondos variados

Autor: Álvaro Zarabanda - 20251595006
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import albumentations as A
from collections import defaultdict
import shutil

# ==================== CONFIGURACIÓN ====================

PROCESSED_DIR = Path("../objetos-salon/processed")
OUTPUT_DIR = Path("dataset_synthetic")
BACKGROUNDS_DIR = Path("backgrounds")  # Opcional: fondos personalizados

# Parámetros de generación
NUM_TRAIN_IMAGES = 3000  # Número de imágenes de entrenamiento a generar
NUM_VAL_IMAGES = 750     # Número de imágenes de validación
MIN_OBJECTS_PER_IMAGE = 2
MAX_OBJECTS_PER_IMAGE = 8
IMG_SIZE = 640

# Clases (sin 'nada')
CLASSES = {
    'cpu': 0,
    'mesa': 1,
    'mouse': 2,
    'pantalla': 3,
    'silla': 4,
    'teclado': 5
}

# Probabilidad de cada clase por imagen (simula distribución de salón real)
CLASS_PROBABILITIES = {
    'cpu': 0.8,       # 80% de las imágenes tendrán CPUs
    'mesa': 0.7,      # 70% tendrán mesas
    'mouse': 0.85,    # 85% tendrán mouses
    'pantalla': 0.8,  # 80% tendrán pantallas
    'silla': 0.9,     # 90% tendrán sillas
    'teclado': 0.85   # 85% tendrán teclados
}

# ==================== FUNCIONES AUXILIARES ====================

def load_object_with_mask(image_path):
    """
    Carga un objeto y extrae su máscara usando segmentación automática
    
    Returns:
        object_img: Imagen del objeto (con canal alpha si es posible)
        mask: Máscara binaria del objeto
        bbox: Bounding box del objeto [x, y, w, h]
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    
    # Convertir a escala de grises para detección
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Usar Canny + dilatación para encontrar el objeto
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Si no encuentra contornos, usar toda la imagen
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        h, w = img.shape[:2]
        bbox = [0, 0, w, h]
    else:
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Crear máscara
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Aplicar morfología para suavizar
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = [x, y, w, h]
    
    return img, mask, bbox


def extract_object(img, mask, bbox, add_padding=10):
    """
    Extrae el objeto con su máscara y lo recorta
    
    Returns:
        object_crop: Imagen recortada del objeto
        mask_crop: Máscara recortada
    """
    x, y, w, h = bbox
    
    # Agregar padding
    x1 = max(0, x - add_padding)
    y1 = max(0, y - add_padding)
    x2 = min(img.shape[1], x + w + add_padding)
    y2 = min(img.shape[0], y + h + add_padding)
    
    object_crop = img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    
    return object_crop, mask_crop


def augment_object(obj_img, obj_mask):
    """
    Aplica augmentaciones al objeto individual
    """
    transform = A.Compose([
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ], additional_targets={'mask': 'mask'})
    
    augmented = transform(image=obj_img, mask=obj_mask)
    return augmented['image'], augmented['mask']


def create_background(size=640):
    """
    Crea un fondo aleatorio que simula diferentes superficies/escenarios
    """
    bg_type = random.choice(['solid', 'gradient', 'texture', 'noise'])
    
    if bg_type == 'solid':
        # Color sólido (tonos grises, blancos, beige)
        color = random.randint(200, 250)
        background = np.full((size, size, 3), color, dtype=np.uint8)
        
    elif bg_type == 'gradient':
        # Gradiente suave
        background = np.zeros((size, size, 3), dtype=np.uint8)
        c1 = random.randint(180, 240)
        c2 = random.randint(200, 255)
        for i in range(size):
            val = int(c1 + (c2 - c1) * i / size)
            background[i, :] = val
            
    elif bg_type == 'texture':
        # Textura simulada (ruido estructurado)
        base = random.randint(200, 230)
        background = np.full((size, size, 3), base, dtype=np.uint8)
        noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        background = cv2.GaussianBlur(background, (5, 5), 0)
        
    else:  # noise
        # Ruido gaussiano
        mean = random.randint(210, 240)
        background = np.random.normal(mean, 15, (size, size, 3)).astype(np.uint8)
        background = cv2.GaussianBlur(background, (7, 7), 0)
    
    return background


def paste_object_on_background(background, obj_img, obj_mask, position, scale=1.0):
    """
    Pega un objeto en el fondo en la posición indicada
    
    Args:
        background: Imagen de fondo
        obj_img: Imagen del objeto
        obj_mask: Máscara del objeto
        position: (x, y) posición donde pegar (centro del objeto)
        scale: Factor de escala del objeto
    
    Returns:
        background_updated: Fondo con el objeto pegado
        bbox: [x_center, y_center, width, height] en coordenadas absolutas
    """
    h, w = obj_img.shape[:2]
    
    # Escalar objeto si es necesario
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        obj_img = cv2.resize(obj_img, (new_w, new_h))
        obj_mask = cv2.resize(obj_mask, (new_w, new_h))
        h, w = new_h, new_w
    
    # Calcular posición de pegado (centro del objeto en position)
    x_center, y_center = position
    x1 = x_center - w // 2
    y1 = y_center - h // 2
    x2 = x1 + w
    y2 = y1 + h
    
    # Verificar límites
    bg_h, bg_w = background.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > bg_w or y2 > bg_h:
        # Fuera de límites, ajustar
        x1 = max(0, min(x1, bg_w - w))
        y1 = max(0, min(y1, bg_h - h))
        x2 = x1 + w
        y2 = y1 + h
        
        if x2 > bg_w or y2 > bg_h:
            return background, None  # No cabe, skip
    
    # Preparar región del fondo
    bg_region = background[y1:y2, x1:x2].copy()
    
    # Normalizar máscara
    mask_normalized = obj_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    # Mezclar objeto con fondo usando máscara
    blended = (obj_img * mask_3ch + bg_region * (1 - mask_3ch)).astype(np.uint8)
    
    # Pegar en el fondo
    background[y1:y2, x1:x2] = blended
    
    # Calcular bbox en formato YOLO (x_center, y_center, width, height) absolutos
    bbox = [x_center, y_center, w, h]
    
    return background, bbox


def check_overlap(bbox1, bbox2, max_overlap=0.5):
    """
    Verifica si dos bounding boxes se solapan demasiado
    
    Args:
        bbox1, bbox2: [x_center, y_center, width, height]
        max_overlap: Máximo IoU permitido (0.5 = 50%)
    
    Returns:
        True si el overlap es aceptable, False si se solapan demasiado
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convertir a formato xyxy
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # Calcular intersección
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_min >= x_inter_max or y_inter_min >= y_inter_max:
        return True  # No se solapan
    
    # Área de intersección
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Áreas de cada bbox
    area1 = w1 * h1
    area2 = w2 * h2
    
    # IoU
    iou = inter_area / (area1 + area2 - inter_area)
    
    return iou <= max_overlap


def generate_synthetic_image(objects_dict, output_size=640):
    """
    Genera una imagen sintética con múltiples objetos
    
    Args:
        objects_dict: Dict con objetos cargados por clase
        output_size: Tamaño de salida
    
    Returns:
        image: Imagen sintética generada
        annotations: Lista de anotaciones YOLO [(class_id, x, y, w, h), ...]
    """
    # Crear fondo
    background = create_background(output_size)
    
    # Determinar qué clases incluir en esta imagen
    selected_classes = []
    for class_name, prob in CLASS_PROBABILITIES.items():
        if random.random() < prob:
            selected_classes.append(class_name)
    
    if not selected_classes:
        # Al menos una clase
        selected_classes = [random.choice(list(CLASSES.keys()))]
    
    # Determinar número de objetos
    num_objects = random.randint(MIN_OBJECTS_PER_IMAGE, MAX_OBJECTS_PER_IMAGE)
    
    # Seleccionar objetos aleatoriamente (respetando las clases seleccionadas)
    objects_to_paste = []
    for _ in range(num_objects):
        class_name = random.choice(selected_classes)
        if class_name in objects_dict and objects_dict[class_name]:
            obj_data = random.choice(objects_dict[class_name])
            objects_to_paste.append((class_name, obj_data))
    
    # Ordenar por tamaño (grandes primero para mejor composición)
    objects_to_paste.sort(key=lambda x: x[1]['size'], reverse=True)
    
    annotations = []
    placed_bboxes = []
    
    for class_name, obj_data in objects_to_paste:
        # Aplicar augmentaciones
        aug_img, aug_mask = augment_object(obj_data['image'].copy(), obj_data['mask'].copy())
        
        # Determinar escala aleatoria
        scale = random.uniform(0.4, 1.2)
        
        # Intentar colocar el objeto (máximo 10 intentos)
        placed = False
        for attempt in range(10):
            # Posición aleatoria
            margin = int(output_size * 0.1)  # 10% de margen
            x = random.randint(margin, output_size - margin)
            y = random.randint(margin, output_size - margin)
            
            # Crear bbox temporal
            temp_h, temp_w = aug_img.shape[:2]
            temp_w_scaled = int(temp_w * scale)
            temp_h_scaled = int(temp_h * scale)
            temp_bbox = [x, y, temp_w_scaled, temp_h_scaled]
            
            # Verificar overlap con objetos ya colocados
            overlap_ok = True
            for placed_bbox in placed_bboxes:
                if not check_overlap(temp_bbox, placed_bbox, max_overlap=0.3):
                    overlap_ok = False
                    break
            
            if overlap_ok:
                # Pegar objeto
                background, bbox = paste_object_on_background(
                    background, aug_img, aug_mask, (x, y), scale
                )
                
                if bbox is not None:
                    placed_bboxes.append(bbox)
                    
                    # Convertir bbox a formato YOLO normalizado
                    x_center, y_center, width, height = bbox
                    x_norm = x_center / output_size
                    y_norm = y_center / output_size
                    w_norm = width / output_size
                    h_norm = height / output_size
                    
                    class_id = CLASSES[class_name]
                    annotations.append((class_id, x_norm, y_norm, w_norm, h_norm))
                    placed = True
                    break
        
        if not placed:
            # No se pudo colocar después de 10 intentos
            pass
    
    # Aplicar augmentaciones globales a la imagen completa
    global_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    ])
    
    background = global_transform(image=background)['image']
    
    return background, annotations


def load_all_objects():
    """
    Carga todos los objetos del dataset original con sus máscaras
    """
    objects_dict = defaultdict(list)
    
    print("📦 Cargando objetos del dataset original...")
    
    for class_name, class_id in CLASSES.items():
        class_dir = PROCESSED_DIR / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Directorio no encontrado: {class_dir}")
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Cargando {class_name}", leave=False):
            obj_img, obj_mask, obj_bbox = load_object_with_mask(img_path)
            
            if obj_img is not None:
                # Extraer objeto
                obj_crop, mask_crop = extract_object(obj_img, obj_mask, obj_bbox)
                
                objects_dict[class_name].append({
                    'image': obj_crop,
                    'mask': mask_crop,
                    'size': obj_crop.shape[0] * obj_crop.shape[1],  # Para ordenar
                    'original_path': img_path
                })
        
        print(f"   ✅ {class_name}: {len(objects_dict[class_name])} objetos cargados")
    
    return objects_dict


def save_yolo_format(image, annotations, output_path, label_path):
    """
    Guarda la imagen y las anotaciones en formato YOLO
    """
    # Guardar imagen
    cv2.imwrite(str(output_path), image)
    
    # Guardar anotaciones
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_id, x, y, w, h = ann
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def create_data_yaml():
    """
    Crea el archivo data.yaml para YOLO
    """
    yaml_content = f"""# Dataset Sintético - Inventario Salón de Cómputo
# Generado automáticamente con múltiples objetos por imagen

path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(CLASSES)}
names: {list(CLASSES.keys())}
"""
    
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Archivo data.yaml creado en {OUTPUT_DIR / 'data.yaml'}")


def main():
    """
    Función principal para generar el dataset sintético
    """
    print("=" * 70)
    print("🎨 GENERADOR DE DATASET SINTÉTICO CON MÚLTIPLES OBJETOS")
    print("=" * 70)
    print(f"\n📊 Configuración:")
    print(f"   - Imágenes de entrenamiento: {NUM_TRAIN_IMAGES}")
    print(f"   - Imágenes de validación: {NUM_VAL_IMAGES}")
    print(f"   - Objetos por imagen: {MIN_OBJECTS_PER_IMAGE}-{MAX_OBJECTS_PER_IMAGE}")
    print(f"   - Tamaño de imagen: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   - Clases: {len(CLASSES)}")
    print()
    
    # Crear estructura de directorios
    (OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Cargar todos los objetos
    objects_dict = load_all_objects()
    
    total_objects = sum(len(objs) for objs in objects_dict.values())
    print(f"\n✅ Total de objetos únicos cargados: {total_objects}")
    print()
    
    # Generar imágenes de entrenamiento
    print(f"🔨 Generando {NUM_TRAIN_IMAGES} imágenes de entrenamiento...")
    stats_train = defaultdict(int)
    
    for i in tqdm(range(NUM_TRAIN_IMAGES)):
        # Generar imagen sintética
        image, annotations = generate_synthetic_image(objects_dict, IMG_SIZE)
        
        # Contar objetos por clase
        for ann in annotations:
            class_id = ann[0]
            class_name = list(CLASSES.keys())[class_id]
            stats_train[class_name] += 1
        
        # Guardar
        img_name = f"synthetic_train_{i:05d}.jpg"
        save_yolo_format(
            image,
            annotations,
            OUTPUT_DIR / "images" / "train" / img_name,
            OUTPUT_DIR / "labels" / "train" / f"synthetic_train_{i:05d}.txt"
        )
    
    print(f"✅ Entrenamiento completado: {NUM_TRAIN_IMAGES} imágenes")
    print(f"   Distribución de objetos:")
    for class_name, count in sorted(stats_train.items()):
        print(f"      - {class_name}: {count} instancias")
    print()
    
    # Generar imágenes de validación
    print(f"🔨 Generando {NUM_VAL_IMAGES} imágenes de validación...")
    stats_val = defaultdict(int)
    
    for i in tqdm(range(NUM_VAL_IMAGES)):
        image, annotations = generate_synthetic_image(objects_dict, IMG_SIZE)
        
        for ann in annotations:
            class_id = ann[0]
            class_name = list(CLASSES.keys())[class_id]
            stats_val[class_name] += 1
        
        img_name = f"synthetic_val_{i:05d}.jpg"
        save_yolo_format(
            image,
            annotations,
            OUTPUT_DIR / "images" / "val" / img_name,
            OUTPUT_DIR / "labels" / "val" / f"synthetic_val_{i:05d}.txt"
        )
    
    print(f"✅ Validación completada: {NUM_VAL_IMAGES} imágenes")
    print(f"   Distribución de objetos:")
    for class_name, count in sorted(stats_val.items()):
        print(f"      - {class_name}: {count} instancias")
    print()
    
    # Crear data.yaml
    create_data_yaml()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("🎉 DATASET SINTÉTICO GENERADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\n📁 Ubicación: {OUTPUT_DIR.absolute()}")
    print(f"\n📊 Estadísticas totales:")
    print(f"   - Total imágenes entrenamiento: {NUM_TRAIN_IMAGES}")
    print(f"   - Total imágenes validación: {NUM_VAL_IMAGES}")
    print(f"   - Total instancias (train): {sum(stats_train.values())}")
    print(f"   - Total instancias (val): {sum(stats_val.values())}")
    print(f"   - Promedio objetos/imagen (train): {sum(stats_train.values())/NUM_TRAIN_IMAGES:.1f}")
    print(f"   - Promedio objetos/imagen (val): {sum(stats_val.values())/NUM_VAL_IMAGES:.1f}")
    print()
    print("🚀 Siguiente paso:")
    print(f"   Entrena el modelo con: python 05_entrenar_yolo_sintetico.py")
    print(f"   O usa el notebook: 05_entrenar_yolo_sintetico.ipynb")
    print()


if __name__ == "__main__":
    main()
