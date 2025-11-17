"""
Script para generar bounding boxes automáticamente desde las imágenes procesadas
Usa segmentación y detección de contornos con OpenCV

Autor: Álvaro Zarabanda - 20251595006
"""

import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

# Configuración
PROCESSED_DIR = "../objetos-salon/processed"
OUTPUT_DIR = "dataset"
IMG_SIZE = (640, 640)  # Tamaño para YOLO

# Mapeo de clases (sin 'nada' porque no es un objeto a detectar)
CLASSES = {
    'cpu': 0,
    'mesa': 1,
    'mouse': 2,
    'pantalla': 3,
    'silla': 4,
    'teclado': 5
}

def detect_object_bbox(image_path, method='contour', padding=0.05):
    """
    Detecta automáticamente el objeto principal en la imagen y retorna su bounding box
    
    Args:
        image_path: Ruta a la imagen
        method: Método de detección ('contour', 'edge', 'grabcut')
        padding: Porcentaje de padding a agregar al bbox (0.05 = 5%)
    
    Returns:
        (x_center, y_center, width, height) normalizados [0-1] o None si falla
    """
    
    # Leer imagen
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error leyendo: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Método 1: Detección por contornos (funciona bien con fondos uniformes)
    if method == 'contour':
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilatar para conectar bordes cercanos
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Si no encuentra contornos, usar toda la imagen
            x, y, bw, bh = 0, 0, w, h
        else:
            # Obtener el contorno más grande (asumiendo que es el objeto principal)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    # Método 2: Por umbralización adaptativa
    elif method == 'threshold':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Umbralización adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            x, y, bw, bh = 0, 0, w, h
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    # Método 3: Por diferencia con fondo
    elif method == 'background':
        # Calcular el color más frecuente en los bordes (probablemente el fondo)
        border_pixels = np.concatenate([
            img[0, :, :].reshape(-1, 3),
            img[-1, :, :].reshape(-1, 3),
            img[:, 0, :].reshape(-1, 3),
            img[:, -1, :].reshape(-1, 3)
        ])
        
        # Color promedio del borde
        bg_color = np.median(border_pixels, axis=0)
        
        # Calcular diferencia con el fondo
        diff = np.abs(img.astype(float) - bg_color).sum(axis=2)
        
        # Umbralizar
        thresh = (diff > 30).astype(np.uint8) * 255
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            x, y, bw, bh = 0, 0, w, h
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    else:
        # Por defecto, usar toda la imagen
        x, y, bw, bh = 0, 0, w, h
    
    # Agregar padding
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    bw = min(w - x, bw + 2 * pad_w)
    bh = min(h - y, bh + 2 * pad_h)
    
    # Convertir a formato YOLO (normalizado, centro)
    x_center = (x + bw / 2) / w
    y_center = (y + bh / 2) / h
    width = bw / w
    height = bh / h
    
    # Validar que los valores estén en rango [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        print(f"⚠️  Bbox fuera de rango: {image_path}")
        return None
    
    return (x_center, y_center, width, height)


def create_yolo_dataset(processed_dir, output_dir, method='contour', train_split=0.8):
    """
    Crea dataset en formato YOLO desde las imágenes procesadas
    
    Args:
        processed_dir: Directorio con las carpetas de clases
        output_dir: Directorio de salida para el dataset YOLO
        method: Método de detección de bbox
        train_split: Porcentaje para entrenamiento (0.8 = 80% train, 20% val)
    """
    
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    
    # Crear estructura de directorios YOLO
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print("🚀 Generando dataset YOLO automáticamente...\n")
    
    stats = {cls: {'train': 0, 'val': 0, 'failed': 0} for cls in CLASSES.keys()}
    
    # Procesar cada clase
    for class_name, class_id in CLASSES.items():
        class_dir = processed_path / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Carpeta no encontrada: {class_dir}")
            continue
        
        # Obtener todas las imágenes
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠️  No hay imágenes en: {class_dir}")
            continue
        
        print(f"\n📦 Procesando clase '{class_name}' ({len(image_files)} imágenes)...")
        
        # Mezclar imágenes aleatoriamente
        np.random.shuffle(image_files)
        
        # Dividir en train/val
        split_idx = int(len(image_files) * train_split)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Procesar imágenes de entrenamiento
        for img_path in tqdm(train_files, desc=f"  Train {class_name}"):
            bbox = detect_object_bbox(img_path, method=method)
            
            if bbox is None:
                stats[class_name]['failed'] += 1
                continue
            
            # Leer y redimensionar imagen
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, IMG_SIZE)
            
            # Guardar imagen
            img_filename = f"{class_name}_{img_path.stem}.jpg"
            img_output_path = output_path / "images" / "train" / img_filename
            cv2.imwrite(str(img_output_path), img_resized)
            
            # Guardar label
            label_filename = f"{class_name}_{img_path.stem}.txt"
            label_output_path = output_path / "labels" / "train" / label_filename
            
            with open(label_output_path, 'w') as f:
                x_c, y_c, w, h = bbox
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            stats[class_name]['train'] += 1
        
        # Procesar imágenes de validación
        for img_path in tqdm(val_files, desc=f"  Val   {class_name}"):
            bbox = detect_object_bbox(img_path, method=method)
            
            if bbox is None:
                stats[class_name]['failed'] += 1
                continue
            
            # Leer y redimensionar imagen
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, IMG_SIZE)
            
            # Guardar imagen
            img_filename = f"{class_name}_{img_path.stem}.jpg"
            img_output_path = output_path / "images" / "val" / img_filename
            cv2.imwrite(str(img_output_path), img_resized)
            
            # Guardar label
            label_filename = f"{class_name}_{img_path.stem}.txt"
            label_output_path = output_path / "labels" / "val" / label_filename
            
            with open(label_output_path, 'w') as f:
                x_c, y_c, w, h = bbox
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            stats[class_name]['val'] += 1
    
    # Mostrar estadísticas
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL DATASET")
    print("="*60)
    
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())
    
    print(f"\n{'Clase':<12} {'Train':<10} {'Val':<10} {'Failed':<10}")
    print("-" * 60)
    for class_name in CLASSES.keys():
        print(f"{class_name:<12} {stats[class_name]['train']:<10} "
              f"{stats[class_name]['val']:<10} {stats[class_name]['failed']:<10}")
    
    print("-" * 60)
    print(f"{'TOTAL':<12} {total_train:<10} {total_val:<10} {total_failed:<10}")
    print(f"\n✅ Dataset generado en: {output_path}")
    
    # Crear archivo data.yaml
    yaml_content = f"""# Dataset de objetos del salón
# Auto-generado con bounding boxes automáticos

path: {output_path.absolute()}
train: images/train
val: images/val

# Clases
nc: {len(CLASSES)}
names: {list(CLASSES.keys())}
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Archivo de configuración: {yaml_path}\n")
    
    return stats


if __name__ == "__main__":
    print("="*60)
    print("  GENERADOR AUTOMÁTICO DE BOUNDING BOXES PARA YOLO")
    print("="*60)
    
    # Crear dataset
    stats = create_yolo_dataset(
        processed_dir=PROCESSED_DIR,
        output_dir=OUTPUT_DIR,
        method='contour',  # Opciones: 'contour', 'threshold', 'background'
        train_split=0.8
    )
    
    print("\n🎉 ¡Proceso completado!")
    print("\n📝 Próximos pasos:")
    print("   1. Revisar algunas imágenes en dataset/images/train/")
    print("   2. Verificar las etiquetas en dataset/labels/train/")
    print("   3. Entrenar YOLO con: python 02_entrenar_yolo.py")
