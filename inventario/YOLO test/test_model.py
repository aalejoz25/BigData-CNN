"""
Script de prueba rápida para verificar el modelo YOLOv8
Ejecutar después del entrenamiento para validar funcionamiento
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def test_model():
    """Prueba rápida del modelo entrenado"""
    
    # Rutas
    model_path = "./models/inventario_yolov8n.pt"
    test_images_dir = "./yolo_dataset/val/images"
    
    print("="*60)
    print("🧪 PRUEBA RÁPIDA DEL MODELO")
    print("="*60)
    
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        print("   Por favor, entrena el modelo primero (02_entrenar_yolov8.ipynb)")
        return
    
    # Cargar modelo
    print(f"\n📦 Cargando modelo desde: {model_path}")
    model = YOLO(model_path)
    print("✅ Modelo cargado correctamente")
    
    # Obtener información del modelo
    print(f"\nℹ️  Información del modelo:")
    model.info()
    
    # Buscar imágenes de prueba
    test_images = list(Path(test_images_dir).glob("*.jpeg"))
    print(test_images)

    
    if not test_images:
        print(f"\n⚠️  No se encontraron imágenes de prueba en {test_images_dir}")
        return
    
    print(f"\n🖼️  Encontradas {len(test_images)} imágenes de prueba")
    
    # Probar con la primera imagen
    test_img = test_images[0]
    print(f"\n🔍 Probando con: {test_img.name}")
    
    # Realizar predicción
    results = model.predict(
        source=str(test_img),
        conf=0.25,  # Umbral de confianza
        save=True,  # Guardar resultado
        project="./test_results",
        name="prediction"
    )
    
    # Mostrar resultados
    result = results[0]
    num_detections = len(result.boxes)
    
    print(f"\n✅ Predicción completada!")
    print(f"   - Objetos detectados: {num_detections}")
    
    if num_detections > 0:
        print(f"\n📊 Detecciones:")
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"   {i+1}. {class_name}: {confidence:.2%}")
    
    print(f"\n💾 Resultado guardado en: ./test_results/prediction/")
    print("\n" + "="*60)
    print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
    print("="*60)

if __name__ == "__main__":
    test_model()
