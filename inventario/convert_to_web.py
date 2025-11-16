"""
Script para convertir el modelo YOLOv8 a formato web-ready
Ejecutar después del entrenamiento en 02_entrenar_yolov8.ipynb
"""

import os
import shutil
from pathlib import Path

def convert_model_to_web():
    """
    Convierte el modelo entrenado a formatos optimizados para web
    """
    
    print("="*60)
    print("🔄 CONVERSIÓN DE MODELO PARA WEB")
    print("="*60)
    
    # Rutas
    best_model_path = Path("runs/detect/inventario_salon/weights/best.pt")
    models_dir = Path("models")
    
    # Verificar que existe el modelo entrenado
    if not best_model_path.exists():
        print(f"❌ Error: Modelo no encontrado en {best_model_path}")
        print("   Por favor, entrena el modelo primero (02_entrenar_yolov8.ipynb)")
        return False
    
    print(f"\n✅ Modelo encontrado: {best_model_path}")
    
    # Crear directorio de modelos
    models_dir.mkdir(exist_ok=True)
    
    try:
        from ultralytics import YOLO
        
        # Cargar modelo
        print("\n📦 Cargando modelo...")
        model = YOLO(str(best_model_path))
        
        # 1. Exportar a TFLite INT8 (para referencia)
        print("\n1️⃣ Exportando a TFLite INT8 (optimizado)...")
        tflite_int8_path = model.export(
            format='tflite',
            imgsz=640,
            int8=True,
            data='yolo_dataset/data.yaml'
        )
        print(f"   ✅ TFLite INT8: {tflite_int8_path}")
        
        # Copiar a models/
        tflite_dest = models_dir / "inventario_yolov8n_int8.tflite"
        shutil.copy2(tflite_int8_path, tflite_dest)
        print(f"   📁 Copiado a: {tflite_dest}")
        
        # 2. Exportar a TensorFlow.js
        print("\n2️⃣ Exportando a TensorFlow.js (para web)...")
        tfjs_path = model.export(
            format='tfjs',
            imgsz=640
        )
        print(f"   ✅ TensorFlow.js: {tfjs_path}")
        
        # Copiar a models/tfjs_model/
        tfjs_dest = models_dir / "tfjs_model"
        if tfjs_dest.exists():
            shutil.rmtree(tfjs_dest)
        shutil.copytree(tfjs_path, tfjs_dest)
        print(f"   📁 Copiado a: {tfjs_dest}")
        
        # 3. Copiar PyTorch model
        print("\n3️⃣ Copiando modelo PyTorch...")
        pt_dest = models_dir / "inventario_yolov8n.pt"
        shutil.copy2(best_model_path, pt_dest)
        print(f"   ✅ PyTorch: {pt_dest}")
        
        # 4. Crear archivo de labels
        print("\n4️⃣ Creando archivo de labels...")
        labels_path = models_dir / "labels.txt"
        class_names = ['cpu', 'mesa', 'mouse', 'pantalla', 'silla', 'teclado']
        with open(labels_path, 'w') as f:
            for i, name in enumerate(class_names):
                f.write(f"{i} {name}\n")
        print(f"   ✅ Labels: {labels_path}")
        
        # Mostrar tamaños
        print("\n" + "="*60)
        print("📊 TAMAÑOS DE MODELOS")
        print("="*60)
        
        def get_size_mb(path):
            if path.is_file():
                return path.stat().st_size / (1024 * 1024)
            elif path.is_dir():
                total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                return total / (1024 * 1024)
            return 0
        
        models = [
            ("PyTorch (.pt)", pt_dest),
            ("TFLite INT8", tflite_dest),
            ("TensorFlow.js", tfjs_dest),
        ]
        
        for name, path in models:
            if path.exists():
                size = get_size_mb(path)
                print(f"{name:20s}: {size:>6.2f} MB")
        
        print("\n" + "="*60)
        print("✅ CONVERSIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        
        print("\n📝 SIGUIENTE PASO:")
        print("   1. Abre index.html en un navegador")
        print("   2. Sube una imagen del salón")
        print("   3. El sistema detectará objetos automáticamente")
        
        print("\n💡 NOTA:")
        print("   - El index.html actual funciona en MODO DEMO")
        print("   - Para usar el modelo real, descomenta el código de carga en index.html")
        print("   - Busca la sección 'initializeModel()' en el script")
        
        return True
        
    except ImportError:
        print("\n❌ Error: ultralytics no está instalado")
        print("   Instala con: pip install ultralytics")
        return False
        
    except Exception as e:
        print(f"\n❌ Error durante la conversión: {e}")
        return False

def verify_web_files():
    """
    Verifica que todos los archivos necesarios para la web existan
    """
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DE ARCHIVOS WEB")
    print("="*60)
    
    files_to_check = [
        ("index.html", "Aplicación web principal"),
        ("models/tfjs_model/model.json", "Modelo TensorFlow.js"),
        ("models/inventario_yolov8n_int8.tflite", "Modelo TFLite"),
        ("models/labels.txt", "Archivo de labels"),
    ]
    
    all_ok = True
    for file_path, description in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {description:40s} → {file_path}")
        else:
            print(f"❌ {description:40s} → {file_path} (NO ENCONTRADO)")
            all_ok = False
    
    print("="*60)
    
    if all_ok:
        print("\n🎉 ¡Todos los archivos están listos!")
        print("\n🌐 Para probar la aplicación:")
        print("   1. Abre index.html en tu navegador")
        print("   2. O usa un servidor local:")
        print("      python -m http.server 8000")
        print("   3. Luego abre: http://localhost:8000")
    else:
        print("\n⚠️ Faltan algunos archivos")
        print("   Ejecuta primero los notebooks de entrenamiento")
    
    return all_ok

def create_demo_image():
    """
    Crea una imagen de demostración para probar la app
    """
    try:
        import cv2
        import numpy as np
        
        print("\n🎨 Creando imagen de demostración...")
        
        # Crear imagen simple
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        # Agregar texto
        cv2.putText(img, "IMAGEN DE PRUEBA", (150, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Guardar
        demo_path = Path("demo_image.jpg")
        cv2.imwrite(str(demo_path), img)
        
        print(f"✅ Imagen de prueba creada: {demo_path}")
        
    except ImportError:
        print("⚠️ OpenCV no disponible para crear imagen demo")

if __name__ == "__main__":
    print("\n🚀 PREPARACIÓN DE MODELO PARA APLICACIÓN WEB\n")
    
    # Convertir modelo
    success = convert_model_to_web()
    
    if success:
        # Verificar archivos web
        verify_web_files()
        
        # Crear imagen demo
        create_demo_image()
        
        print("\n" + "="*60)
        print("🎯 ¡TODO LISTO!")
        print("="*60)
        print("\nPuedes usar la aplicación web ahora.")
        print("El modelo actual funciona en MODO DEMO.")
        print("\nPara integrar el modelo real:")
        print("1. Edita index.html")
        print("2. Busca la función initializeModel()")
        print("3. Descomenta el código de carga del modelo TFJS")
        print("4. Implementa la función detectObjects() con el modelo real")
        
    else:
        print("\n❌ La conversión falló")
        print("   Verifica que hayas entrenado el modelo primero")
