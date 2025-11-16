# 🔧 Script para Convertir Modelo a TensorFlow.js

## Opción 1: Conversión desde TFLite (RECOMENDADA)

Este script convierte el modelo TFLite a formato TensorFlow.js para usar en el navegador.

```bash
#!/bin/bash

echo "🔄 Convirtiendo modelo TFLite a TensorFlow.js..."

# Instalar dependencias si no están
pip install tensorflowjs

# Convertir modelo
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    models/saved_model \
    models/tfjs_model

echo "✅ Conversión completada!"
echo "📁 Modelo guardado en: models/tfjs_model/"
echo ""
echo "Archivos generados:"
ls -lh models/tfjs_model/

echo ""
echo "🌐 Para usar en la web, copia la carpeta tfjs_model/ a tu servidor"
```

## Opción 2: Exportar Directamente a TFJS desde Python

```python
from ultralytics import YOLO
import os

# Cargar modelo entrenado
model_path = "runs/detect/inventario_salon/weights/best.pt"
model = YOLO(model_path)

print("🔄 Exportando modelo a TensorFlow.js...")

# Exportar a TFJS
tfjs_path = model.export(
    format='tfjs',
    imgsz=640,
    int8=False,  # TFJS no soporta INT8 directamente
    optimize=True
)

print(f"✅ Modelo exportado a: {tfjs_path}")

# Copiar a carpeta de modelos
import shutil
dest_dir = "models/tfjs_model"
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
shutil.copytree(tfjs_path, dest_dir)

print(f"📁 Modelo copiado a: {dest_dir}")
print("\n🌐 Ahora puedes usar el modelo en index.html")
```

## Opción 3: Usar TFLite directamente con tfjs-tflite

Instala la dependencia en tu HTML:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite"></script>
```

Luego en JavaScript:

```javascript
async function loadTFLiteModel() {
    // Cargar modelo TFLite directamente
    const tflite = await tflite.loadTFLiteModel(
        './models/inventario_yolov8n_int8.tflite'
    );
    return tflite;
}
```

## Guía de Uso según Tamaño

| Tamaño Modelo | Método Recomendado | Ubicación |
|---------------|-------------------|-----------|
| <10 MB | Incluir en Git | `models/tfjs_model/` |
| 10-50 MB | GitHub Releases | Release assets |
| >50 MB | Google Drive | Link en README |

## Estructura Final

```
inventario/
├── index.html                    # Aplicación web ✅
├── models/
│   ├── inventario_yolov8n.pt    # Modelo PyTorch
│   ├── inventario_yolov8n_int8.tflite  # TFLite optimizado
│   └── tfjs_model/               # Modelo para web
│       ├── model.json
│       └── group1-shard1of1.bin
├── 01_generar_dataset_sintetico.ipynb
├── 02_entrenar_yolov8.ipynb
└── README.md
```

## Próximos Pasos

1. ✅ Ejecuta el notebook de entrenamiento
2. ⏳ Convierte el modelo a TFJS
3. ⏳ Actualiza index.html con la carga del modelo
4. ⏳ Prueba con imágenes reales
