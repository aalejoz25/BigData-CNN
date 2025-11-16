# Plan de Implementación - Sistema de Inventario

## 🎯 Objetivo
Crear aplicación web que detecte y cuente objetos del salón de cómputo

## 📊 Opciones Evaluadas

### Opción 1: YOLOv8 Nano (RECOMENDADA) ⭐
- **Tamaño**: ~6 MB (TFLite)
- **Velocidad**: Muy rápida
- **Precisión**: Alta para detección
- **Trabajo requerido**: Etiquetar ~50-100 imágenes con bounding boxes

### Opción 2: Clasificador + Sliding Window
- **Tamaño**: ~2-3 MB (tu modelo actual)
- **Velocidad**: Lenta (múltiples inferencias)
- **Precisión**: Media-Baja (muchos falsos positivos)
- **Trabajo requerido**: Implementar lógica de ventanas

### Opción 3: SSD MobileNet
- **Tamaño**: ~15-20 MB
- **Velocidad**: Media
- **Precisión**: Alta
- **Trabajo requerido**: Etiquetar imágenes + entrenar

## ✅ DECISIÓN: YOLOv8 Nano

### Razones:
1. Mejor balance entre tamaño y precisión
2. Detección real de objetos (no clasificación)
3. Entrenamiento rápido (~30 min en GPU)
4. Fácil conversión a TFLite y TensorFlow.js
5. Código simple para web

---

## 📝 Plan de Acción (4-6 horas total)

### Fase 1: Preparación de Datos (1-2 horas)
```
1. Tomar ~20 fotos del salón con múltiples objetos
2. Etiquetar con Roboflow (automático + corrección manual)
3. Exportar en formato YOLO
4. Aumentar dataset a ~100 imágenes (Roboflow automático)
```

**Herramientas:**
- Roboflow.com (GRATIS, 10,000 imágenes/mes)
- Alternativa: LabelImg (offline)

### Fase 2: Entrenamiento YOLOv8n (1 hora)
```python
from ultralytics import YOLO

# Cargar modelo nano pre-entrenado
model = YOLO('yolov8n.pt')

# Entrenar con tus datos (30-50 epochs)
model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU
)

# Exportar a TFLite
model.export(format='tflite', int8=True)  # Cuantización INT8
```

**Resultado esperado:**
- Modelo TFLite: ~3-6 MB
- mAP50: >0.70 (bueno para este caso)

### Fase 3: Conversión para Web (30 min)
```bash
# Opción A: TensorFlow.js (recomendado)
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    model_saved/ \
    model_web/

# Opción B: TFLite + WASM
# Usar directamente el .tflite con tfjs-tflite
```

### Fase 4: Aplicación Web (1-2 horas)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Inventario Salón</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
</head>
<body>
    <h1>🖥️ Inventario Automático</h1>
    
    <input type="file" id="imageUpload" accept="image/*">
    <canvas id="canvas"></canvas>
    
    <div id="results">
        <h2>📊 Resultados:</h2>
        <ul id="counts"></ul>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

**Funcionalidades:**
1. Subir imagen JPG
2. Detectar objetos con YOLOv8
3. Dibujar bounding boxes con números azules
4. Mostrar conteo por categoría

---

## 🎨 Formato de Salida Requerido

```javascript
// Ejemplo de output
{
    detections: [
        {class: 0, label: "CPU", bbox: [x, y, w, h], confidence: 0.92},
        {class: 3, label: "Pantalla", bbox: [x, y, w, h], confidence: 0.88},
        // ...
    ],
    counts: {
        "CPU": 2,
        "Mesa": 1,
        "Mouse": 3,
        "Pantalla": 2,
        "Silla": 4,
        "Teclado": 3
    }
}
```

---

## 📦 Estructura de Archivos

```
inventario/
├── index.html              # Aplicación principal
├── app.js                  # Lógica de detección
├── styles.css              # Estilos
├── model/                  # Modelos convertidos
│   ├── model.json          # TensorFlow.js
│   ├── group1-shard1of1.bin
│   └── labels.json         # Mapeo de clases
├── README.md               # Documentación + link Drive
└── examples/               # Imágenes de prueba
    └── salon_test.jpg
```

---

## ⚡ Optimizaciones para Menor Tamaño

### 1. Cuantización
```python
# Cuantización INT8 (reduce 4x el tamaño)
model.export(format='tflite', int8=True)
```

### 2. Pruning (Poda de conexiones)
```python
import tensorflow_model_optimization as tfmot

# Aplicar pruning antes de exportar
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=0,
    end_step=1000
)
```

### 3. Usar YOLOv8n en lugar de YOLOv8s
```
YOLOv8n:  6 MB  (nano)    ⭐ USAR ESTE
YOLOv8s: 22 MB  (small)
YOLOv8m: 52 MB  (medium)
```

---

## 🧪 Testing

### Dataset de prueba:
1. Foto del salón completo (múltiples objetos)
2. Foto de escritorio individual
3. Foto con objetos parcialmente ocultos
4. Foto con iluminación diferente

### Métricas esperadas:
- **Precisión**: >85% en objetos visibles
- **Recall**: >80% (detecta la mayoría)
- **Velocidad**: <500ms por imagen en navegador
- **Tamaño**: <10 MB total (modelo + assets)

---

## 🚀 Siguiente Paso INMEDIATO

**¿Quieres que te ayude a:**

### A) Empezar con YOLOv8 (recomendado)
- Crear notebook de entrenamiento
- Preparar estructura de datos
- Script de etiquetado rápido

### B) Implementar Sliding Window con tu modelo actual
- Convertir tu mejor modelo a TFLite
- Implementar detección por ventanas
- Crear la app web

### C) Explorar alternativa híbrida
- Usar modelo ligero de detección genérica
- Aplicar tu clasificador para refinar
- Mejor balance rendimiento/tamaño

## 💰 Estimación de Nota por Opción

| Opción | Detección (40%) | Tamaño (40%) | App Web (15%) | TOTAL |
|--------|-----------------|--------------|---------------|-------|
| YOLOv8n INT8 | 38/40 (95%) | 35/40 (~6MB) | 14/15 | **87/95** |
| Sliding Window | 32/40 (80%) | 40/40 (~2MB) | 13/15 | **85/95** |
| SSD MobileNet | 39/40 (98%) | 28/40 (~18MB)| 14/15 | **81/95** |

**Recomendación: YOLOv8n con cuantización INT8**

---

¿Qué opción prefieres? Te ayudo a implementarla paso a paso.
