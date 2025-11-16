# 📦 Guía de Exportación de Modelos

## Formatos Soportados por YOLOv8

### 1. **TensorFlow Lite (TFLite)** ⭐ RECOMENDADO PARA WEB

```python
# Float32 (sin optimización)
model.export(format='tflite')

# INT8 (cuantizado - MÁS PEQUEÑO)
model.export(format='tflite', int8=True)

# Float16 (intermedio)
model.export(format='tflite', half=True)
```

**Tamaños:**
- Float32: ~12 MB
- INT8: ~3-4 MB ⭐ Mejor balance
- Float16: ~6 MB

**Uso:** JavaScript con TensorFlow.js o TFLite runtime

---

### 2. **TensorFlow SavedModel**

```python
model.export(format='saved_model')
```

**Tamaño:** ~15-20 MB
**Uso:** Servidor Python, TensorFlow Serving

---

### 3. **TensorFlow.js** (Directo para web)

```python
model.export(format='tfjs')
```

**Genera:** Carpeta con model.json + shards
**Tamaño:** ~8-10 MB (shards múltiples)
**Uso:** Directamente en navegador con @tensorflow/tfjs

---

### 4. **ONNX** (Universal)

```python
model.export(format='onnx', simplify=True)
```

**Tamaño:** ~6 MB
**Uso:** ONNX Runtime (multi-plataforma)

---

### 5. **CoreML** (iOS/macOS)

```python
model.export(format='coreml')
```

**Uso:** Apps de Apple

---

### 6. **OpenVINO** (Intel)

```python
model.export(format='openvino')
```

**Uso:** Hardware Intel optimizado

---

## 🎯 Recomendaciones por Caso de Uso

### Para Aplicación Web HTML/JS:

**Opción A: TFLite INT8** (RECOMENDADA)
```python
tflite_path = model.export(format='tflite', int8=True, imgsz=640)
```
- ✅ Más pequeño (~3-4 MB)
- ✅ Más rápido
- ✅ Compatible con TensorFlow.js
- ⚠️ Requiere conversión adicional a tfjs

**Opción B: TensorFlow.js directo**
```python
tfjs_path = model.export(format='tfjs', imgsz=640)
```
- ✅ Listo para usar en navegador
- ✅ No requiere conversión
- ❌ Más grande (~8-10 MB)

---

## 📊 Comparación de Tamaños

| Formato | Tamaño | Velocidad | Compatibilidad Web |
|---------|--------|-----------|-------------------|
| PyTorch .pt | ~6 MB | Rápido | ❌ No |
| TFLite Float32 | ~12 MB | Medio | ✅ Con conversión |
| TFLite INT8 | ~3-4 MB | Rápido | ✅ Con conversión |
| TFLite Float16 | ~6 MB | Rápido | ✅ Con conversión |
| TensorFlow.js | ~8-10 MB | Medio | ✅ Directo |
| ONNX | ~6 MB | Rápido | ✅ Con ONNX.js |
| SavedModel | ~15-20 MB | Rápido | ❌ Solo servidor |

---

## 💾 Opciones para Almacenamiento

### Opción 1: Incluir en el Repositorio (Si es pequeño)

```bash
# Si el modelo es <10 MB
git add inventario/models/inventario_yolov8n_int8.tflite
git commit -m "Add optimized detection model"
git push
```

### Opción 2: Google Drive (Si es grande)

```bash
# Subir a Google Drive y compartir link público
# En README.md:
```

**Link del modelo:** [Descargar desde Google Drive](https://drive.google.com/...)

### Opción 3: Git LFS (Large File Storage)

```bash
# Para archivos grandes en Git
git lfs install
git lfs track "*.tflite"
git lfs track "*.pt"
git add .gitattributes
```

### Opción 4: GitHub Releases

```bash
# Crear release con el modelo como asset
gh release create v1.0 \
  models/inventario_yolov8n_int8.tflite \
  --title "Modelo de Inventario v1.0"
```

---

## 🔄 Conversión TFLite → TensorFlow.js

Si exportas a TFLite pero quieres usar en web:

```bash
# Instalar convertidor
pip install tensorflowjs

# Convertir TFLite a TFJS
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    saved_model/ \
    tfjs_model/

# O desde TFLite directamente
tensorflowjs_converter \
    --input_format=keras_saved_model \
    model.tflite \
    tfjs_model/
```

---

## 📝 Código Completo de Exportación

### En el notebook (ya incluido):

```python
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO('runs/detect/inventario_salon/weights/best.pt')

# Exportar a TODOS los formatos útiles
print("🔄 Exportando modelo a múltiples formatos...\n")

# 1. TFLite INT8 (más pequeño) ⭐
print("1. TFLite INT8...")
tflite_int8 = model.export(
    format='tflite',
    imgsz=640,
    int8=True,
    data='yolo_dataset/data.yaml'  # Necesario para calibración INT8
)
print(f"   ✅ Guardado: {tflite_int8}")

# 2. TFLite Float32 (alternativa sin cuantización)
print("\n2. TFLite Float32...")
tflite_float = model.export(
    format='tflite',
    imgsz=640
)
print(f"   ✅ Guardado: {tflite_float}")

# 3. TensorFlow.js (listo para web)
print("\n3. TensorFlow.js...")
tfjs_model = model.export(
    format='tfjs',
    imgsz=640
)
print(f"   ✅ Guardado: {tfjs_model}")

# 4. ONNX (portable)
print("\n4. ONNX...")
onnx_model = model.export(
    format='onnx',
    imgsz=640,
    simplify=True
)
print(f"   ✅ Guardado: {onnx_model}")

# 5. SavedModel (para TensorFlow)
print("\n5. TensorFlow SavedModel...")
saved_model = model.export(
    format='saved_model',
    imgsz=640
)
print(f"   ✅ Guardado: {saved_model}")

print("\n✅ Todos los modelos exportados!")
```

---

## 🎯 Para Tu Proyecto Específico

### Recomendación FINAL:

**Usar TFLite INT8** por estas razones:

1. ✅ **Tamaño más pequeño** (~3-4 MB) → Mejor nota (40%)
2. ✅ **Velocidad rápida** en navegador
3. ✅ **Compatible con TensorFlow.js**
4. ✅ **Pérdida mínima de precisión** (<2%)

### Almacenamiento:

**Si <10 MB:** Incluir en repositorio directamente

```bash
git add inventario/models/inventario_yolov8n_int8.tflite
```

**Si >10 MB:** Subir a Google Drive

```markdown
# En README.md
## 📦 Modelo Entrenado

Debido al tamaño del archivo, el modelo está disponible en Google Drive:

🔗 [Descargar modelo TFLite INT8 (3.5 MB)](https://drive.google.com/file/d/...)

### Instrucciones:
1. Descargar el archivo `.tflite`
2. Colocar en `inventario/models/`
3. La aplicación web lo cargará automáticamente
```

---

## 🔍 Verificar Tamaño del Modelo

```python
import os

def get_size_mb(filepath):
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

# Verificar tamaños
models = {
    'PyTorch': 'best.pt',
    'TFLite INT8': 'best_int8.tflite',
    'TFLite Float32': 'best_float32.tflite',
    'TFJS': 'best_web_model/',
}

for name, path in models.items():
    if os.path.exists(path):
        if os.path.isfile(path):
            size = get_size_mb(path)
            print(f"{name:20s}: {size:.2f} MB")
        else:
            # Para carpetas (TFJS)
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(path)
                for f in filenames
            )
            size = total_size / (1024 * 1024)
            print(f"{name:20s}: {size:.2f} MB (carpeta)")
```

---

## ✅ Resumen

### Formato RECOMENDADO:
**TFLite INT8** (~3-4 MB)

### Código de Exportación:
```python
model.export(format='tflite', int8=True, imgsz=640)
```

### Dónde Guardar:
- **<10 MB:** Git repository
- **>10 MB:** Google Drive + link en README

### Para la App Web:
Usar TensorFlow.js para cargar el .tflite en el navegador

---

**¿Necesitas ayuda con algún formato específico o con la configuración de Google Drive?**
