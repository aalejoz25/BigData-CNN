# 🎯 GUÍA PASO A PASO - Sistema de Inventario

## 📋 Resumen

Esta guía te llevará desde tus imágenes individuales hasta una aplicación web completa de detección de objetos en **3-4 horas**.

---

## ⚡ PROCESO RÁPIDO (Para empezar YA)

### Paso 1: Generar Dataset Sintético (30-45 min)

1. Abre el notebook `01_generar_dataset_sintetico.ipynb`
2. Ejecuta todas las celdas en orden
3. Verifica que se generaron:
   - ✅ `synthetic_dataset/images/` (~200 imágenes)
   - ✅ `synthetic_dataset/labels/` (~200 archivos .txt)
   - ✅ `yolo_dataset/` (estructura train/val)

**Qué hace:**
- Carga tus imágenes de `objetos-salon/processed/`
- Crea escenas sintéticas combinando objetos
- Genera anotaciones automáticas
- Prepara dataset en formato YOLO

---

### Paso 2: Entrenar YOLOv8 (30-120 min dependiendo de GPU)

1. Abre el notebook `02_entrenar_yolov8.ipynb`
2. Ejecuta todas las celdas en orden
3. El entrenamiento tomará:
   - Con GPU: 30-60 minutos
   - Con CPU: 1-3 horas (más lento pero funciona)

**Qué hace:**
- Entrena YOLOv8 nano (modelo pequeño)
- Aplica optimizaciones y data augmentation
- Exporta a TFLite con cuantización INT8
- Genera métricas y visualizaciones

**Resultados esperados:**
```
models/
├── inventario_yolov8n.pt              # ~6 MB
├── inventario_yolov8n_int8.tflite    # ~3-4 MB ⭐
├── inventario_yolov8n_float32.tflite # ~12 MB
└── labels.txt
```

---

### Paso 3: Probar el Modelo (5 min)

```bash
cd inventario
python test_model.py
```

Esto verificará que el modelo funciona antes de la app web.

---

### Paso 4: Crear Aplicación Web (Próximo paso)

Necesitarás:
- `index.html` - Interfaz principal
- JavaScript para cargar modelo TFLite
- Lógica de detección y dibujo de bounding boxes

---

## 📊 Qué Esperar

### Dataset Sintético

**Antes:**
```
processed/
├── cpu/       (20 imágenes individuales)
├── mesa/      (20 imágenes individuales)
├── mouse/     (20 imágenes individuales)
└── ...
```

**Después:**
```
yolo_dataset/
├── train/
│   ├── images/  (160 escenas sintéticas)
│   └── labels/  (160 anotaciones YOLO)
└── val/
    ├── images/  (40 escenas sintéticas)
    └── labels/  (40 anotaciones YOLO)
```

Cada escena tiene 3-8 objetos distribuidos aleatoriamente.

---

### Modelo Entrenado

**Métricas objetivo:**
- mAP50: >0.70 (bueno para este caso)
- Precision: >0.75
- Recall: >0.70

**Tamaño:**
- TFLite INT8: ~3-6 MB (óptimo para web)
- Velocidad: <100ms por imagen en GPU, <500ms en CPU

---

## 🔧 Solución de Problemas

### Error: "No se encontraron imágenes"

**Problema:** El script no encuentra las imágenes en `processed/`

**Solución:**
```python
# Verifica que esta ruta sea correcta en el notebook:
PROCESSED_DIR = "../objetos-salon/processed"

# Debe apuntar a donde están tus carpetas:
# cpu, mesa, mouse, pantalla, silla, teclado
```

---

### Error: "Out of memory" durante entrenamiento

**Problema:** Memoria GPU insuficiente

**Solución:**
```python
# En el notebook 02, reduce el batch size:
BATCH_SIZE = 8  # En lugar de 16
# o
BATCH_SIZE = 4  # Si sigue fallando
```

---

### Entrenamiento muy lento en CPU

**Problema:** No tienes GPU disponible

**Solución:**
```python
# Reduce épocas para prueba rápida:
EPOCHS = 30  # En lugar de 100

# O entrena con menos imágenes:
NUM_SYNTHETIC_IMAGES = 100  # En lugar de 200
```

---

## 📈 Optimizaciones Opcionales

### 1. Agregar Imágenes Reales

Si tienes fotos reales del salón:

```python
# En 01_generar_dataset_sintetico.ipynb
# Agrega después de generar sintéticas:

real_images_dir = "./real_photos"
# Copia tus fotos reales ahí
# Etiquétalas manualmente con LabelImg o Roboflow
```

---

### 2. Aumentar Tamaño del Dataset

```python
# En 01_generar_dataset_sintetico.ipynb
NUM_SYNTHETIC_IMAGES = 500  # Más imágenes = mejor modelo
```

⚠️ Más imágenes = más tiempo de entrenamiento

---

### 3. Fine-tuning del Modelo

```python
# En 02_entrenar_yolov8.ipynb
EPOCHS = 150  # Más épocas para mejor precisión
lr0 = 0.0005  # Learning rate más bajo para fine-tuning
```

---

## ✅ Checklist de Progreso

### Generación de Datos
- [ ] Notebook 01 ejecutado sin errores
- [ ] 200 imágenes sintéticas generadas
- [ ] Archivos .txt de anotaciones creados
- [ ] Dataset dividido en train/val (80/20)

### Entrenamiento
- [ ] Notebook 02 ejecutado sin errores
- [ ] Modelo converge (loss disminuye)
- [ ] mAP50 > 0.60
- [ ] Modelo exportado a TFLite

### Validación
- [ ] `test_model.py` funciona
- [ ] Detecta objetos correctamente
- [ ] Tamaño del modelo <10 MB

### Aplicación Web (Próximo)
- [ ] `index.html` creado
- [ ] Modelo TFLite cargado en JavaScript
- [ ] Detección funciona en navegador
- [ ] Bounding boxes dibujados en azul
- [ ] Conteo de objetos mostrado

---

## 🎓 Conceptos Clave

### ¿Qué son las imágenes sintéticas?

Son imágenes creadas artificialmente combinando objetos individuales. En lugar de tomar 200 fotos del salón, usamos tus 20 imágenes por objeto y las combinamos automáticamente.

**Ventajas:**
- No necesitas tomar fotos
- Control total sobre anotaciones
- Dataset balanceado automáticamente

---

### ¿Qué es YOLO?

**You Only Look Once** - Algoritmo de detección de objetos ultra-rápido.

- Detecta múltiples objetos en una sola pasada
- Predice bounding boxes y clases simultáneamente
- YOLOv8n es la versión nano (más pequeña y rápida)

---

### ¿Qué es la cuantización INT8?

Reducir precisión de números de 32 bits a 8 bits.

**Efecto:**
- Tamaño del modelo: ÷4
- Velocidad: ×2-3
- Precisión: -1% a -2% (casi imperceptible)

**Ejemplo:**
- Float32: 12.3456789 (32 bits)
- INT8: 12 (8 bits)

---

## 📚 Recursos Adicionales

### Documentación
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Roboflow (para etiquetar)](https://roboflow.com/)

### Herramientas Útiles
- **Netron**: Visualizar arquitectura del modelo
- **TensorBoard**: Monitorear entrenamiento
- **LabelImg**: Etiquetar imágenes manualmente

---

## 🆘 Ayuda

Si encuentras problemas:

1. **Revisa los logs** de los notebooks
2. **Verifica las rutas** de archivos
3. **Comprueba la memoria** disponible
4. **Lee los mensajes de error** completos

---

## 🎯 Siguiente Paso

Una vez completados los pasos 1-3, continúa con:

**Crear la aplicación web** (`index.html`)

Esto se hará en el siguiente paso de la guía.

---

**¡Buena suerte! 🚀**
