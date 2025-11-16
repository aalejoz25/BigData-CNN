# Proyecto: Inventario automático del salón de cómputo

## 1. Objetivo principal

Implementar un sistema web que **detecte y cuente los objetos** del salón de cómputo utilizando un **modelo de TensorFlow Lite optimizado para ejecución local**.  
Los objetos a identificar son:

| Código | Objeto   |
|:-------:|:----------|
| 0 | CPU |
| 1 | Mesa |
| 2 | Mouse |
| 3 | Pantalla |
| 4 | Silla |
| 5 | Teclado |

---

## 2. Condiciones

- El modelo puede **entrenarse en cualquier entorno** (local, nube, etc.)
- La **predicción debe ejecutarse completamente de forma local**, sin depender de servicios o APIs en línea. Sin embargo, sí pueden escribir funciones locales, adicionales a la red neuronal, para la búsqueda y conteo de objetos.  
- Se recomienda convertir el modelo de la red neuronal a **TensorFlow Lite (TFLite)** o **TensorFlow.js** para su integración en la aplicación web.  
- La nota mejorará mientras más pequeño y eficiente sea el modelo utilizado para la predicción.  

---

## 3. Evaluación

| Criterio | Porcentaje |
|:----------|:-----------:|
| Detección y conteo correcto de los objetos solicitados | 40% |
| Tamaño del archivo de parámetros (menos parámetros = mejor nota) | 40% |
| Funcionamiento general de la aplicación web (interfaz, estabilidad y experiencia de usuario) | 15% |
| Documentación básica del proyecto (descripción del modelo, entrenamiento y uso) | 5% |

---

## 4. Entrega

En el repositorio reportado en el formulario enviado deben tener una carpeta llamada **`inventario/`**, donde debe encontrarse la aplicación **`index.html`**.  
Si el archivo de pesos del modelo es muy pesado, pueden colocar en el archivo `inventario/README.md` un **enlace público de Google Drive** con el modelo almacenado.

La aplicación .html **debe permitir subir una imagen `.jpg`** con la foto del salón de cómputo y luego presentar:

- Una **imagen resultante** en la que cada objeto detectado esté marcado con su **número correspondiente** (según la tabla del objetivo principal), en **color azul**.  
- La **cantidad total** de cada tipo de objeto encontrado.

---

## 🚀 IMPLEMENTACIÓN

### Estrategia Utilizada: YOLOv8 Nano + Dataset Sintético

**Por qué esta estrategia:**
- ✅ Aprovecha las imágenes individuales ya existentes
- ✅ Genera automáticamente 200+ imágenes sintéticas con anotaciones
- ✅ YOLOv8n es ultra-ligero (~6 MB) y rápido
- ✅ Cuantización INT8 reduce tamaño a ~3-4 MB
- ✅ Detección real de múltiples objetos simultáneos

### 📝 Proceso de Desarrollo

#### **Paso 1: Generar Dataset Sintético**
```bash
# Ejecutar notebook de generación de datos
jupyter notebook 01_generar_dataset_sintetico.ipynb
```

Este notebook:
- Carga imágenes individuales de `../objetos-salon/processed/`
- Crea 200 imágenes sintéticas combinando múltiples objetos
- Genera automáticamente anotaciones en formato YOLO
- Divide dataset en train (80%) y validación (20%)

**Resultado:**
- `synthetic_dataset/` - Imágenes sintéticas generadas
- `yolo_dataset/` - Dataset en formato YOLO listo para entrenar

---

#### **Paso 2: Entrenar YOLOv8 Nano**
```bash
# Ejecutar notebook de entrenamiento
jupyter notebook 02_entrenar_yolov8.ipynb
```

Este notebook:
- Entrena YOLOv8n con el dataset sintético
- Optimiza hiperparámetros y data augmentation
- Exporta a múltiples formatos (PyTorch, TFLite, ONNX)
- Aplica cuantización INT8 para reducir tamaño

**Parámetros de entrenamiento:**
- Épocas: 100 (con early stopping)
- Batch size: 16
- Image size: 640x640
- Optimizer: AdamW

**Resultados esperados:**
- mAP50: >0.70
- Tamaño TFLite INT8: ~3-6 MB
- Velocidad: <100ms por imagen

---

#### **Paso 3: Probar el Modelo**
```bash
# Ejecutar script de prueba
python test_model.py
```

Verifica que el modelo funciona correctamente antes de integrarlo en la web.

---

#### **Paso 4: Crear Aplicación Web**
```bash
# La aplicación estará en:
index.html
```

**Funcionalidades:**
- 📤 Subir imagen JPG del salón
- 🎯 Detectar objetos con YOLOv8 (TFLite)
- 🔵 Dibujar bounding boxes con números azules
- 📊 Mostrar conteo total por categoría

---

### 📁 Estructura de Archivos

```
inventario/
├── 01_generar_dataset_sintetico.ipynb  # Generación de datos sintéticos
├── 02_entrenar_yolov8.ipynb            # Entrenamiento del modelo
├── test_model.py                       # Script de prueba
├── index.html                          # ⏳ Aplicación web (próximo paso)
├── models/                             # Modelos entrenados
│   ├── inventario_yolov8n.pt          # Modelo PyTorch
│   ├── inventario_yolov8n_int8.tflite # Modelo optimizado ⭐
│   ├── labels.txt                      # Clases
│   └── README.md                       # Documentación del modelo
├── synthetic_dataset/                  # Dataset sintético generado
├── yolo_dataset/                       # Dataset en formato YOLO
└── README.md                           # Este archivo
```

---

### 🎯 Modelo Entrenado

**Arquitectura:** YOLOv8 Nano  
**Framework:** Ultralytics YOLOv8  
**Tamaño de entrada:** 640x640  

**Clases:**
- 0: CPU
- 1: Mesa  
- 2: Mouse
- 3: Pantalla
- 4: Silla
- 5: Teclado

**Tamaños de modelo:**
- PyTorch (.pt): ~6 MB
- TFLite Float32: ~12 MB
- TFLite INT8: ~3-4 MB ⭐ **USADO EN LA WEB**

---

### 💡 Optimizaciones Aplicadas

1. **Cuantización INT8**
   - Reduce tamaño 4x
   - Pérdida de precisión <2%
   - Inferencia más rápida

2. **Data Augmentation**
   - HSV color jittering
   - Rotación ±10°
   - Escalado 0.5-1.5x
   - Flip horizontal 50%
   - Mosaic augmentation

3. **Early Stopping**
   - Paciencia: 15 épocas
   - Evita overfitting
   - Guarda mejor modelo

---

### 📊 Rendimiento Esperado

| Métrica | Valor Objetivo |
|---------|----------------|
| mAP50 | >0.70 |
| mAP50-95 | >0.50 |
| Precision | >0.75 |
| Recall | >0.70 |
| Tamaño modelo | <6 MB |
| Velocidad (CPU) | <500ms |

---

### 🔧 Requisitos

```bash
pip install ultralytics opencv-python pillow numpy matplotlib pyyaml tqdm
```

---

### 📝 Notas de Implementación

1. **Dataset Sintético**
   - Las imágenes sintéticas simulan escenas reales del salón
   - Se eliminan fondos blancos de objetos individuales
   - Se aplican transformaciones aleatorias (escala, posición, rotación)
   - NO hay solapamiento significativo entre objetos

2. **Entrenamiento**
   - GPU recomendada (entrenamiento ~30-60 min)
   - CPU posible pero lento (~2-4 horas)
   - Transfer learning desde COCO dataset

3. **Exportación**
   - TFLite INT8 es el formato recomendado para web
   - Compatible con TensorFlow.js y TFLite Runtime
   - Mantiene precisión aceptable

---

### 🚧 Próximos Pasos

- [x] Generar dataset sintético
- [x] Entrenar YOLOv8n
- [x] Exportar a TFLite
- [ ] **Crear aplicación web (index.html)**
- [ ] Integrar modelo TFLite en JavaScript
- [ ] Implementar detección y conteo
- [ ] Dibujar bounding boxes azules
- [ ] Probar con imágenes reales del salón

---



