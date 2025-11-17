# Proyecto: Inventario del salón

### Inicio Rápido

```bash
# 1. Abrir la aplicación web
python3 -m http.server 8000
# Navegar a: http://localhost:8000/index.html

# 2. Subir una imagen del salón
# 3. Ver detecciones automáticas con cajas azules y conteo
```

---

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

## IMPLEMENTACIÓN

### Enfoque Utilizado

Se implementó un sistema de detección de objetos basado en **YOLOv8 Nano** con las siguientes características:

- **Generación automática de bounding boxes** desde imágenes individuales
- **Dataset de 1,550 imágenes** (1,238 train / 312 val)
- **Modelo YOLOv8n**: ~3M parámetros
- **Conversión a ONNX** para ejecución web local con ONNX Runtime Web
- **Aplicación web 100% funcional** con inferencia local en el navegador

### Estructura del Proyecto

```
inventario/
├── 01_generar_bboxes_automatico.py # Generador automático de bounding boxes
├── 02_entrenar_yolo.ipynb          # Notebook de entrenamiento YOLOv8
├── index.html                      # Aplicación web 
├── training.log                    # Log del entrenamiento
├── yolov8n.pt                      # Modelo base YOLOv8n
├── dataset/                        # Dataset YOLO generado
│   ├── images/
│   │   ├── train/                 # 1,238 imágenes de entrenamiento
│   │   └── val/                   # 312 imágenes de validación
│   ├── labels/
│   │   ├── train/                 # Etiquetas YOLO formato .txt
│   │   └── val/
│   └── data.yaml                  # Configuración del dataset
└── runs/detect/inventario/         # Resultados del entrenamiento
    └── weights/
        ├── best.pt                # Mejor modelo PyTorch 
        ├── best.onnx             # Modelo ONNX Web 
        └── last.pt               # Último checkpoint 
```

---

## GUÍA DE USO

### Paso 1: Generar Dataset

El dataset se genera automáticamente desde las imágenes ya procesadas en `../objetos-salon/processed/`:

```bash
python 01_generar_bboxes_automatico.py
```

**Salida:**
- 1,238 imágenes de entrenamiento
- 312 imágenes de validación
- Bounding boxes generados automáticamente por detección de contornos
- Formato YOLO (.txt con `class_id x_center y_center width height`)

### Paso 2: Entrenar Modelo YOLO

Usar el notebook Jupyter para entrenamiento:

```bash
jupyter notebook 02_entrenar_yolo.ipynb
```

O ejecutar directamente con Python:

```python
from ultralytics import YOLO

# Cargar modelo base
model = YOLO('yolov8n.pt')

# Entrenar
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='inventario'
)

# Exportar a ONNX
model.export(format='onnx', imgsz=640, simplify=True)
```

**Configuración:**
- Modelo: YOLOv8n (Nano - el más pequeño)
- Épocas: 100 (con early stopping)
- Batch size: 16
- Imagen: 640x640
- Optimizador: AdamW
- Augmentación: Activada

**Salida:**
- Modelo entrenado: `runs/detect/inventario/weights/best.pt` 
- Modelo ONNX: `runs/detect/inventario/weights/best.onnx` 
- Métricas: mAP50 = 50.3%, mAP50-95 = 31.4%
- Logs y gráficas en `runs/detect/inventario/`

### Paso 3: Usar Aplicación Web

**La aplicación está lista para usar:**

1. **Abrir en el navegador:**
   ```bash
   # Opción 1: Abrir directamente index.html en Chrome/Firefox/Edge
   
   # Opción 2: Usar servidor HTTP simple
   python3 -m http.server 8000
   # Navegar a: http://localhost:8000/index.html
   ```

2. **Usar la interfaz:**
   - Esperar a que cargue el modelo ONNX (~2-3 segundos la primera vez)
   - Ver mensaje: "Modelo cargado. Listo para detectar objetos."
   - Hacer clic en "Seleccionar Imagen del Salón"
   - Elegir archivo `.jpg`, `.jpeg` o `.png`
   - Ver resultados automáticamente


---

## 🔧 DETALLES TÉCNICOS

### Generación Automática de Bounding Boxes

El script `01_generar_bboxes_automatico.py` utiliza **detección de contornos con OpenCV**:

1. **Preprocesamiento:**
   - Conversión a escala de grises
   - Gaussian blur (reducción de ruido)
   - Detección de bordes (Canny)
   - Dilatación morfológica

2. **Detección del objeto:**
   - `cv2.findContours()` encuentra contornos
   - Selección del contorno más grande (objeto principal)
   - `cv2.boundingRect()` extrae coordenadas

