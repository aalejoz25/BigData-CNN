# Objetos_salon
## Proyecto: Reconocimiento de objetos con TensorFlow

* Este repositorio forma parte del módulo de **Redes Convolucionales** 2025-3
* En el curso **BigData**

---

## Objetivo
Construir, entrenar y evaluar un modelo de visión que identifique objetos del entorno grabados por los estudiantes, utilizando redes convolucionales, autoencoders y clasificadores.

---

## Objetos a detectar
Cada estudiante grabará un video corto (1 a 3 segundos) de los siguientes objetos del salón:

- 🧰 teclado  
- 🖱️ mouse  
- 💻 pantalla  
- 🖥️ cpu  
- 🪑 silla  
- 🧫 mesa
-  nada

>  Todos los videos deben tomarse en el entorno del aula o laboratorio para mantener consistencia visual.

---

## 📁 Estructura del repositorio

```
data/
├─📁 raw/ # Videos originales
│ ├── 20202020202_teclado.mp4
│ ├── 20202020202_mouse.mp4
│ └── ...
├─📁 processed/ # Frames extraídos por clase
│ ├─📁 teclado/
│ │ ├── 20202020202_0001.jpg
│ │ ├── 20202020202_0002.jpg
│ ├─📁 mouse/
│ │ ├── 20202020202_0001.jpg
│ │ ├── 20202020202_0002.jpg
│ └── ...
notebooks/
├── 20202020202.ipynb # Experimentos y entrenamiento por estudiante
├── 20202020202.ipynb
└── ...
models/
├── 20202020202_autoencoder.h5 # Pesos entrenados del autoencoder
├── 20202020202_classifier.h5 # Pesos del clasificador
└── ...
```

>  Los nombres de archivos siguen el formato:
> - Videos: `"{codigo}_{objeto}.mp4"`
> - Frames: `"{objeto}/{codigo}_{indice}.jpg"`
> - Notebooks: `"{codigo}.ipynb"`
> - Pesos: `"{codigo}_autoencoder.h5"` o `"{codigo}_classifier.h5"`

---


## Requisitos básicos

* Extraer frames (1 cada 10–15 cuadros).
* Entrenar un autoencoder con las imágenes de todos los objetos.
* Usar el encoder para extraer features de cada imagen.
* Entrenar un clasificador (CNN) para predecir el tipo de objeto.
* Evaluar el modelo con los videos de otros compañeros.

---

## Resultados esperados
* Representaciones latentes del autoencoder (espacio comprimido de imágenes).
* Clasificación de objetos del entorno.
* Comparación de rendimiento entre distintos entornos o cámaras.
