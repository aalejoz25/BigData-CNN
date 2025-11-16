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



