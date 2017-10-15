layout: true

---
class: middle, center, general
# Image Stitching
## Una introducción práctica a OpenCV
Una pequeña charla por [Carlos González](http://caal-15.github.io)

Síguela en [caal-15.github.io/image-stitching-talk](http://caal-15.github.io/image-stitching-talk)
.footnote[.alt-link[[Sitio de OpenCV](https://opencv.org/)]]

---
class: left, middle

# ¿Qué Es OpenCV?

.big[Es una _Librería_, escrita en __C++__, para construir aplicaciones que requieran herramientas de _Visión por Computadora_.]

---
class: center, middle

# ¿En qué lenguajes de programación lo Puedo Usar?

.big[
Existen bindings __oficiales__ para:

* _MATLAB/OCTAVE_
* _Python_
* _Java_
]

En este caso la estaremos utilizando específicamente dentro de _Python 3.6_.

---
class: left middle

# Herramientas

.big[
* Herramientas básicas para _cargar, escribir, y mostrar imágenes_.
* Herramientas básicas de manejo de imágenes (por ej. _conversión a escala de grises_).
* _Detección de Puntos Clave_ (__Harris__, __SIFT__, __SURF__).
* _Correspondencias entre Puntos Clave_ (__Knn__, __Flann__).
* Estimación de _Homografías_ y cambio de _perspectiva_.
* Herramientas para _detección de objetos_.
* Herramientas para _detección de movimientos_.
* _Mucho más_ de lo que cabe en esta diapositiva...
]

---
class: left middle

# Ventajas

.big[
* __Performance:__  _C++_ ha sido, y aún es, uno de los lenguajes compilados altamente utilizados gracias a su desempeño.

* __Facilidad de uso:__ Las interfaces y métodos provistos por _OpenCV_ son muy amigables con el usuario, y no requieren conocimiento demasiado profundo del algoritmo para usarlos.

* __Módulo de Machine Learning:__ La librería incluye un módulo de _Machine Learning_ que cubre la mayoría de necesidades.

* __Compatiblidad:__ La librería es compatible tanto con _OpenGL_, como _CUDA_.

* __Integración:__ Si se están usando los bindings de _Python_, la librería está muy bien integrada con la popular librería _numpy_ (También se puede integrar fácilmente con _TensorFlow_ o _Torch_).
]

---
class: left, middle

# ¿Qué vamos a hacer?

.big[
Vamos a _coser_ (stitch) dos imágenes para lograr un efecto de imagen _panorámica_, usando _Detección y Correspondencias entre Puntos Clave_, _Homografías_ y _Cambio de Perspectiva_!
]

.footnote[.alt-link[[Implementación original](https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)]]
---
class: left, middle

# Paso 0: _Importar numpy y OpenCV, La clase!_

```python
import numpy as np
import cv2

class ImageStitcher():
  ...
```

.footnote[.alt-link[Just in Case...]]

---
class: left, middle

# Paso 1: _Detección de Puntos Clave_
```python
def find_keypts_and_feats(self, img):
  color = cv2.COLOR_BGR2GRAY
  gray = cv2.cvtColor(img, color)
  finder = cv2.xfeatures2d.SIFT_create()
  (keypts, feats) = finder.detectAndCompute(
    gray, None)
  np_keypts = np.array(
    [pt.pt for pt in keypts], np.float32)
  return np_keypts, feats
```

---
class: left, middle

# Paso 2a: _Correspondencias entre los Puntos_
```python
def match_keypts(self, data_a, data_b, ratio):
  matcher = cv2.DescriptorMatcher_create(
    "BruteForce")
  raw_matches = matcher.knnMatch(
    data_a[1], data_b[1], 2)
```

---
class: left, middle

# Paso 2b: _Removiendo con Lowe_
```python
def match_keypts(self, data_a, data_b, ratio):
  ...
  final_matches = []
  for m in raw_matches:
    c1 = len(m) == 2
    c2 = m[0].distance < m[1].distance * ratio
    if c1 and c2:
      final_matches.append((
        m[0].trainIdx, m[0].queryIdx))
```

---
class: left, middle

# Paso 2c: _Limpiando un poco_
```python
def match_keypts(self, data_a, data_b, ratio):
  ...
  pts_a = np.array(
    [data_a[0][i] for (_, i) in final_matches],
    np.float32)
  pts_b = np.array(
    [data_b[0][i] for (i, _) in final_matches],
    np.float32)
  return (pts_a, pts_b)
```

---
class: left, middle

### __Ejemplo__: Detección y Correspondencias entre Puntos

![Detection a matching](img/MatchingPoints.jpg)

---
class: left, middle

# Paso 3a: _Juntando todo, la Homografía_
```python
def stitch(self, img_a, img_b, ratio, thresh):
  data_a = self.find_keypts_and_feats(img_a)
  data_b = self.find_keypts_and_feats(img_b)
  (pts_a, pts_b) = self.match_keypts(
    data_a, data_b, ratio)
  (H, _) = cv2.findHomography(
    pts_b, pts_a, cv2.RANSAC, thresh)
```

---
class: left, middle

### __Ejemplo:__ Cambio de Perspectiva con Homografías

![Warp Perspective](img/picture.jpg)

---
class: left, middle

### __Ejemplo:__ Cambio de Perspectiva con Homografías

![Warp Perspective](img/transplaned.png)

---
class: left, middle

# Paso 3b: _Cambia la Perspectiva!_
```python
def stitch(self, img_a, img_b, ratio, thresh):
  ...
  new_dims = (
    img_a.shape[1] + img_b.shape[1],
    img_a.shape[0])
  stitch = cv2.warpPerspective(
    img_b, H, new_dims)
  stitch[
    0:img_a.shape[0],
    0:img_a.shape[1]
  ] = img_a
  return stitch
```

---
class: left, middle

# Paso 4: _Tómate una Cerveza!_

## Una bien _fría_, te lo mereces.

---
class: left, middle

# Algunas Consideraciones

.big[
* Vas a necesitar compilar _OpenCV_ con los módulos extra (__contrib__)
  para usar toda la funcionalidad presentada aquí (con la bandera
  __OPENCV_EXTRA_MODULES_PATH__).

* También necesitarás funciones básicas de _OpenCV_ para ver los resultados,
  por ejemplo _imread_ e _imwrite_.
]

.footnote[.alt-link[[Link al repo Contrib](https://github.com/opencv/opencv_contrib)]]
---
class: center, middle

# Gracias por su atención!
