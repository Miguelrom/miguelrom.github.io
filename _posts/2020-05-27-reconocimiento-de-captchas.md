---
title: Reconocimiento de CAPTCHAS mediante Redes Neuronales Convolucionales
---

## Introducción
En este artículo se describe brevemente qué son las Redes Neuronales Convolucionales, su aplicación a la clasificación de imágenes, además de presentar un ejemplo de una red convolucional que reconoce los caracteres en la imagen de un CAPTCHA.

<hr>

## Redes Neuronales Convolucionales
Una Red Neuronal Convolucional, o CNN por sus siglas en inglés, es una clase de red neuronal que toma como entrada una imagen para después extraer ciertos tipos de características de dicha imagen en cada capa de neuronas subsiguiente, con el fin de identificar bordes, sombras, cambios de color, etc. Además de identificar los elementos anteriores, las primeras capas por las que pasa la imagen también tienen el objetivo de reducir la dimensionalidad y de esta manera reducir el costo computacional y el sobreaprendizaje. Las últimas capas de la CNN consisten en una red de alimentación hacia adelante que realiza la clasificación o identificación de un objeto dentro de una imagen.
El proceso de una CNN se puede dividir en dos etapas, a saber:
* Aprendizaje de características
	* Convolución + ReLU: en esta capa se realiza el proceso de escaneo de la imagen convertida en una matriz multidimensional (_tensor_) mediante uno o más filtros para extraer características. A la matriz resultante se la aplica la función ReLU para eliminar los valores negativos.
	* Max Pooling. En la siguiente capa se realiza el escaneo de la matriz anterior con un filtro (usualmente de 2x2) que obtiene los valores máximos.
* Clasificación
	* Aplanamiento. En esta capa la matriz se transforma en un vector columna para que sea la entrada de una red alimentada hacia adelante.
	* Alimentación hacia adelante. Por último, los datos pasan por una red alimentada hacia adelante que tiene como función de activación en la última capa la función softmax, la cual realiza la clasificación.

El proceso que realiza una Red Neuronal Convolucional explicado anteriormente se puede apreciar en la siguiente imagen:

![CNN photo](https://cdn-images-1.medium.com/fit/t/1600/480/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

<hr>

## ¿Qué es un CAPTCHA?
CAPTCHA es un acrónimo en inglés que significa Completely Automated Public Turing test to tell Computers and Humands Apart. Es decir, es un tipo de medida de seguiridad para poder distinguir a un humano de una computadora.
La prueba de un CAPTCHA consta de dos partes simples: una secuencia de letras o de números generada aleatoriamente que aparece como una imagen distorsionada y un cuadro de texto. Para superar la prueba y probar que eres un ser humano, simplemente tienes que escribir los caracteres que veas en la imagen en el cuadro de texto.
A continuación se presenta una imagen con ejemplos de CAPTCHAs.

![captchas](https://camo.githubusercontent.com/d2ec7ccc16dacc36732ff2c6bad51df1bad2428b/687474703a2f2f677265677761722e636f6d2f63617074636861732e706e67)

Ejemplo de una prueba de CAPTCHA:

![mas captchas](https://2015.800noticias.com/cms/wp-content/uploads/2015/03/captchafb.jpg)

<hr>

## Reconocimiento de CAPTCHAS mediante una Red Neuronal Convolucional
La red pre-entrenada para reconocer los caracteres de un CAPTCHA fue desarrollada en pytorch y puede ser encontrada en el siguiente <a href="https://github.com/skyduy/CNN_keras/tree/pytorch">repositorio</a> de GitHub.
El modelo obtenido por la red es capaz de reconocer cuatro letras mayúsculas en un CAPTCHA con una efectividad de 96% en la identificación de caracteres.
<br>
El generador de CAPTCHAs fue desarrollado por el mismo usuario y puede ser
encontrado <a href="https://github.com/skyduy/CAPTCHA_generator">aquí</a> .

La arquitectura de la red es la siguiente:
![png](/assets/images/cnn/model.png)


#### Pruebas
Para poner a prueba el reconocimiento de CAPCHAs primero tenemos que generar las imágenes utilizando el generador de CAPTCHAs. Por facilidad, hagamos un script en _Python_ llamado _generar_captchas.py_ para generan 10 imágenes. El código del script es el siguiente:

```python
from captcha import Captcha
caracteres = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'M',
               'N', 'P', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']
num_caracteres = 4
ancho = 120
alto = 36
c = Captcha(ancho, alto, caracteres, num_caracteres)
c.batch_create_img(10)
```

El código anterior genera 10 imágenes de CAPTCHAs con 4 caracteres elegidos aleatoriamente del conjunto de caracteres definido.<br>
**NOTA:** para que las imágenes sean compatibles con la red pre-entrenada, lo único que puede cambiar en el script anterior es el número de imágenes generadas.

Ahora corramos el script utilizando _IPython_ por comodidad.

```ipython
In [21]: run generar_captchas.py
10 generated.
10 captchas saved into C:\Users\migue\Documents\GITHUB\CAPTCHA_generator-master\samples.
```
Las archivos que se generaron tienen en las primeras cuatro letras de su nombre los caracteres del CAPTCHA, seguido de un guión bajo y un número único para evitar que dos o más archivos tengan el mismo nombre.

Lo que sigue ahora es identificar los caracteres de los captchas usando el modelo pre-entrenado. Para esto vamos a hacer otro script en _Python_ al que llamaremos _identificar_captchas.py_ que nos automatizará las cosas. El código del script es el siguiente:

```python
import os
from predict import Predictor

directorio = r'..\\CAPTCHA_generator-master\\samples'
mi_pred = Predictor('pretrained')

print("_"*49)
print("|                    |                          |")
print("| Nombre del archivo | Caracteres identificados |")
print("|____________________|__________________________|")

for archivo in os.listdir(directorio):
	path = os.path.join(directorio, archivo)
	print("|                    |                          |")
	print("|   %s    |   %s   |" %(archivo, mi_pred.identify(path)))
	print("|____________________|__________________________|")
```

Ahora corramos el script en _IPyton_:

```ipython
In [66]: run identificar_captchas.py
_________________________________________________
|                    |                          |
| Nombre del archivo | Caracteres identificados |
|____________________|__________________________|
|                    |                          |
|   FWHH_4969.jpg    |   ['F', 'W', 'H', 'H']   |
|____________________|__________________________|
|                    |                          |
|   HHXH_c2dd.jpg    |   ['H', 'H', 'X', 'H']   |
|____________________|__________________________|
|                    |                          |
|   NDHM_a1a4.jpg    |   ['N', 'D', 'H', 'M']   |
|____________________|__________________________|
|                    |                          |
|   NFBD_4d9d.jpg    |   ['N', 'F', 'B', 'D']   |
|____________________|__________________________|
|                    |                          |
|   NGDE_3925.jpg    |   ['N', 'G', 'D', 'F']   |
|____________________|__________________________|
|                    |                          |
|   TNAX_6c98.jpg    |   ['G', 'N', 'N', 'X']   |
|____________________|__________________________|
|                    |                          |
|   TNGV_caa4.jpg    |   ['T', 'N', 'G', 'V']   |
|____________________|__________________________|
|                    |                          |
|   UKFH_ba87.jpg    |   ['D', 'K', 'B', 'B']   |
|____________________|__________________________|
|                    |                          |
|   VVBE_d0a7.jpg    |   ['V', 'V', 'B', 'E']   |
|____________________|__________________________|
|                    |                          |
|   WNGB_0f07.jpg    |   ['W', 'N', 'G', 'B']   |
|____________________|__________________________|
```

Para poder visualizar los aciertos y errores del programa, se presenta la siguiente imagen en donde se yuxtapone la salida del script anterior y los correspondientes CAPTCHAs en cada renglón:

![tabla](/assets/images/cnn/tabla-captcha.png)

Podemos observar que el programa tuvo 6 errores en la identificación de caracteres presentes en los CAPTCHAs. Es decir se obtuvo una efectividad del 85%, lo cual es menor a lo esperado. Sin embargo, cabe mencionar que solo se tomaron 10 imágenes como muestra.

<hr>

## Conclusiones
Las Redes Neuronales Convolucionales (CNN) son una gran herramienta para las aplicaciones de Visión por Computadora. El programa presentado en este artículo para el reconocimiento de CAPTCHAs es un claro ejemplo de ello. La red neuronal pre-entrenada tuvo un buen desempaño al hora de identificar los caracteres borrosos y torcidos en las imágenes.
Algunas mejoras que se le pueden hacer al modelo son, por ejemplo, entrenar a la red para que reconozca todas las letras del alfabeto en mayúsculas y minúsculas, números, además de poder reconocer CAPTCHAs con mayor número de caracteres.






