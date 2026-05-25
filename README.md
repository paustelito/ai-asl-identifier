# AI ASL Identifier
### Paulina Almada Martínez - A01710029
Este repositorio contiene todo el código necesario para la generación de un modelo de inteligencia artificial que clasifica imágenes de signos de ASL (American Sign Language) para identificar qué letra representan.

## Obtención de datos
Para este proyecto, utilizamos el dataset [American Sign Language](https://www.kaggle.com/datasets/kapillondhe/american-sign-language/data?select=ASL_Dataset) de Kapil Londhe, consultado en Kaggle. Se utilizó la librería KaggleHub para cargar las imágenes del dataset al notebook en Google Colab.

Este dataset incluye **165,782 fotografías RBG** (a color) de todas las letras del alfabeto del lenguaje de señas americano. Viene pre-dividido en conjuntos de test y train con **28 categorías** en cada subconjunto de datos. En train, tenemos fotografías etiquetadas para cada signo con la misma mano y el mismo fondo pero variaciones en la cercanía de la mano a la cámara. La cantidad de imágenes por letra varía, con un mínimo de 4,542 fotos y un máximo de 5,996 fotos. En test, tenemos fotografías con la misma mano y fondo que train, pero todas las letras solamente tienen 4 fotos. Además, todas las fotos de test para cada letra tienen la misma cercanía a la cámara.

## Split de datos
Considerando la disparidad en cantidad y variaciones en los conjuntos pre-establecidos de test y train, definimos realizar nuestro propio split aleatorio entre train y test para el propósito de este proyecto. Además, agregamos una división para tener datos de validación.

Para realizar esto, primero debemos cargar las imágenes a nuestro proyecto. Estas van a venir en su división original, por lo que procedemos a crear una nueva carpeta (*asl_mixed*) donde mezclamos las imágenes de test y train en una sola sub-carpeta por categoría. A base de esta carpeta mezclada, realizamos nuestro split con la librería de sklearn. Ya que estamos trabajando con carpetas, vamos a realizar este split a través de pasar las imágenes a tres nuevas carpetas (test, train y val) dentro de una nueva carpeta padre (*asl_split*). Realizamos una división **80% - 10% - 10%** por la gran cantidad de imágenes con las que contamos. Esto resulta en **132K imágenes de train** (~4.7K por categoría) y **16.5K imágenes para test y val** respectivamente (~589 por categoría).

## Preprocesado de datos
Ya que las imágenes, como discutimos anteriormente, tienen iluminación y fondo parejos y están a color, nuestro único paso de preprocesado es un rescalamiento de los pixeles a rangos de 0-1 (dividiendo por 255) y una reducción del tamaño de las imágenes a 64 x 64 para mejorar el rendimiento del modelo (si no, es demasiado lento un epoch).