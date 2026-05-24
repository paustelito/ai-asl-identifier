# AI ASL Identifier
### Paulina Almada Martínez - A01710029
Este repositorio contiene todo el código necesario para la generación de un modelo de inteligencia artificial que clasifica imágenes de signos de ASL (American Sign Language) para identificar qué letra representan.

## Obtención de datos
Para este proyecto, utilizamos el dataset [American Sign Language](https://www.kaggle.com/datasets/kapillondhe/american-sign-language/data?select=ASL_Dataset) de Kapil Londhe, consultado en Kaggle. Se utilizó la librería KaggleHub para cargar las imágenes del dataset al notebook en Google Colab.

Este dataset incluye **165,782 fotografías RBG** (a color) de todas las letras del alfabeto del lenguaje de señas americano. Viene pre-dividido en conjuntos de test y train con **28 categorías** en cada subconjunto de datos. En train, tenemos fotografías etiquetadas para cada signo con la misma mano y el mismo fondo pero variaciones en la cercanía de la mano a la cámara. La cantidad de imágenes por letra varía, con un mínimo de 4,542 fotos y un máximo de 5,996 fotos. En test, tenemos fotografías con la misma mano y fondo que train, pero todas las letras solamente tienen 4 fotos. Además, todas las fotos de test para cada letra tienen la misma cercanía a la cámara.

Considerando esta disparidad en cantidad y variaciones en los conjuntos pre-establecidos de test y train, definimos realizar nuestro propio split aleatorio entre train y test para el propósito de este proyecto. Además, agregamos una división para tener datos de validación.