# Spaceship-Titanic

En este repositorio de GitHub se ha realizado todo el código, tanto el EDA y el preprocesamiento como los scripts para los cinco métodos de clasificación.

## Carpeta "data"

En esta carpeta se encuentran tanto los .csv de la competición como la imputación de missing values por distintos métodos. train_pr.csv y test_pr.csv fue la primera imputación manual, mientras que train_pr_KnnImputed.csv es el dataset con imputación por k-NN, por ejemplo.

Junto a estos archivos se encuentra la carpeta "pickles", donde se encuentran los archivos .pck utilizados para almacenar los modelos de los one-hot encoders y los scalers y normalizers.

## Carpeta "src"

Esta es la carpeta más importante del repositorio, en ella se encuentra el notebook del análisis exploratorio de datos (incluye la imputación manual de missing values), el discretizador, la eliminación de outliers, la imputación de missing values con k-NN, etc.

En el script utils.py se encuentran definidas varias funciones, como la generación de las predicciones en el formato correcto para Kaggle, las funciones para utilizar los scalers y encoders, la lectura de los datasets de entrenamiento y test, etc. Todas estas funciones se han utilizado en los scripts individuales de cada uno de los modelos de clasificación.

Lo mismo ocurre con el archivo ensemble.py, donde están definididas las dos clases de los dos ensembles implementados, un Bagging con 'soft voting' y el algoritmo SAMME.R utilizado en Adaboost.

En esta carpeta también se encuentra el directorio "preproc", donde se encuentran los scripts que generan los archivos .pck de la carpeta "data" necesarios para el preprocesamiento de los datos.

Por último, nos encontramos con la carpeta "models", donde separados en subcarpetas por el nombre de cada método de clasificación, se encuentra el código individual realizado por cada uno de nosotros para el estudio de su método asignado, también se encuentra la carpeta "ensemble" con los tests realizados para comprobar la correcta implementación del Bagging y el Adaboost.

## Carpeta "submissions"

Aquí se encuentran, también divididos en subcarpetas con el nombre de cada método, las predicciones generadas por cada uno de nosotros para enviar a Kaggle.
