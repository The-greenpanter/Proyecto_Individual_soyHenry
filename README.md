# Proyecto Individual
## Sistema de Recomendación de Películas 🎬

## Video

[![Explicación del proyecto](https://i9.ytimg.com/vi/cSlxdvYirgs/mqdefault.jpg?sqp=CKST9bsG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGFUgYShlMA8=&rs=AOn4CLBUtTBBhJSN5f0


## Deploy render
[Ver el deploy en Render](https://proyecto-individual-soyhenry.onrender.com/docs)

Este proyecto consiste en la creación de un sistema de recomendación de películas, utilizando FastAPI para crear una API que permita realizar consultas sobre datos de películas, transformarlos y preparar el sistema para recomendaciones basadas en machine learning.

### Tabla de Contenidos
- [Proyecto Individual](#proyecto-individual)
  - [Sistema de Recomendación de Películas 🎬](#sistema-de-recomendación-de-películas-)
    - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Transformaciones](#transformaciones)
  - [Desarrollo de la API](#desarrollo-de-la-api)
  - [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
  - [Sistema de Recomendación](#sistema-de-recomendación)
  - [Deployment](#deployment)
  - [Estructura del Proyecto](#estructura-del-proyecto)

## Transformaciones

Se realizaron las siguientes transformaciones a los datos para hacerlos adecuados para su consumo por la API:

1. **Desanidado de datos**: Algunas columnas como `belongs_to_collection` y `production_companies` tenían datos anidados (listas o diccionarios). Se desanidaron y se integraron nuevamente al dataset.
   
2. **Valores nulos**: 
   - Los campos `revenue` y `budget` con valores nulos fueron reemplazados por `0`.
   - Los valores nulos en `release_date` se eliminaron.
   
3. **Formato de fecha**: Se aseguró que las fechas tengan el formato `AAAA-mm-dd`. Además, se creó una columna `release_year` extrayendo el año de la fecha de estreno.
   
4. **Cálculo del retorno de inversión**: Se creó la columna `return` que calcula el retorno de inversión dividiendo `revenue / budget`. Si no se dispone de estos datos, se asigna un valor de `0`.

5. **Eliminación de columnas**: Se eliminaron las columnas no utilizadas: `video`, `imdb_id`, `adult`, `original_title`, `poster_path`, y `homepage`.

## Desarrollo de la API

La API se creó utilizando FastAPI y contiene los siguientes endpoints:

- **`cantidad_filmaciones_mes(Mes)`**: Devuelve la cantidad de películas estrenadas en un mes específico (en español).
  
- **`cantidad_filmaciones_dia(Dia)`**: Devuelve la cantidad de películas estrenadas en un día específico (en español).
  
- **`score_titulo(titulo)`**: Proporciona el título, año de estreno y score (popularidad) de una película.

- **`votos_titulo(titulo)`**: Devuelve la cantidad de votos y el promedio de una película, si tiene al menos 2000 valoraciones.

- **`get_actor(nombre_actor)`**: Devuelve el éxito de un actor, el número de películas en las que ha participado y el promedio de retorno.

- **`get_director(nombre_director)`**: Proporciona el éxito de un director, junto con detalles sobre las películas que ha dirigido (fecha de lanzamiento, retorno individual, costo y ganancia).

- **`recomendacion(titulo)`**: Basado en un título de película, recomienda las 5 películas más similares.

## Análisis Exploratorio de Datos (EDA)

Una vez limpiados los datos, se realizaron varias exploraciones para detectar patrones interesantes y relaciones entre variables. 
Se generaron gráficos para visualizar la distribución de puntuaciones, presupuestos y otros datos importantes. Algunas visualizaciones importantes incluyen:

- **Distribución de presupuestos y retornos**
- **Frecuencia de palabras en títulos**: Se generó una nube de palabras para identificar las palabras más comunes en los títulos de películas.

## Sistema de Recomendación

Con los datos listos, se construyó un sistema de recomendación que utiliza la similitud entre películas para sugerir opciones a los usuarios. El sistema toma como entrada el título de una película y devuelve una lista de 5 películas similares, basándose en el score de popularidad.

Este sistema se agregó como un endpoint adicional a la API, accesible a través del método `recomendacion(titulo)`.

## Deployment

Para la implementación de esta API, se recomienda utilizar servicios como **Render** o **Railway**, ambos compatibles con FastAPI y fácilmente deployables. Se ha preparado un tutorial para facilitar el deployment.

## Estructura del Proyecto

```bash
.
├── ETL.ipynb
├── README.md
├── data
│   └── ReadytoETA.csv
├── notebooks
│   ├── EDA-Movies.ipynb
│   └── ETL.ipynb
├── requirements.txt
└── src
    └── main.py
