from fastapi import FastAPI, HTTPException
import pandas as pd
from contextlib import asynccontextmanager
from sklearn.neighbors import NearestNeighbors
import pickle

def obtener_recomendaciones_por_pelicula(pelicula_id: int):
    # Comprobar si el ID de la película es válido
    if pelicula_id < 0 or pelicula_id >= len(features_imputed):
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Usar el modelo KNN para encontrar películas similares
    distances, indices = model_knn.kneighbors([features_imputed[pelicula_id]])

    # Obtener los títulos de las películas recomendadas
    peliculas_similares = df.iloc[indices[0]]['title'].drop_duplicates().tolist()
    return peliculas_similares
# Crear instancia de FastAPI
app = FastAPI()

# Lifespan context manager para cargar el dataset y el modelo de recomendación
# Lifespan context manager para cargar el dataset y el modelo de recomendación
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, model_knn, features_imputed
    df = pd.read_csv('../data/ReadytoETA.csv', dtype={'nombre_columna': 'object'}, low_memory=False)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')  # Asegurar que release_date esté en formato datetime
    
    # Cargar el modelo KNN (o el que estés usando)
    with open('../reports/recommendation_model.pkl', 'rb') as f:
        data = pickle.load(f)
        model_knn = data['knn']
        imputer = data['imputer']
    
    # Recalcular features_imputed
    features = df[['vote_average', 'popularity', 'runtime', 'revenue']].values
    features_imputed = imputer.transform(features)
    
    # Iniciar la app
    yield
app = FastAPI(lifespan=lifespan)

# Redirigir a /docs directamente desde la raíz
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return {"message": "Accede a la documentación interactiva en /docs"}

# ---------------------------------------------------------
# Endpoints sin sistema de recomendación (basados en dataset)
# ---------------------------------------------------------

# 1. Cantidad de filmaciones por mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_num = meses.get(mes.lower())
    
    if mes_num:
        cantidad = df[df['release_date'].dt.month == mes_num].shape[0]
        return {"mensaje": f"{cantidad} películas fueron estrenadas en el mes de {mes}"}
    else:
        raise HTTPException(status_code=400, detail="Mes no válido")

# 2. Cantidad de filmaciones por día
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6
    }
    dia_num = dias.get(dia.lower())
    
    if dia_num is not None:
        cantidad = df[df['release_date'].dt.weekday == dia_num].shape[0]
        return {"mensaje": f"{cantidad} películas fueron estrenadas en los días {dia}"}
    else:
        raise HTTPException(status_code=400, detail="Día no válido")

# 3. Score por título
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    pelicula = df[df['title'].str.lower() == titulo.lower()]
    
    if not pelicula.empty:
        return {
            "mensaje": f"La película {titulo} fue estrenada en el año {int(pelicula['release_year'].values[0])} con un score de {pelicula['vote_average'].values[0]}"
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontró la película {titulo}")

# 4. Votos por título
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    pelicula = df[df['title'].str.lower() == titulo.lower()]
    
    if not pelicula.empty:
        votos = pelicula['vote_count'].values[0]
        if votos >= 2000:
            return {
                "mensaje": f"La película {titulo} tiene {votos} valoraciones."
            }
        else:
            return {"mensaje": "La película no tiene al menos 2000 valoraciones."}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontró la película {titulo}")

# 5. Obtener información de un actor
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    peliculas_actor = df[df['cast'].str.contains(nombre_actor, case=False, na=False)]
    
    if not peliculas_actor.empty:
        cantidad_peliculas = peliculas_actor.shape[0]
        retorno_total = peliculas_actor['return'].sum()
        retorno_promedio = peliculas_actor['return'].mean()
        return {
            "mensaje": f"El actor {nombre_actor} ha participado en {cantidad_peliculas} películas, con un retorno total de {retorno_total:.2f} y un retorno promedio de {retorno_promedio:.2f}."
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")

# 6. Obtener información de un director
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    peliculas_director = df[df['director'].str.contains(nombre_director, case=False, na=False)]
    
    if not peliculas_director.empty:
        peliculas = peliculas_director[['title', 'release_date', 'return']].to_dict(orient='records')
        return {
            "peliculas": peliculas
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el director {nombre_director}")

# ------------------------------------------------------------
# Endpoints con sistema de recomendación usando modelo KNN
# ------------------------------------------------------------

# 1. Recomendaciones por título
@app.get("/recomendar_titulo/{titulo}")
def recomendar_titulo(titulo: str):
    pelicula = df[df['title'].str.lower() == titulo.lower()]
    
    if not pelicula.empty:
        # Obtener el índice de la película
        pelicula_id = pelicula.index[0]
        recomendaciones = obtener_recomendaciones_por_pelicula(pelicula_id)
        return {"recomendaciones": recomendaciones}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontró la película {titulo}")

# 2. Recomendaciones para un actor
@app.get("/recomendar_actor/{nombre_actor}")
def recomendar_actor(nombre_actor: str):
    peliculas_actor = df[df['cast'].str.contains(nombre_actor, case=False, na=False)]
    
    if not peliculas_actor.empty:
        pelicula_id = peliculas_actor.index[0]  # Tomamos la primera película del actor
        recomendaciones = obtener_recomendaciones_por_pelicula(pelicula_id)
        return {"recomendaciones": recomendaciones}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")

# 3. Recomendaciones para un director
@app.get("/recomendar_director/{nombre_director}")
def recomendar_director(nombre_director: str):
    peliculas_director = df[df['director'].str.contains(nombre_director, case=False, na=False)]
    
    if not peliculas_director.empty:
        pelicula_id = peliculas_director.index[0]  # Tomamos la primera película del director
        recomendaciones = obtener_recomendaciones_por_pelicula(pelicula_id)
        return {"recomendaciones": recomendaciones}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el director {nombre_director}")