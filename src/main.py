from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from typing import List
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import pickle
from contextlib import asynccontextmanager
import os

app = FastAPI()

# Global variables
df = None
knn = None
imputer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, knn, imputer
    df = pd.read_csv('../data/ReadytoETA.csv')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Load the model and imputer
    with open('../reports/recommendation_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        knn = model['knn']
        imputer = model['imputer']
    
    yield
    # Cleanup code if necessary

app.lifespan = lifespan

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

def get_recommendations(features: List[float], n_neighbors: int = 10) -> List[str]:
    features_imputed = imputer.transform([features])
    distances, indices = knn.kneighbors(features_imputed, n_neighbors=n_neighbors)
    similar_movies = df.iloc[indices[0]]['title'].drop_duplicates().tolist()
    return similar_movies

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, 
        "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, 
        "noviembre": 11, "diciembre": 12
    }
    mes_num = meses.get(mes.lower())
    
    if mes_num:
        cantidad = df[df['release_date'].dt.month == mes_num].shape[0]
        recommendations = get_recommendations([0, 0, 0, 0])
        return {
            "mensaje": f"{cantidad} películas fueron estrenadas en el mes de {mes}",
            "recomendaciones": recommendations
        }
    else:
        raise HTTPException(status_code=400, detail="Mes no válido")

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, 
        "viernes": 4, "sábado": 5, "domingo": 6
    }
    dia_num = dias.get(dia.lower())
    
    if dia_num is not None:
        cantidad = df[df['release_date'].dt.weekday == dia_num].shape[0]
        recommendations = get_recommendations([0, 0, 0, 0])
        return {
            "mensaje": f"{cantidad} películas fueron estrenadas en los días {dia}",
            "recomendaciones": recommendations
        }
    else:
        raise HTTPException(status_code=400, detail="Día no válido")

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    pelicula = df[df['title'].str.lower() == titulo.lower()]
    
    if not pelicula.empty:
        recommendations = get_recommendations([
            pelicula['vote_average'].values[0], 
            pelicula['popularity'].values[0], 
            pelicula['runtime'].values[0], 
            pelicula['revenue'].values[0]
        ])
        return {
            "mensaje": f"La película {titulo} fue estrenada en el año {int(pelicula['release_year'].values[0])} con un score de {pelicula['vote_average'].values[0]}",
            "recomendaciones": recommendations
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontró la película {titulo}")

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    pelicula = df[df['title'].str.lower() == titulo.lower()]
    
    if not pelicula.empty:
        votos = pelicula['vote_count'].values[0]
        promedio_votos = pelicula['vote_average'].values[0]
        
        if votos >= 2000:
            recommendations = get_recommendations([
                pelicula['vote_average'].values[0], 
                pelicula['popularity'].values[0], 
                pelicula['runtime'].values[0], 
                pelicula['revenue'].values[0]
            ])
            return {
                "mensaje": f"La película {titulo} tiene un total de {votos} votos, con un promedio de {promedio_votos}.",
                "recomendaciones": recommendations
            }
        else:
            return {"mensaje": "La película no tiene al menos 2000 valoraciones."}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontró la película {titulo}")

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    peliculas_actor = df[df['cast'].str.contains(nombre_actor, case=False, na=False)]
    
    if not peliculas_actor.empty:
        cantidad_peliculas = peliculas_actor.shape[0]
        retorno_total = peliculas_actor['revenue'].sum() / peliculas_actor['budget'].sum()
        retorno_promedio = retorno_total / cantidad_peliculas
        
        return {
            "mensaje": f"El actor {nombre_actor} ha participado en {cantidad_peliculas} películas, con un retorno total de {retorno_total:.2f} y un retorno promedio de {retorno_promedio:.2f}."
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    peliculas_director = df[df['crew'].str.contains(nombre_director, case=False, na=False)]
    
    if not peliculas_director.empty:
        peliculas = []
        
        for _, row in peliculas_director.iterrows():
            titulo = row['title']
            fecha_lanzamiento = row['release_date']
            retorno = row['revenue'] / row['budget'] if row['budget'] > 0 else 0
            costo = row['budget']
            ganancia = row['revenue'] - row['budget']
            
            peliculas.append({
                "titulo": titulo,
                "fecha_lanzamiento": fecha_lanzamiento,
                "retorno": retorno,
                "costo": costo,
                "ganancia": ganancia
            })
        
        return {
            "mensaje": f"El director {nombre_director} ha dirigido {len(peliculas)} películas",
            "peliculas": peliculas
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el director {nombre_director}")
