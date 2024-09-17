from fastapi import FastAPI, HTTPException
import pandas as pd
from contextlib import asynccontextmanager

# Create FastAPI instance
app = FastAPI()

# Lifespan context manager for loading the dataset
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
    df = pd.read_csv('./data/ReadytoETA.csv')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')  # Ensure release_date is in datetime format
    
    # Start the app
    yield
    # Here you could place cleanup code, if necessary

app = FastAPI(lifespan=lifespan)
# Root endpoint
@app.get("/")
def root():
    return {"message": "Bienvenido a la API de Películas"}

# 1. Cantidad de filmaciones por mes
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
        return {"mensaje": f"{cantidad} películas fueron estrenadas en el mes de {mes}"}
    else:
        raise HTTPException(status_code=400, detail="Mes no válido")

# 2. Cantidad de filmaciones por día
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, 
        "viernes": 4, "sábado": 5, "domingo": 6
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
        promedio_votos = pelicula['vote_average'].values[0]
        
        if votos >= 2000:
            return {
                "mensaje": f"La película {titulo} tiene un total de {votos} votos, con un promedio de {promedio_votos}."
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
        retorno_total = peliculas_actor['revenue'].sum() / peliculas_actor['budget'].sum()
        retorno_promedio = retorno_total / cantidad_peliculas
        
        return {
            "mensaje": f"El actor {nombre_actor} ha participado en {cantidad_peliculas} películas, con un retorno total de {retorno_total:.2f} y un retorno promedio de {retorno_promedio:.2f}."
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")

# 6. Obtener información de un director
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