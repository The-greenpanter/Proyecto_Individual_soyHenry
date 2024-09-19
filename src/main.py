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
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, model_knn, features_imputed
    df = pd.read_csv('./data/ReadytoETA.csv', dtype={'nombre_columna': 'object'}, low_memory=False)
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

# 5. Obtener información de una compañía productora
@app.get("/get_compania/{nombre_compania}")
def get_compania(nombre_compania: str):
    peliculas_compania = df[df['production_companies'].str.contains(nombre_compania, case=False, na=False)]
    
    if not peliculas_compania.empty:
        cantidad_peliculas = peliculas_compania.shape[0]
        retorno_total = peliculas_compania['revenue'].sum()
        retorno_promedio = peliculas_compania['revenue'].mean()
        return {
            "mensaje": f"La compañía {nombre_compania} ha producido {cantidad_peliculas} películas, con un ingreso total de {retorno_total:.2f} y un ingreso promedio de {retorno_promedio:.2f}."
        }
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para la compañía {nombre_compania}")


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

# 2. Recomendaciones por compañía productora
@app.get("/recomendar_produccion/{nombre_compania}")
def recomendar_produccion(nombre_compania: str):
    peliculas_compania = df[df['production_companies'].str.contains(nombre_compania, case=False, na=False)]
    
    if not peliculas_compania.empty:
        pelicula_id = peliculas_compania.index[0]  # Tomamos la primera película
        recomendaciones = obtener_recomendaciones_por_pelicula(pelicula_id)
        return {"recomendaciones": recomendaciones}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para la compañía {nombre_compania}")
# 3. Recomendaciones por fecha de lanzamiento
@app.get("/recomendar_fecha/{fecha}")
def recomendar_fecha(fecha: str):
    peliculas_fecha = df[df['release_date'] == fecha]
    
    if not peliculas_fecha.empty:
        pelicula_id = peliculas_fecha.index[0]  # Tomamos la primera película
        recomendaciones = obtener_recomendaciones_por_pelicula(pelicula_id)
        return {"recomendaciones": recomendaciones}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para la fecha {fecha}")