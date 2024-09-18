import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import os
import pickle

# Especificar los tipos de datos para las columnas problemáticas
dtype_dict = {
    'budget': float,
    'id_original': str,
    'original_language': str,
    'overview': str,
    'popularity': float,
    'release_date': str,
    'revenue': float,
    'runtime': float,
    'status': str,
    'tagline': str,
    'title': str,
    'vote_average': float,
    'vote_count': float,
    'id_genre': float,
    'genres': str,
    'production_companies': str,
    'id': float,
    'iso_3166_1': str,
    'name_production_countries': str,
    'release_year': float,
    'return': float
}

# Cargar el archivo CSV con tipos de datos especificados
df = pd.read_csv("../data/ReadytoETA.csv", dtype=dtype_dict, low_memory=False)

# Seleccionar las características relevantes para el modelo de recomendación
features = df[['vote_average', 'popularity', 'runtime', 'revenue']].values

# Usar SimpleImputer para rellenar los valores nulos con el promedio de cada columna
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Crear y entrenar el modelo de vecinos más cercanos (K-Nearest Neighbors)
knn = NearestNeighbors(n_neighbors=10)
knn.fit(features_imputed)

# Encontrar las películas más similares a la primera película
distances, indices = knn.kneighbors([features_imputed[0]])

# Definir la carpeta de salida
output_dir = '../reports'
os.makedirs(output_dir, exist_ok=True)

# Guardar el modelo
model_path = os.path.join(output_dir, 'recommendation_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump({
        'imputer': imputer,
        'knn': knn
    }, file)

print(f"Modelo exportado con éxito a {model_path}.")
