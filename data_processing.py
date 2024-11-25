import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


# Chargement et prétraitement des données
def load_and_process_data():
    # Charger le fichier CSV des films
    movies_df = pd.read_csv('tmdb_5000_movies.csv')

    # Fonction pour extraire les genres d'une liste de dictionnaires
    def extract_genres(genre_list):
        try:
            genres = ast.literal_eval(genre_list)
            return [genre['name'] for genre in genres if 'name' in genre]
        except Exception as e:
            print(f"Erreur lors de l'extraction des genres : {e}")
            return []

    # Fonction pour extraire les langues parlées d'une liste de dictionnaires
    def extract_languages(language_list):
        try:
            languages = ast.literal_eval(language_list)
            return [lang['name'] for lang in languages if 'name' in lang]
        except Exception as e:
            print(f"Erreur lors de l'extraction des langues : {e}")
            return []

    # Appliquer les fonctions d'extraction aux colonnes correspondantes
    movies_df['genres'] = movies_df['genres'].apply(extract_genres)
    movies_df['spoken_languages'] = movies_df['spoken_languages'].apply(extract_languages)

    # Obtenir la liste des genres uniques
    all_genres = sorted(set(genre for genres in movies_df['genres'] for genre in genres))
    print("Genres uniques disponibles :", all_genres)

    # Normaliser les caractéristiques numériques
    scaler = MinMaxScaler()
    movies_df['vote_average_scaled'] = scaler.fit_transform(movies_df[['vote_average']].fillna(0))
    movies_df['popularity_scaled'] = scaler.fit_transform(movies_df[['popularity']].fillna(0))

    # Encoder les genres
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(movies_df['genres'])
    print("Genres encodés :", mlb.classes_)

    # Fusionner les caractéristiques
    features = np.hstack([genre_encoded,
                           movies_df['vote_average_scaled'].values.reshape(-1, 1),
                           movies_df['popularity_scaled'].values.reshape(-1, 1)])

    # Binariser la note moyenne pour créer des labels
    movies_df['label'] = (movies_df['vote_average'] >= movies_df['vote_average'].median()).astype(int)
    labels = movies_df['label'].values



    return movies_df, features, labels, mlb, scaler
