import requests
import numpy as np
from data_processing import load_and_process_data
from model import build_model

# Récupération des affiches de films via l'API TMDB
def get_movie_poster(title):
    TMDB_API_KEY = "af70f4ba7e4bd896e3a923e3fc717543"
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Erreur API TMDB : {e}")
    return None

# Fonction pour recommander des films
def recommend_movies(selected_genre, start_idx, movies_df, features, mlb, model, num_recommendations=5):
    genre_filter = movies_df['genres'].apply(lambda genres: selected_genre in genres)
    filtered_movies_df = movies_df[genre_filter]
    filtered_features = features[genre_filter]

    if filtered_features.shape[0] == 0:
        return []

    scores = model.predict(filtered_features).flatten()
    recommended_indices = np.argsort(scores)[::-1][start_idx:start_idx + num_recommendations]
    recommended_movies = filtered_movies_df.iloc[recommended_indices]
    recommendations = []
    for _, row in recommended_movies.iterrows():
        title = row['title']
        poster_url = get_movie_poster(title)
        release_date = row['release_date']
        genres = ', '.join(row['genres'])
        homepage = row['homepage']
        spoken_languages = ', '.join(row['spoken_languages'])
        overview = row['overview']
        recommendations.append((title, poster_url, release_date, genres, homepage, spoken_languages, overview))
    return recommendations


# Fonction pour calculer l'accuracy
def calculate_accuracy(true_labels, predictions):
    correct_predictions = np.sum(true_labels == predictions)
    total_predictions = len(predictions)
    return correct_predictions / total_predictions

# Fonction pour calculer la précision
def calculate_precision(true_labels, predictions, relevant_indices):
    correct_predictions = np.sum((true_labels == predictions) & (relevant_indices == 1))
    total_relevant_predictions = np.sum(relevant_indices)
    return correct_predictions / total_relevant_predictions

# Fonction pour calculer le rappel
def calculate_recall(true_labels, predictions, relevant_indices):
    correct_predictions = np.sum((true_labels == predictions) & (relevant_indices == 1))
    total_relevant_items = np.sum(relevant_indices)
    return correct_predictions / total_relevant_items

# Fonction pour calculer le F1-score
def calculate_f1_score(true_labels, predictions, relevant_indices):
    precision = calculate_precision(true_labels, predictions, relevant_indices)
    recall = calculate_recall(true_labels, predictions, relevant_indices)
    return 2 * (precision * recall) / (precision + recall)