import streamlit as st
from data_processing import load_and_process_data
from model import build_model
from recommendation import recommend_movies, get_movie_poster, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
from tensorflow.keras.callbacks import EarlyStopping

# Interface principale de l'application Streamlit
def main():
    # Configurer la page Streamlit
    st.set_page_config(page_title="Recommandations de Films IA", page_icon="🎬", layout="wide")

    # Titre principal
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Système de Recommandation de Films 🎥</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #333;'>Découvrez les meilleurs films selon vos préférences</h2>", unsafe_allow_html=True)

    # Charger et prétraiter les données
    movies_df, features, labels, mlb, scaler = load_and_process_data()

    # Vérifier que les données sont correctement chargées
    if features is not None and labels is not None:
        # Construire et entraîner le modèle
        model = build_model(features.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(features, labels, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        # Initialiser l'index de départ pour les recommandations
        if "start_idx" not in st.session_state:
            st.session_state.start_idx = 0

        # Initialiser l'historique des recommandations
        if "recommendation_history" not in st.session_state:
            st.session_state.recommendation_history = []

        # Sélectionner un genre
        all_genres = sorted(set(genre for genres in movies_df['genres'] for genre in genres))
        selected_genre = st.selectbox("🎬 Choisissez un Genre", all_genres)

        if selected_genre:
            # Obtenir les recommandations de films
            recommendations = recommend_movies(selected_genre, st.session_state.start_idx, movies_df, features, mlb, model)
            true_labels = labels[st.session_state.start_idx:st.session_state.start_idx + len(recommendations)]
            predictions = [1 if label == 1 else 0 for label in true_labels]
            relevant_indices = [1 if label == 1 else 0 for label in true_labels]

            if recommendations:
                st.write("### Films Recommandés :")
                cols = st.columns(5)
                for idx, (title, poster_url, release_date, genres, homepage, spoken_languages, overview) in enumerate(recommendations):
                    with cols[idx]:
                        st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;'>"
                                        f"<img src='{poster_url}' style='width: 100%; border-radius: 5px;'/>"
                                        f"<h3 style='color: #333;'>{title}</h3>"
                                        f"<p style='color: #555;'>📅 Date de sortie: {release_date}</p>"
                                        f"<p style='color: #555;'>🎭 Genres: {genres}</p>"
                                        f"<p style='color: #555;'>🗣 Langues parlées: {spoken_languages}</p>"
                                        f"<p style='color: #555;'>📜 Résumé: {overview}</p>"
                                        f"<a href='{homepage}' target='_blank' style='color: #FF4B4B;'>🌐 Homepage</a>"
                                        f"</div>", unsafe_allow_html=True)

                # Ajouter les recommandations à l'historique
                st.session_state.recommendation_history.extend(recommendations)
            else:
                st.write("Aucun film trouvé pour ce genre.")

            # Boutons pour voir plus de films ou réinitialiser la liste
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Voir Plus", use_container_width=True):
                    st.session_state.start_idx += 5
                    st.success("Chargement de plus de films...")
                    st.rerun()
            with col2:
                if st.button("🔄 Réinitialiser", use_container_width=True):
                    st.session_state.start_idx = 0
                    st.success("Liste réinitialisée.")
                    st.rerun()

        # Section "À propos"
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #333;'>À propos</h3>", unsafe_allow_html=True)
        st.markdown("Ce système de recommandation de films utilise un modèle d'IA avancé pour vous proposer les meilleurs films selon vos préférences. Explorez différents genres et découvrez de nouveaux films à regarder!")

        # Afficher l'historique des recommandations
        st.markdown("### Historique des Recommandations")
        if st.session_state.recommendation_history:
            for idx, (title, poster_url, release_date, genres, homepage, spoken_languages, overview) in enumerate(st.session_state.recommendation_history):
                
                st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;'>"
                                f"<img src='{poster_url}' style='width: 10%; border-radius: 5px;'/>"
                                f"<h3 style='color: #333;'>{title}</h3>"
                                f"<p style='color: #555;'>📅 Date de sortie: {release_date}</p>"
                                f"<p style='color: #555;'>🎭 Genres: {genres}</p>"
                                f"<p style='color: #555;'>🗣 Langues parlées: {spoken_languages}</p>"
                                f"<p style='color: #555;'>📜 Résumé: {overview}</p>"
                                f"<a href='{homepage}' target='_blank' style='color: #FF4B4B;'>🌐 Homepage</a>"
                                f"</div>", unsafe_allow_html=True)
        else:
            st.write("Aucune recommandation dans l'historique.")

        # Calculer et afficher les métriques de performance
        accuracy = calculate_accuracy(true_labels, predictions)
        precision = calculate_precision(true_labels, predictions, relevant_indices)
        recall = calculate_recall(true_labels, predictions, relevant_indices)
        f1_score = calculate_f1_score(true_labels, predictions, relevant_indices)

        st.write(f"### Métriques de Performance")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1-score:** {f1_score:.2f}")
    else:
        st.error("Erreur lors du chargement des données. Veuillez vérifier le fichier CSV et réessayer.")

if __name__ == "__main__":
    main()
