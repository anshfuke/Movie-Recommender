import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

st.set_page_config(page_title="Bollywood Movie Recommender", layout="centered")

# Load and preprocess data

@st.cache_data
def load_and_prepare_data():
    movies_df = pd.read_csv("BollywoodMovieDetail.csv")
    movies_df['genre'] = movies_df['genre'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else x)
    movies_df['actors'] = movies_df['actors'].fillna('')
    movies_df['description'] = movies_df['title']
    movies_df['releaseYear'] = pd.to_numeric(movies_df['releaseYear'], errors='coerce').fillna(0).astype(int)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(movies_df['description']).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(movies_df['genre'])     #to convert genre (categorical data) into numerical data
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder, movies_df

# Build model

@st.cache_resource
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),  
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')   #layer for genre prediction
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Recommendation Logic

def recommend_movie(user_inputs, actor_pref, actress_pref, start_year, end_year, vectorizer, label_encoder, model, movies_df):
    user_vectors = vectorizer.transform(user_inputs).toarray()
    predictions = model.predict(user_vectors)
    predicted_genres = [label_encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

    unique_genres = list(set(predicted_genres))
    recommended_movies = movies_df[movies_df['genre'].isin(unique_genres)]

    if actor_pref:
        recommended_movies = recommended_movies[recommended_movies['actors'].str.contains(actor_pref, case=False, na=False)]

    if actress_pref:
        recommended_movies = recommended_movies[recommended_movies['actors'].str.contains(actress_pref, case=False, na=False)]

    recommended_movies = recommended_movies[
        (recommended_movies['releaseYear'] >= start_year) &
        (recommended_movies['releaseYear'] <= end_year)
    ]

    num_movies_to_return = min(20, len(recommended_movies))
    recommended_movies = recommended_movies.sample(n=num_movies_to_return, random_state=np.random.randint(0, 1000))

    return unique_genres, recommended_movies[['title', 'releaseYear']].values.tolist()

# Streamlit App UI

st.title('Bollywood Movie Recommender')

st.write("""
Enter 2–3 movie descriptions you like, your actor or actress preferences, and a release year range.
The model will recommend movies based on your input.
""")

# Sidebar content
st.sidebar.header("Project Info")
st.sidebar.write("""
This is a Bollywood movie recommender system that uses genre prediction powered by a deep learning model trained on movie titles.
""")

# Load + Train 
X_train, X_test, y_train, y_test, vectorizer, label_encoder, movies_df = load_and_prepare_data()
model = build_model(X_train.shape[1], y_train.shape[1])

if 'model_trained' not in st.session_state:
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    st.session_state['model_trained'] = True
    st.success("Model trained successfully.")


user_inputs = []
for i in range(3):
    desc = st.text_input(f"Movie {i+1} description (optional):", "")
    if desc:
        user_inputs.append(desc.lower())

actor_pref = st.text_input("Favorite actor (optional):", "").strip().lower()
actress_pref = st.text_input("Favorite actress (optional):", "").strip().lower()
start_year = st.number_input("Earliest release year:", min_value=1900, max_value=2025, value=2000)
end_year = st.number_input("Latest release year:", min_value=1900, max_value=2025, value=2025)


if st.button('Get Recommendations'):
    if user_inputs:
        predicted_genres, recommendations = recommend_movie(
            user_inputs, actor_pref, actress_pref, start_year, end_year,
            vectorizer, label_encoder, model, movies_df
        )

        st.subheader("Predicted Genres")
        st.write(", ".join(predicted_genres))

        st.subheader("Recommended Movies")
        if recommendations:
            for title, year in recommendations:
                st.write(f"- {title} ({year})")
        else:
            st.warning("No matching movies found with the given filters.")
    else:
        st.warning("Please enter at least one movie description.")
