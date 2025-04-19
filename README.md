# Cine Buddy

### **Software Lab 2 Project**

**Team Members:**
- Kartik Fuke (13)
- Himanshu Khobaragade (04)
- Himanshu Loriya (05)

## **Problem Statement**

Bollywood's large database of movies poses a challenge for users to access movies that cater to their interest. Classical recommenders based on user ratings may not be efficient for new users. This project creates a **content-based movie recommender** that generates movie genres based on descriptions and offers personalized recommendations based on **actor, actress, and release year preferences**, delivering a more relevant discovery experience.

## **Project Overview**

This project aims to develop a personalized Bollywood movie recommendation system that:
1. **Predicts genres** based on the movie description using deep learning techniques.
2. Recommends movies that match the user's **preferences for actors, actresses, and release years**.
3. Provides a tailored movie discovery experience even for new users who don't have rating history.

## **Technologies Used**

- **Python** (Programming Language)
- **TensorFlow** (Deep Learning Framework)
- **Streamlit** (Web App Framework)
- **scikit-learn** (Machine Learning Tools)
- **Pandas** (Data Manipulation)
- **NumPy** (Numerical Computing)

## **How It Works**

1. **Data Preparation**:
   - The dataset, `BollywoodMovieDetail.csv`, contains movie descriptions, genres, actors, actresses, and release years.
   - The data is preprocessed by extracting movie genres, handling missing data, and converting text descriptions into TF-IDF features.

2. **Model**:
   - A deep learning model is built using the **Keras Sequential API** with fully connected layers and dropout layers for regularization.
   - The model is trained to predict movie genres based on the movie description text.

3. **User Interaction**:
   - The user provides **2-3 movie descriptions**, along with **actor**, **actress**, and **release year** preferences.
   - The model predicts the genres of the movies based on their descriptions and filters movie recommendations based on user preferences.

4. **Recommendation System**:
   - The system generates a list of recommended movies based on the predicted genres, actors/actresses, and release years provided by the user.

## **Features**

- **Movie Description-Based Genre Prediction**: The system predicts the genre of the movie based on its description.
- **Actor and Actress Filtering**: Recommendations can be filtered by specific actors or actresses.
- **Release Year Filtering**: Recommendations can be filtered by a user-defined range of movie release years.
- **User-Friendly Interface**: Built using **Streamlit** for easy interaction with the model and viewing of recommendations.
