# Movie Recommender Project

This repository contains a simple movie recommendation web application using Django and a Neural Collaborative Filtering (NCF) model. Users can rate a set of movies, and the system will suggest new films based on their preferences. Dataset from Movielens. Utilized TMDB API for movie posters.

---

## Overview

- **Frontend/UI**: Users are presented with multiple movie cards in a grid. They can give star ratings or skip movies they haven’t seen.
- **Backend**: Django handles requests, updates the user’s temporary ratings, and returns a set of recommended movies.
- **Model**: A fresh neural model is built and trained at startup (using TensorFlow/Keras). This model predicts ratings for any user–movie pair.

---

## Key Features

1. **Movie Rating**  
   Users select a star rating (1–5) for a selection of movies. The system records these ratings.

2. **Skipping Movies**  
   Users can skip any movie, triggering a fetch for a new random movie to replace it.

3. **Neural CF Model**  
   - Trains on historical user–movie ratings (`ratings.csv`).
   - Learns separate embeddings for users and movies, then passes them through MLP layers.
   - Uses a custom `RatingScale` layer to map predictions from `[0..1]` to `[0.5..5.0]`.

4. **New-User Handling**  
   For a brand-new user, their “user embedding” is computed on the fly by averaging the embeddings of the movies they rated. A sub-model then predicts ratings for all unseen items.

5. **Variety in Recommendations**  
   After sorting the candidate predictions, the system takes the top 30, shuffles them, and finally returns the top `N` from that shuffled subset. This adds more diversity to final suggestions.

---

## Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/movie_recommender.git
   cd movie_recommender
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run database migrations** (if using Django’s default DB):

   ```bash
   python manage.py migrate
   ```

4. **Start the development server**:

   ```bash
   python manage.py runserver
   ```

5. **Access the app**:  
   Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.  

   > On startup, the model will be trained with `ratings.csv`. Then you can rate movies on the home page.

---

## How the Model Works

1. **Data**  
   - Uses `ratings.csv` (user–movie–rating) and `movies.csv` (movie metadata).
   - User and movie IDs are mapped to integer indices.

2. **NCF Architecture**  
   - **Embedding Layers**: Separate embeddings for users and items, each of size `EMBED_DIM` (e.g., 32).  
   - **MLP Layers**: After flattening and concatenating the user and movie vectors, the model applies fully connected layers with ReLU activation and Dropout for regularization.  
   - **RatingScale**: The final output is scaled from a `[0..1]` sigmoid to `[0.5..5.0]` to match typical star ratings.

3. **New User Scenario**  
   - The model is trained on known users.  
   - For a *new user*, their embedding is created by computing a weighted average of the movie embeddings they’ve rated.  
   - A special “sub-model” reuses the main movie embedding and MLP weights to predict for all unrated items.

4. **Recommendation Step**  
   - Sort all predictions for a user in descending order.  
   - Take the top 30 and shuffle for variety, then select the final top `N`.  
   - Return those movie titles and predicted scores.

---

## Potential Improvements

- **Additional Features**: Could incorporate content-based filters or hybrid approaches.  
- **User Embedding Updates**: Fine-tuning on a new user’s ratings could yield even better results.  
- **Scalability**: For large datasets, consider more efficient training strategies or distributed systems.

---

## License

This project is free to use or modify. Refer to the repository’s license file (if provided) for details.
