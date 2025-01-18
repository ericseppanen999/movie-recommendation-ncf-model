from django.shortcuts import render
from .recommendation_logic import get_recommendations
import pandas as pd
import random
import requests
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from django.core.cache import cache
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY=os.getenv('TMDB_API_KEY')

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_PATH=os.path.join(BASE_DIR,'recommender/static/data/movies.csv')
RATINGS_PATH=os.path.join(BASE_DIR,'recommender/static/data/ratings.csv')
LINKS_PATH=os.path.join(BASE_DIR,'recommender/static/data/links.csv')

base_url="https://api.themoviedb.org/3/movie/"
# url = "https://api.themoviedb.org/3/movie/{}?api_key=x&language=en-US".format(movie_id)

def fetch_poster_url(tmdb_id):
    try:
        url = f"{base_url}{tmdb_id}api_key={TMDB_API_KEY}&language=en-US".format(tmdb_id)
        print(url)
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {TMDB_API_KEY}"
        }
        response = requests.get(url, headers=headers).json()
        print(response)
        poster_path = response.get('poster_path', None)
        if poster_path:
            # "https://image.tmdb.org/t/p/w185"+response['poster_path']
            return f"https://image.tmdb.org/t/p/w185{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for tmdbId {tmdb_id}: {e}")
    return None


def home(request):
    if request.method == 'GET':
        # Initialize or reset the shown movies list when loading the home page
        request.session['shown_movies'] = []
    
    if request.method == 'POST':
        user_ratings = {}
        for key, value in request.POST.items():
            if key.startswith('rating_'):  # Identify rating fields
                movie_id = int(key.split('_')[1])  # Extract movieId from the field name
                rating = request.POST.get(key)  # Safely get the value
                if rating:  # Only add the rating if it exists
                    user_ratings[movie_id] = float(rating)

        # Ensure all movies are rated
        if len(user_ratings) < 5:  # Adjust as needed for the number of movies
            return render(request, 'recommender/error.html', {'message': 'Please rate all movies before submitting.'})

        # Get recommendations
        recommendations = get_recommendations(user_ratings)

        for rec in recommendations:
            assert 0.5 <= rec["score"] <= 5.0, f"Invalid score {rec['score']} for movie {rec['title']}"
        
        # Get tmdbIds for recommended movies
        movies = pd.read_csv(MOVIES_PATH)
        links = pd.read_csv(LINKS_PATH)
        movies_with_links = movies.merge(links, on='movieId')
        
        # Add poster URLs to recommendations
        for rec in recommendations:
            try:
                movie_data = movies_with_links[movies_with_links['movieId'] == rec['movieId']].iloc[0]
                rec['poster_url'] = fetch_poster_url(movie_data['tmdbId'])
            except Exception as e:
                print(f"Error processing movie {rec['movieId']}: {e}")
                rec['poster_url'] = None  # Provide a fallback if poster fetch fails
        
        return render(request, 'recommender/results.html', {'recommendations': recommendations})


    # Load datasets
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)

    # Get the top 100 most popular movies
    popular_movies = ratings.groupby('movieId').size().reset_index(name='num_ratings')
    popular_movies = popular_movies.sort_values(by='num_ratings', ascending=False)
    popular_movies = popular_movies.merge(movies, on='movieId')

    # Join with links dataset to fetch tmdbId
    popular_movies = popular_movies.merge(links, on='movieId')

    # Select top 100 movies and sample 5
    top_100_movies = popular_movies.head(100)
    sampled_movies = top_100_movies.sample(5)

    # Fetch poster URLs for the sampled movies
    sampled_movies['poster_url'] = sampled_movies['tmdbId'].apply(fetch_poster_url)
    print(sampled_movies)
    movies_list = sampled_movies.to_dict('records')  # Convert DataFrame to list of dictionaries

    return render(request, 'recommender/home.html', {
        'movies': movies_list,
        'stars_range': range(5, 0, -1),  # Precompute the range
    })

@require_POST
def skip_movie(request, movie_id):
    # Load datasets
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)

    # Get or initialize the set of shown movie IDs from the session
    shown_movies = request.session.get('shown_movies', [])
    
    # Get the top 100 most popular movies
    popular_movies = ratings.groupby('movieId').size().reset_index(name='num_ratings')
    popular_movies = popular_movies.sort_values(by='num_ratings', ascending=False)
    popular_movies = popular_movies.merge(movies, on='movieId')
    popular_movies = popular_movies.merge(links, on='movieId')

    # Get one random movie from top 100 that hasn't been shown yet
    top_100_movies = popular_movies.head(100)
    available_movies = top_100_movies[~top_100_movies['movieId'].isin(shown_movies)]
    
    # If we've shown all top 100 movies, reset the shown movies list
    if len(available_movies) == 0:
        shown_movies = [movie_id]  # Keep only the current movie in the list
        available_movies = top_100_movies[~top_100_movies['movieId'].isin(shown_movies)]
    
    new_movie = available_movies.sample(1).iloc[0]
    
    # Add the new movie to the shown movies list
    shown_movies.append(int(new_movie['movieId']))
    request.session['shown_movies'] = shown_movies
    
    # Fetch poster URL for the new movie
    poster_url = fetch_poster_url(new_movie['tmdbId'])
    
    return JsonResponse({
        'movieId': int(new_movie['movieId']),
        'title': new_movie['title'],
        'poster_url': poster_url
    })
