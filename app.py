from flask import Flask, render_template, request, jsonify, session
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import date, datetime
import os
from collections import Counter

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load models and data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
import gzip
import pickle

with gzip.open('similarity.pkl.gz', 'rb') as f:
    similarity = pickle.load(f)

clf = pickle.load(open('model2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Constants
TMDB_API_KEY = "4158f8d4403c843543d3dc953f225d77"
WATCHLIST_FILE = 'watchlist.json'

# Mood and seasonal categories
MOOD_CATEGORIES = {
    'happy': ['Comedy', 'Animation', 'Family'],
    'sad': ['Drama', 'Romance'],
    'excited': ['Action', 'Adventure', 'Thriller'],
    'relaxed': ['Documentary', 'Family'],
    'romantic': ['Romance', 'Comedy']
}

SEASONAL_CATEGORIES = {
    'christmas': {
        'keywords': ['christmas', 'holiday'],
        'genres': ['Family', 'Comedy', 'Romance'],
        'exclude_genres': ['Horror', 'Thriller']
    },
    'halloween': {
        'keywords': ['halloween', 'horror'],
        'genres': ['Horror', 'Thriller', 'Mystery']
    },
    'valentine': {
        'keywords': ['romance', 'love'],
        'genres': ['Romance', 'Comedy', 'Drama']
    },
    'summer': {
        'keywords': ['summer', 'vacation'],
        'genres': ['Action', 'Adventure', 'Comedy']
    }
}

# Initialize watchlist file
if not os.path.exists(WATCHLIST_FILE):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump([], f)


def get_movie_details(movie_id):
    """Get comprehensive movie details including crew, reviews, and trailers"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US")
        if response.status_code != 200:
            return None

        movie_data = response.json()

        # Get crew info
        crew_names, crew_images = get_crew(movie_id)

        # Get trailer
        trailer_key = get_trailer(movie_id)

        # Get reviews and sentiment
        reviews = get_reviews_with_sentiment(movie_id)

        return {
            'basic_info': movie_data,
            'crew_names': crew_names,
            'crew_images': crew_images,
            'trailer': trailer_key,
            'reviews': reviews
        }
    except Exception as e:
        print(f"Error getting movie details: {str(e)}")
        return None


def get_crew(movie_id):
    """Get crew information"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}")
        data = response.json()
        crew_names = []
        crew_images = []

        if response.status_code == 200 and data.get('cast'):
            for i, cast_member in enumerate(data['cast']):
                if i >= 6:  # Limit to 6 cast members
                    break
                if cast_member.get('profile_path'):
                    crew_names.append(cast_member.get('name', 'Unknown'))
                    crew_images.append(f"https://image.tmdb.org/t/p/w500{cast_member['profile_path']}")

        return crew_names, crew_images
    except Exception as e:
        print(f"Error getting crew: {str(e)}")
        return [], []


def get_trailer(movie_id):
    """Get movie trailer"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}")
        data = response.json()
        if response.status_code == 200 and data.get('results'):
            for video in data['results']:
                if video['site'] == 'YouTube' and video['type'] == 'Trailer':
                    return video['key']
        return None
    except Exception as e:
        print(f"Error getting trailer: {str(e)}")
        return None


def get_reviews_with_sentiment(movie_id):
    """Get movie reviews with sentiment analysis"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}")
        data = response.json()
        reviews = []

        if response.status_code == 200 and data.get('results'):
            for review in data['results'][:5]:  # Limit to 5 reviews
                review_text = review.get('content', '')
                if review_text:
                    sentiment = analyze_sentiment(review_text)
                    reviews.append({
                        'text': review_text,
                        'sentiment': sentiment
                    })
        return reviews
    except Exception as e:
        print(f"Error getting reviews: {str(e)}")
        return []


def analyze_sentiment(text):
    """Analyze sentiment of text"""
    try:
        vector = vectorizer.transform([text])
        prediction = clf.predict(vector)
        return 'Positive' if prediction[0] else 'Negative'
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 'Unknown'


def load_watchlist():
    """Load watchlist from JSON file"""
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading watchlist: {str(e)}")
        return []


def save_watchlist(watchlist):
    """Save watchlist to JSON file"""
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(watchlist, f)
        return True
    except Exception as e:
        print(f"Error saving watchlist: {str(e)}")
        return False


def get_similar_recommendations(movie_title, offset=0):
    """Get similar movie recommendations"""
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

        start_idx = (offset % (len(movies_list) - 1)) + 1
        movies_list = movies_list[start_idx:start_idx + 6]

        recommendations = []
        for i in movies_list:
            movie_info = get_movie_details(int(movies.iloc[i[0]].movie_id))
            if movie_info:
                recommendations.append({
                    'title': movies.iloc[i[0]].title,
                    'similarity': float(i[1]),
                    'info': movie_info
                })

        return recommendations
    except Exception as e:
        print(f"Error in get_similar_recommendations: {str(e)}")
        return []


def get_mood_recommendations(mood, offset=0):
    """Get mood-based recommendations"""
    try:
        if mood not in MOOD_CATEGORIES:
            return []

        preferred_genres = MOOD_CATEGORIES[mood]
        recommendations = []
        movies_processed = 0

        for _, movie in movies.iloc[offset:].iterrows():
            try:
                movie_info = get_movie_details(int(movie.movie_id))
                if not movie_info:
                    continue

                movie_genres = [g['name'] for g in movie_info['basic_info'].get('genres', [])]
                if any(genre in preferred_genres for genre in movie_genres):
                    recommendations.append({
                        'title': movie.title,
                        'info': movie_info
                    })

                movies_processed += 1
                if len(recommendations) >= 6:
                    break

            except Exception as e:
                continue

        return recommendations
    except Exception as e:
        print(f"Error in get_mood_recommendations: {str(e)}")
        return []


def get_seasonal_recommendations(season, offset=0):
    """Get seasonal recommendations"""
    try:
        if season not in SEASONAL_CATEGORIES:
            return []

        criteria = SEASONAL_CATEGORIES[season]
        recommendations = []
        movies_processed = 0

        for _, movie in movies.iloc[offset:].iterrows():
            try:
                movie_info = get_movie_details(int(movie.movie_id))
                if not movie_info:
                    continue

                movie_genres = [g['name'] for g in movie_info['basic_info'].get('genres', [])]

                has_required_genre = any(genre in criteria['genres'] for genre in movie_genres)
                no_excluded_genre = not any(genre in criteria.get('exclude_genres', [])
                                            for genre in movie_genres)

                if has_required_genre and no_excluded_genre:
                    recommendations.append({
                        'title': movie.title,
                        'info': movie_info
                    })

                movies_processed += 1
                if len(recommendations) >= 6:
                    break

            except Exception as e:
                continue

        return recommendations
    except Exception as e:
        print(f"Error in get_seasonal_recommendations: {str(e)}")
        return []


# Routes
@app.route('/')
def home():
    """Home page"""
    return render_template('index.html',
                           movies=movies['title'].values.tolist(),
                           moods=list(MOOD_CATEGORIES.keys()),
                           seasons=list(SEASONAL_CATEGORIES.keys()))


@app.route('/movie_details/<movie_title>')
def movie_details(movie_title):
    """Movie details page"""
    try:
        movie = movies[movies['title'] == movie_title].iloc[0]
        details = get_movie_details(movie.movie_id)
        if details:
            return render_template('movie_details.html',
                                   movie=movie,
                                   details=details)
        return "Movie not found", 404
    except Exception as e:
        return str(e), 500


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get movie recommendations"""
    try:
        data = request.get_json()
        rec_type = data.get('type', 'similar')
        offset = int(data.get('offset', 0))

        if rec_type == 'similar':
            movie_title = data.get('movie')
            if not movie_title:
                return jsonify({'error': 'No movie title provided'}), 400
            recommendations = get_similar_recommendations(movie_title, offset)

        elif rec_type == 'mood':
            mood = data.get('mood')
            if not mood:
                return jsonify({'error': 'No mood specified'}), 400
            recommendations = get_mood_recommendations(mood, offset)

        elif rec_type == 'seasonal':
            season = data.get('season')
            if not season:
                return jsonify({'error': 'No season specified'}), 400
            recommendations = get_seasonal_recommendations(season, offset)

        else:
            return jsonify({'error': 'Invalid recommendation type'}), 400

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/watchlist')
def view_watchlist():
    """View watchlist page"""
    try:
        watchlist = load_watchlist()
        watchlist_movies = []

        for movie_title in watchlist:
            try:
                movie = movies[movies['title'] == movie_title].iloc[0]
                movie_info = get_movie_details(int(movie.movie_id))
                if movie_info:
                    watchlist_movies.append({
                        'title': movie_title,
                        'info': movie_info
                    })
            except Exception as e:
                continue

        return render_template('watchlist.html', movies=watchlist_movies)
    except Exception as e:
        return render_template('watchlist.html', movies=[], error=str(e))


@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    """Add movie to watchlist"""
    try:
        data = request.get_json()
        movie_title = data.get('movie')

        if not movie_title:
            return jsonify({'error': 'No movie title provided'}), 400

        watchlist = load_watchlist()
        if movie_title in watchlist:
            return jsonify({'message': 'Movie already in watchlist'}), 200

        watchlist.append(movie_title)
        if save_watchlist(watchlist):
            return jsonify({'message': 'Added to watchlist successfully'}), 200
        else:
            return jsonify({'error': 'Error saving to watchlist'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/remove_from_watchlist', methods=['POST'])
def remove_from_watchlist():
    """Remove movie from watchlist"""
    try:
        data = request.get_json()
        movie_title = data.get('movie')

        if not movie_title:
            return jsonify({'error': 'No movie title provided'}), 400

        watchlist = load_watchlist()
        if movie_title in watchlist:
            watchlist.remove(movie_title)
            if save_watchlist(watchlist):
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'error': 'Error saving watchlist'}), 500

        return jsonify({'message': 'Movie not in watchlist'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# In your app.py
@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal Server Error"), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page Not Found"), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)