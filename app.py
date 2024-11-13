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
app.secret_key = 'movie-recommender-dev-key'

# Load the models and data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))
clf = pickle.load(open('model2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


# Helper functions for crew and trailer info
def crew(movie_id):
    """Get crew information with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        crew_name = []
        final_cast = []

        if response.status_code == 200 and data.get('cast'):
            k = 0
            for i in data["cast"]:
                if k != 6 and i.get('profile_path'):
                    crew_name.append(i.get('name', 'Unknown'))
                    final_cast.append("https://image.tmdb.org/t/p/w500" + i['profile_path'])
                    k += 1
                if k >= 6:
                    break

        return crew_name, final_cast
    except:
        return [], []


def trailer(movie_id):
    """Get trailer link with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('results') and len(data['results']) > 0:
            return data['results'][0]['key']
        return ""
    except:
        return ""


def get_seasonal_recommendations(season, offset=0):
    """Get seasonal movie recommendations"""
    try:
        # Define seasonal criteria
        seasonal_mappings = {
            'christmas': {
                'keywords': ['christmas', 'holiday'],
                'genres': ['Family', 'Comedy', 'Romance'],
                'exclude_genres': ['Horror', 'Thriller']
            },
            'halloween': {
                'keywords': ['halloween', 'horror'],
                'genres': ['Horror', 'Thriller', 'Mystery']
            },
            'summer': {
                'keywords': ['summer', 'vacation'],
                'genres': ['Action', 'Adventure', 'Comedy']
            }
        }

        if season not in seasonal_mappings:
            return []

        criteria = seasonal_mappings[season]
        recommendations = []

        # Get movies starting from offset
        start_idx = offset
        movies_processed = 0

        for _, movie in movies.iloc[start_idx:].iterrows():
            try:
                movie_info = get_movie_info(movie.movie_id)
                if not movie_info:
                    continue

                # Check if movie matches seasonal criteria
                movie_genres = [g.get('name', '') for g in movie_info.get('genres', [])]

                if any(genre in criteria['genres'] for genre in movie_genres) and \
                        not any(genre in criteria.get('exclude_genres', []) for genre in movie_genres):
                    recommendations.append({
                        'title': movie.title,
                        'info': movie_info
                    })

                movies_processed += 1
                if len(recommendations) >= 6:  # Get 6 recommendations
                    break

            except Exception as e:
                print(f"Error processing movie {movie.title}: {str(e)}")
                continue

        return recommendations

    except Exception as e:
        print(f"Error in get_seasonal_recommendations: {str(e)}")
        return []

# API functions
def poster(movie_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        return ""
    except:
        return ""


def get_movie_info(movie_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        if response.status_code != 200:
            return None

        data = response.json()

        # Get additional info
        crew_names, crew_images = crew(movie_id)
        trailer_key = trailer(movie_id)

        return {
            'movie_id': movie_id,
            'title': data.get('title', ''),
            'genres': data.get('genres', []),
            'rating': float(data.get('vote_average', 0)),
            'poster': "https://image.tmdb.org/t/p/w500" + data['poster_path'] if data.get('poster_path') else "",
            'overview': data.get('overview', "No overview available"),
            'release_date': data.get('release_date', "Release date not available"),
            'trailer': trailer_key,
            'crew_names': crew_names,
            'crew_images': crew_images
        }
    except Exception as e:
        print(f"Error getting movie info for {movie_id}: {str(e)}")
        return None


def get_recommendations(movie_title, offset=0):
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

        # Apply offset to get different recommendations
        start_idx = offset % (len(movies_list) - 1)  # -1 to exclude the movie itself
        movies_list = movies_list[start_idx:start_idx + 6]  # Get 6 recommendations

        recommendations = []
        for i in movies_list:
            # Convert NumPy types to Python native types
            movie_info = get_movie_info(int(movies.iloc[i[0]].movie_id))  # Convert movie_id to int
            if movie_info:
                recommendations.append({
                    'title': movies.iloc[i[0]].title,
                    'similarity': float(i[1]),  # Convert numpy.float to Python float
                    'info': {
                        'movie_id': int(movie_info['movie_id']),
                        'title': str(movie_info['title']),
                        'genres': movie_info['genres'],
                        'rating': float(movie_info['rating']),
                        'poster': str(movie_info['poster']),
                        'overview': str(movie_info['overview']),
                        'release_date': str(movie_info['release_date']),
                        'trailer': str(movie_info['trailer']),
                        'crew_names': movie_info['crew_names'],
                        'crew_images': movie_info['crew_images']
                    }
                })

        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return []


# Also add a custom JSON encoder to handle any remaining NumPy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
        except Exception as e:
            print(f"Error in JSON encoding: {str(e)}")
            return str(obj)


# Update Flask app configuration to use custom encoder
app.json_encoder = CustomJSONEncoder


# Routes
@app.route('/')
def home():
    return render_template('index.html', movies=movies['title'].values.tolist())


@app.route('/get_recommendations', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        movie_title = data.get('movie')
        offset = data.get('offset', 0)

        if not movie_title:
            return jsonify({'error': 'No movie title provided'}), 400

        # Store the movie title in session for refresh functionality
        session['last_movie'] = movie_title
        session['current_offset'] = offset

        recommendations = get_recommendations(movie_title, offset)
        recommendations_json = json.loads(json.dumps(recommendations, cls=CustomJSONEncoder))

        return jsonify({'recommendations': recommendations_json})
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/refresh_recommendations', methods=['POST'])
def refresh():
    try:
        # Get the last movie from session
        last_movie = session.get('last_movie')
        if not last_movie:
            return jsonify({'error': 'No movie selected. Please select a movie first.'}), 400

        # Get and increment the offset
        current_offset = session.get('current_offset', 0)
        new_offset = (current_offset + 6) % len(movies)  # 6 is the number of recommendations we show
        session['current_offset'] = new_offset

        # Get new recommendations with the new offset
        recommendations = get_recommendations(last_movie, new_offset)
        recommendations_json = json.loads(json.dumps(recommendations, cls=CustomJSONEncoder))

        return jsonify({'recommendations': recommendations_json})
    except Exception as e:
        print(f"Error in refresh route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.get_json()
    movie_title = data.get('movie')

    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400

    watchlist = session.get('watchlist', [])
    if movie_title not in watchlist:
        watchlist.append(movie_title)
        session['watchlist'] = watchlist
        return jsonify({'message': 'Added to watchlist'})
    return jsonify({'message': 'Already in watchlist'})


if __name__ == '__main__':
    app.run(debug=True)