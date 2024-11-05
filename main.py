import re

import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import requests
from datetime import date, datetime
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go


st.set_page_config(page_title="Recommender system", layout="wide")


# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



@st.cache_data
# Functions for getting movie information from TMDB API
def crew(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}/credits?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    crew_name = []
    final_cast = []
    k = 0
    if 'cast' in data:
        for i in data["cast"]:
            if k != 6 and i['profile_path'] is not None:
                crew_name.append(i['name'])
                final_cast.append("https://image.tmdb.org/t/p/w500/" + i['profile_path'])
                k += 1
            else:
                break
    return crew_name, final_cast



def date(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    if 'release_date' in data:
        return data['release_date']
    else:
        return "Release date not available"


def genres(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    if 'genres' in data:
        return data['genres']
    else:
        return []

def overview(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    if 'overview' in data:
        return data['overview']
    else:
        return "Overview not available"


def poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(movie_id))
    data = response.json()
    if 'poster_path' in data:
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    else:
        return ""


def rating(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    if 'vote_average' in data:
        return data['vote_average']
    else:
        return "Rating not available"



def review(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}/reviews?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US&page=1".format(movie_id))
    data = response.json()
    reviews = []
    for i in data['results'][:3]:
        reviews.append(i['content'])

    if len(reviews) > 0:
        return reviews
    else:
        return "No reviews found for this movie."

# Define the function to get movie reviews
def get_reviews(movie_id):

    # Create the URL for the API request with the provided movie ID
    url = 'https://api.themoviedb.org/3/movie/{}/reviews?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US&page=1'.format(movie_id)

    try:
        # Send a GET request to the API with the URL and check for any errors
        response = requests.get(url)
        response.raise_for_status()
        # Get the reviews from the API response and store them in a list
        results = response.json().get('results')
        # If there are no reviews, print a message and return None
        if not results:
            print('No reviews found for this movie.')
            return None

        # Initialize two empty lists to store the reviews and their sentiment
        reviews_list = []
        reviews_status = []

        # Loop through each review in the results list
        for review in results:
            # Get the text content of the review
            review_text = review.get('content')
            # If the review has text content, add it to the reviews_list and predict the sentiment
            if review_text:
                reviews_list.append(review_text)
                # passing the review to our model
                movie_review_list = np.array([review_text])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                # Append the sentiment ('Positive' or 'Negative') to the reviews_status list
                reviews_status.append('Positive' if pred else 'Negative')

        # Combine the reviews and their sentiment into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

        # Loop through each review and its sentiment in the dictionary and print it
        for review, status in movie_reviews.items():
            print('{} - {}'.format(status, review))
        # Return the dictionary of reviews and their sentiment
        return movie_reviews

    # Catch any HTTP errors and print an error message
    except requests.exceptions.HTTPError as e:
        print('Error retrieving reviews: {}'.format(e))
        return None



def trailer(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}/videos?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    if 'results' in data and data['results']:
        return data['results'][0]['key']
    else:
        return "Trailer not available"



def get_watch_providers(movie_id):
    """Get streaming/watch providers for the movie"""
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key=4158f8d4403c843543d3dc953f225d77"
    )
    data = response.json()
    providers = data.get('results', {}).get('US', {})
    return {
        'stream': providers.get('flatrate', []),
        'rent': providers.get('rent', []),
        'buy': providers.get('buy', [])
    }


class MovieRecommenderBot:
    def __init__(self, movies_df, similarity):
        self.movies_df = movies_df
        self.similarity = similarity
        self.mood_keywords = {
            'happy': ['comedy', 'animation', 'family', 'musical'],
            'sad': ['drama', 'romance', 'indie'],
            'excited': ['action', 'adventure', 'sci-fi', 'thriller'],
            'scared': ['horror', 'thriller', 'mystery'],
            'relaxed': ['documentary', 'comedy', 'family'],
            'thoughtful': ['drama', 'documentary', 'history', 'mystery']
        }

        self.holiday_recommendations = {
            'christmas': ['holiday', 'family', 'comedy', 'romance'],
            'halloween': ['horror', 'thriller', 'mystery'],
            'valentine': ['romance', 'comedy', 'drama'],
            'thanksgiving': ['family', 'comedy', 'drama'],
            'summer': ['action', 'adventure', 'comedy']
        }

    def get_movie_info(self, movie_id):
        """Get combined movie information using your existing functions"""
        return {
            'genres': genres(movie_id),
            'rating': rating(movie_id),
            'release_date': date(movie_id)
        }

    def recommend_by_mood(self, mood):
        """Recommend movies based on user's mood"""
        preferred_genres = self.mood_keywords.get(mood.lower(), [])
        recommended_movies = []

        for _, movie in self.movies_df.iterrows():
            movie_info = self.get_movie_info(movie.movie_id)
            movie_genres = [genre['name'].lower() for genre in movie_info['genres']]

            if any(genre in preferred_genres for genre in movie_genres):
                if movie_info['rating'] > 7.0:  # Only recommend highly-rated movies
                    recommended_movies.append({
                        'title': movie.title,
                        'rating': movie_info['rating'],
                        'genres': movie_genres,
                        'poster': poster(movie.movie_id)
                    })

        return sorted(recommended_movies, key=lambda x: x['rating'], reverse=True)[:5]

    def recommend_movie_combination(self, movie1, movie2):
        """Recommend movies based on two movies the user likes"""
        try:
            idx1 = self.movies_df[self.movies_df['title'] == movie1].index[0]
            idx2 = self.movies_df[self.movies_df['title'] == movie2].index[0]

            combined_similarity = (self.similarity[idx1] + self.similarity[idx2]) / 2
            movie_indices = combined_similarity.argsort()[::-1][1:6]

            recommendations = []
            for idx in movie_indices:
                movie = self.movies_df.iloc[idx]
                movie_info = self.get_movie_info(movie.movie_id)
                recommendations.append({
                    'title': movie.title,
                    'rating': movie_info['rating'],
                    'genres': [genre['name'] for genre in movie_info['genres']],
                    'poster': poster(movie.movie_id)
                })

            return recommendations
        except IndexError:
            return []

    def recommend_marathon(self, theme):
        """Recommend movie marathon based on theme"""
        if theme.lower() in self.holiday_recommendations:
            preferred_genres = self.holiday_recommendations[theme.lower()]
            marathon_movies = []

            for _, movie in self.movies_df.iterrows():
                movie_info = self.get_movie_info(movie.movie_id)
                movie_genres = [genre['name'].lower() for genre in movie_info['genres']]

                if any(genre in preferred_genres for genre in movie_genres):
                    if movie_info['rating'] > 7.5:
                        marathon_movies.append({
                            'title': movie.title,
                            'rating': movie_info['rating'],
                            'genres': movie_genres,
                            'poster': poster(movie.movie_id)
                        })

            return sorted(marathon_movies, key=lambda x: x['rating'], reverse=True)[:5]
        return []

    def process_user_input(self, user_input):
        """Process user input and determine intent"""
        user_input = user_input.lower()

        # Check for mood-based recommendation request
        for mood in self.mood_keywords.keys():
            if mood in user_input or f"feeling {mood}" in user_input:
                return {
                    'intent': 'mood',
                    'value': mood,
                    'recommendations': self.recommend_by_mood(mood)
                }

        # Check for movie combination request
        movie_combo_pattern = r"if i like (.*) and (.*)"
        combo_match = re.search(movie_combo_pattern, user_input)
        if combo_match:
            movie1, movie2 = combo_match.groups()
            return {
                'intent': 'combination',
                'movies': (movie1.strip(), movie2.strip()),
                'recommendations': self.recommend_movie_combination(movie1.strip(), movie2.strip())
            }

        # Check for marathon request
        for holiday in self.holiday_recommendations.keys():
            if holiday in user_input or f"{holiday} marathon" in user_input:
                return {
                    'intent': 'marathon',
                    'theme': holiday,
                    'recommendations': self.recommend_marathon(holiday)
                }

        return {'intent': 'unknown'}


def recommend(movie):
    try:
        # Get the index of the selected movie
        movie_index = movies[movies['title'] == movie].index[0]

        # Get cosine similarity scores of the selected movie with all other movies
        cosine_angles = similarity[movie_index]

        # Get the 7 movies with highest similarity scores
        recommended_movies = sorted(list(enumerate(cosine_angles)), reverse=True, key=lambda x: x[1])[0:7]

        # Initialize lists to store recommended movies' details
        final = []
        final_posters = []

        # Get the crew details (director, cast) of the selected movie
        final_name , final_cast = crew(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the genres of the selected movie
        gen = genres(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the overview of the selected movie
        overview_final = overview(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the release date of the selected movie
        rel_date = date(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the ratings of the selected movie
        ratings = rating(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the reviews of the selected movie
        re4view = get_reviews(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the average rating of the selected movie
        rev = rating(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Get the trailer link of the selected movie
        trailer_final = trailer(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        # Loop through recommended movies to store their details
        for i in recommended_movies:
            # Store recommended movie title, rating, and score
            title = movies.iloc[i[0]].title
            ratingg = rating(movies.iloc[i[0]].movie_id)
            score = round(i[1], 2)
            final.append(f"{title} (Rating: {ratingg}) - Similarity score: {score}")


            # Store recommended movie poster
            final_posters.append(poster(movies.iloc[i[0]].movie_id))



        # Return all details
        return final_name , final_cast , rel_date , gen , overview_final , final , final_posters, ratings, re4view, rev, trailer_final

    # Catch index error when selected movie not found in dataset
    except IndexError:
        return None




movies_dict = pickle.load(open('movies_dict.pkl' , 'rb' ))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl' , 'rb'))

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.title('Movie Recommendation System')


filename = 'model2.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))


selected_movie = st.selectbox(
    'Which Movie Do you like?',
     movies['title'].values)



# This function takes a list of genres and returns a list of genre names
def process(genre):
    final = []
    for i in genre:
        final.append(i['name'])

    return final

# When the user clicks the "Search" button in the web app, this code runs
if st.button('Search'):
    result = recommend(selected_movie)

     # If the movie is not found in the database, an error message is displayed
    if result is None:
        st.error("Sorry, the movie you requested is not in our database. Please check the spelling or try with some other movies.")
    else:
         # Extracts the necessary details about the movie from the result
        name, cast, rel_date, gen, overview_final, ans, posters, ratings, re4view, rev, trailer_final = result[:11]

        # Display the movie details in a header and two columns
        st.header(selected_movie)
        col_1, col_2 = st.columns(2)
        with col_1:
            if posters:
                st.image(posters[0], width=325, use_column_width=325)
            else:
                st.write("Poster not available")

        with col_2:
            st.write("Title : {} ".format(ans[0]))

            st.write("Overview : {} ".format(overview_final))
            gen = process(gen)
            gen = " , ".join(gen)
            st.write("Genres : {}".format(gen))
            st.write("Release Date {} : {} ".format(" " , rel_date))
            st.write("Ratings : {} ".format(ratings))

        # Displays the top 6 cast members in a row of images with their names
        st.title("Top Casts")

        c1 , c2 , c3 = st.columns(3)
        if len(cast) >= 6 and len(name) >= 6:
            with c1:
                st.image(cast[0], width=225, use_column_width=225)
                st.caption(name[0])
            with c2:
                st.image(cast[1], width=225, use_column_width=225)
                st.caption(name[1])
            with c3:
                st.image(cast[2], width=225, use_column_width=225)
                st.caption(name[2])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(cast[3], width=225, use_column_width=225)
                st.caption(name[3])
            with c2:
                st.image(cast[4], width=225, use_column_width=225)
                st.caption(name[4])
            with c3:
                st.image(cast[5], width=225, use_column_width=225)
                st.caption(name[5])
        else:
            st.warning("Not enough cast members to display.")
        # Displays the trailer for the movie using a YouTube link
        st.title("  Trailer")
        st.video("https://www.youtube.com/watch?v={}".format(trailer_final))

       # Check if there are any reviews
        if re4view:
            # plot a bar graph of the reviews
            pos_count = 0
            neg_count = 0
            for review in re4view.values():
                if review == 'Positive':
                    pos_count += 1
                else:
                    neg_count += 1


            # Plotting the bar graph
            fig, ax = plt.subplots(dpi=50)
            ax.bar(['Positive', 'Negative'], [pos_count, neg_count])
            ax.set_title('Sentiment Analysis of Reviews')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Number of Reviews')
            st.pyplot(fig)

            # Create a dataframe from the reviews dictionary
            df = pd.DataFrame.from_dict((re4view), orient='index', columns=['Sentiment'])

            # Display the reviews in a table
            # Display the dataframe in Streamlit
            st.write("Reviews:")
            styled_table = df.style.set_table_styles([{'selector': 'tr', 'props': [('background-color', 'white')]},
                                                      {'selector': 'th', 'props': [('background-color', 'lightgrey')]},
                                                      {'selector': 'td', 'props': [('color', 'black')]}])
            styled_table.highlight_max(axis=0)
            st.table(styled_table)
        else:
            st.write("No reviews found for this movie.")


        st.title("")


        st.title("   Similar Movies You May Like")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(posters[1], width=225, use_column_width=225)
            st.write(ans[1])


        with c2:
            st.image(posters[2], width=225, use_column_width=225)
            st.write(ans[2])


        with c3:
            st.image(posters[3], width=225, use_column_width=225)
            st.write(ans[3])


        c1, c2, c3 = st.columns(3)
        with c1:
            if posters[4] is not None:
                st.image(posters[4], width=225, use_column_width=225)
                st.write(ans[4])

        with c2:
            if posters[5] is not None:
                st.image(posters[5], width=225, use_column_width=225)
                st.write(ans[5])

        with c3:
            if posters[6] is not None:
                st.image(posters[6], width=225, use_column_width=225)
                st.write(ans[6])







import os

# Create a file to store the watchlist
WATCHLIST_FILE = 'watchlist.txt'
if not os.path.exists(WATCHLIST_FILE):
    open(WATCHLIST_FILE, 'w').close()

# Load the watchlist from the file
with open(WATCHLIST_FILE, 'r') as f:
    watchlist = f.read().splitlines()

# Create a form to add a movie to the watchlist
with st.form(key='add_movie_form'):
    movie_title = st.text_input(label='WATCHLIST', value=selected_movie)
    add_movie = st.form_submit_button(label='Add Movie')

# If the add movie button is clicked and the movie title is not empty, add the movie to the watchlist
if add_movie and movie_title:
    if movie_title not in watchlist:
        watchlist.append(movie_title)
        with open(WATCHLIST_FILE, 'w') as f:
            f.write('\n'.join(watchlist))
        st.success(f'{movie_title} added to Watchlist!')
    else:
        st.warning(f'{movie_title} is already in the Watchlist!')

# If the add movie button is clicked and the movie title is empty, show an error message
if add_movie and not movie_title:
    st.error('Please enter a movie title.')

# Create a form to remove a movie from the watchlist
with st.form(key='remove_movie_form'):
    # Display the watchlist as a selectbox
    selected_movie = st.selectbox('Select a movie to remove from Watchlist', watchlist)
    remove_movie = st.form_submit_button(label='Remove Movie')

# If the remove movie button is clicked, remove the selected movie from the watchlist

if remove_movie and selected_movie:
    watchlist.remove(selected_movie)
    with open(WATCHLIST_FILE, 'w') as f:
        f.write('\n'.join(watchlist))
    st.success(f'{selected_movie} removed from Watchlist!')


# Display the watchlist with movie details
if watchlist:
    movie_data = []
    for movie in watchlist:
        # Make a request to the TMDb API to get movie details
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie?api_key=4158f8d4403c843543d3dc953f225d77&query={}".format(
                movie))
        if response.status_code == 200:
            results = response.json()['results']
            if len(results) > 0:
                data = results[0]
                movie_data.append(data)
    if movie_data:
        df = pd.DataFrame(movie_data, columns=['title', 'overview'])
        st.write(df)
    else:
        st.write('No movie details found.')
else:
    st.write('Watchlist is empty.')



import streamlit as st

# Add social sharing buttons
import urllib.parse

def add_social_buttons(selected_movie):
    url = selected_movie
    twitter_text = 'Check out this cool movie'
    whatsapp_text = 'Check out this cool movie: {}'.format(url)
    whatsapp_text_encoded = urllib.parse.quote(whatsapp_text)
    twitter_text_encoded = urllib.parse.quote(twitter_text)

    st.sidebar.subheader('Share')
    st.sidebar.write('Share this app with your friends and colleagues:')

    tweet_btn = st.sidebar.button(label='Twitter')
    if tweet_btn:
        tweet_url = 'https://twitter.com/intent/tweet?text={}&url={}'.format(twitter_text_encoded, url)
        st.sidebar.markdown('[![Tweet](https://img.shields.io/twitter/url?style=social&url={})]({})'.format(tweet_url, tweet_url), unsafe_allow_html=True)

    whatsapp_btn = st.sidebar.button(label='WhatsApp')
    if whatsapp_btn:
        whatsapp_url = 'https://wa.me/?text={}'.format(whatsapp_text_encoded, url)
        st.sidebar.markdown('[![WhatsApp](https://img.shields.io/badge/WhatsApp-Chat-green?style=social&logo=whatsapp&alt=Share%20on%20WhatsApp)]({})'.format(whatsapp_url), unsafe_allow_html=True)


# Add the social sharing menu to the app
add_social_buttons(selected_movie)

# Add this after your social sharing buttons section
st.markdown("---")


class ComprehensiveMovieBot:
    def __init__(self, movies_df, similarity):
        self.movies_df = movies_df
        self.similarity = similarity

        # Seasonal/Holiday mappings
        self.seasonal_mappings = {
            'christmas': {
                'keywords': ['christmas', 'xmas', 'holiday season', 'santa', 'december 25'],
                'genres': ['Family', 'Comedy', 'Romance'],
                'exclude_genres': ['Horror', 'Thriller'],
                'mood': 'festive'
            },
            'halloween': {
                'keywords': ['halloween', 'spooky', 'october 31', 'trick or treat'],
                'genres': ['Horror', 'Thriller', 'Mystery'],
                'mood': 'scary'
            },
            'valentine': {
                'keywords': ['valentine', 'february 14', 'date night', 'romantic evening'],
                'genres': ['Romance', 'Drama', 'Comedy'],
                'mood': 'romantic'
            },
            'easter': {
                'keywords': ['easter', 'spring holiday', 'easter bunny'],
                'genres': ['Family', 'Animation', 'Adventure'],
                'mood': 'uplifting'
            },
            'summer': {
                'keywords': ['summer', 'beach', 'vacation', 'summer break'],
                'genres': ['Action', 'Adventure', 'Comedy'],
                'mood': 'fun'
            },
            'thanksgiving': {
                'keywords': ['thanksgiving', 'november', 'turkey day'],
                'genres': ['Family', 'Drama', 'Comedy'],
                'mood': 'heartwarming'
            }
        }

        # Special categories
        self.special_categories = {
            'family night': {
                'genres': ['Family', 'Animation', 'Adventure'],
                'min_rating': 7.0,
                'exclude_genres': ['Horror', 'Thriller']
            },
            'date night': {
                'genres': ['Romance', 'Comedy', 'Drama'],
                'min_rating': 7.5
            },
            'movie marathon': {
                'duration': 'long',
                'min_rating': 8.0,
                'series': True
            },
            'classic movies': {
                'year_range': (1900, 1980),
                'min_rating': 7.5
            },
            'award winners': {
                'min_rating': 8.0,
                'awards': True
            },
            'hidden gems': {
                'rating_range': (7.0, 8.5),
                'popularity': 'low'
            }
        }

        # Age ratings and content filters
        self.age_ratings = {
            'kids': {
                'genres': ['Family', 'Animation'],
                'exclude_genres': ['Horror', 'Thriller'],
                'max_rating': 'PG'
            },
            'teens': {
                'genres': ['Action', 'Adventure', 'Comedy'],
                'max_rating': 'PG-13'
            },
            'adults': {
                'all_genres': True,
                'include_rating': 'R'
            }
        }

        # Adding existing mood and genre mappings
        self.mood_mappings = {
            'happy': ['happy', 'cheerful', 'joyful', 'upbeat', 'fun', 'good', 'positive'],
            'sad': ['sad', 'depressed', 'down', 'blue', 'melancholy', 'emotional'],
            'excited': ['excited', 'thrilled', 'energetic', 'pumped', 'adventurous'],
            'relaxed': ['relaxed', 'calm', 'peaceful', 'chill', 'quiet', 'lazy'],
            'scared': ['scared', 'frightened', 'spooky', 'horror', 'terrifying'],
            'romantic': ['romantic', 'love', 'romance', 'relationship'],
            'thoughtful': ['thoughtful', 'deep', 'meaningful', 'serious', 'intelligent'],
            'angry': ['angry', 'mad', 'furious', 'rage', 'vengeful']
        }

    def get_movie_info(self, movie_id):
        """Get comprehensive movie information with better error handling"""
        try:
            movie_rating = rating(movie_id)
            # Convert rating to float if possible, otherwise keep as is
            if isinstance(movie_rating, (int, float)) or (
                    isinstance(movie_rating, str) and movie_rating.replace('.', '').isdigit()):
                movie_rating = float(movie_rating)

            return {
                'genres': genres(movie_id),
                'rating': movie_rating,
                'poster': poster(movie_id),
                'overview': overview(movie_id),
                'release_date': date(movie_id),
                'trailer': trailer(movie_id),
                'cast': crew(movie_id)[0],
                'watch_providers': get_watch_providers(movie_id)
            }
        except Exception as e:
            print(f"Error getting movie info: {str(e)}")
            return None

    # get mood based recommendations
    def get_mood_based_recommendations(self, mood):
        """Get recommendations based on mood"""
        recommendations = []
        try:
            # Get the preferred genres for this mood
            preferred_genres = []
            if mood in self.mood_mappings:
                # Map mood to genre preferences
                if mood == 'happy':
                    preferred_genres = ['Comedy', 'Animation', 'Family', 'Musical']
                elif mood == 'sad':
                    preferred_genres = ['Drama', 'Romance']
                elif mood == 'excited':
                    preferred_genres = ['Action', 'Adventure', 'Science Fiction', 'Thriller']
                elif mood == 'relaxed':
                    preferred_genres = ['Documentary', 'Family', 'Comedy']
                elif mood == 'romantic':
                    preferred_genres = ['Romance', 'Drama', 'Comedy']
                elif mood == 'thoughtful':
                    preferred_genres = ['Drama', 'Documentary', 'History']

            # Get recommendations based on preferred genres
            for _, movie in self.movies_df.iterrows():
                movie_info = self.get_movie_info(movie.movie_id)
                movie_genres = [g['name'] for g in movie_info['genres']]

                if any(genre in movie_genres for genre in preferred_genres):
                    if isinstance(movie_info['rating'], (int, float)) and movie_info['rating'] > 7.0:
                        recommendations.append({
                            'title': movie.title,
                            'info': movie_info
                        })

                if len(recommendations) >= 5:
                    break

            return recommendations
        except Exception as e:
            print(f"Error in mood recommendations: {str(e)}")
            return []

    def detect_category(self, text):
        """Detect category from user input with improved matching"""
        text = text.lower().strip()

        # Check for direct mood mentions
        for mood, keywords in self.mood_mappings.items():
            if any(keyword in text for keyword in keywords):
                return {'type': 'mood', 'category': mood}

        # Check for seasonal/holiday matches
        for season, data in self.seasonal_mappings.items():
            if any(keyword in text for keyword in data['keywords']):
                return {'type': 'seasonal', 'category': season}

        # Check for special categories
        for category in self.special_categories.keys():
            if category in text:
                return {'type': 'special', 'category': category}

        # Check for age ratings
        for age in self.age_ratings.keys():
            if age in text:
                return {'type': 'age', 'category': age}

        return None

    def get_recommendations(self, user_input):
        """Get movie recommendations based on user input"""
        try:
            # Add debug prints
            print(f"Processing input: {user_input}")

            user_input = user_input.lower().strip()
            category = self.detect_category(user_input)

            print(f"Detected category: {category}")

            recommendations = []

            if category:
                print(f"Getting recommendations for category: {category['type']}")
                if category['type'] == 'mood':
                    recommendations = self.get_mood_based_recommendations(category['category'])
                elif category['type'] == 'seasonal':
                    season_data = self.seasonal_mappings[category['category']]
                    recommendations = self.get_seasonal_recommendations(season_data)
                elif category['type'] == 'special':
                    special_data = self.special_categories[category['category']]
                    recommendations = self.get_special_recommendations(special_data)
                elif category['type'] == 'age':
                    age_data = self.age_ratings[category['category']]
                    recommendations = self.get_age_based_recommendations(age_data)

                print(f"Found {len(recommendations)} recommendations")

                valid_recommendations = []
                for rec in recommendations:
                    if isinstance(rec, dict) and 'title' in rec and 'info' in rec:
                        valid_recommendations.append(rec)
                    if len(valid_recommendations) >= 5:
                        break

                return {
                    'type': category['type'],
                    'category': category['category'],
                    'recommendations': valid_recommendations
                }
            else:
                print("No category detected, trying genre matching")
                # Try to match with general genre keywords
                general_genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'family']
                for genre in general_genres:
                    if genre in user_input:
                        recommendations = self.get_special_recommendations({
                            'genres': [genre.title()],
                            'min_rating': 7.0
                        })
                        return {
                            'type': 'genre',
                            'category': genre,
                            'recommendations': recommendations[:5]
                        }

                return {
                    'type': 'unknown',
                    'category': None,
                    'recommendations': []
                }

        except Exception as e:
            print(f"Error in recommendations: {str(e)}")
            return {
                'type': 'error',
                'category': None,
                'recommendations': []
            }
    def get_seasonal_recommendations(self, season_data):
        """Get recommendations for seasonal/holiday movies"""
        recommendations = []
        for _, movie in self.movies_df.iterrows():
            movie_info = self.get_movie_info(movie.movie_id)
            movie_genres = [g['name'] for g in movie_info['genres']]

            # Check if movie matches seasonal criteria
            if (any(g in season_data['genres'] for g in movie_genres) and
                    not any(g in season_data.get('exclude_genres', []) for g in movie_genres)):

                if isinstance(movie_info['rating'], (int, float)) and movie_info['rating'] > 7.0:
                    recommendations.append({
                        'title': movie.title,
                        'info': movie_info
                    })

        return sorted(recommendations, key=lambda x: x['info']['rating'], reverse=True)

    def get_special_recommendations(self, special_data):
        """Get recommendations for special categories with improved error handling"""
        recommendations = []
        try:
            for _, movie in self.movies_df.iterrows():
                try:
                    movie_info = self.get_movie_info(movie.movie_id)
                    if movie_info is None:
                        continue

                    if self.matches_special_criteria(movie_info, special_data):
                        recommendations.append({
                            'title': movie.title,
                            'info': movie_info
                        })

                        if len(recommendations) >= 5:
                            break

                except Exception as e:
                    print(f"Error processing movie {movie.title}: {str(e)}")
                    continue

            # Sort recommendations by rating, handling non-numeric ratings
            def get_rating(rec):
                rating = rec['info']['rating']
                if rating == "Rating not available":
                    return 0
                try:
                    return float(rating)
                except (ValueError, TypeError):
                    return 0

            return sorted(recommendations, key=get_rating, reverse=True)

        except Exception as e:
            print(f"Error in get_special_recommendations: {str(e)}")
            return []

    def matches_special_criteria(self, movie_info, criteria):
        """Check if movie matches special category criteria with improved error handling"""
        try:
            # Handle minimum rating check
            if 'min_rating' in criteria:
                movie_rating = movie_info['rating']

                # Skip movies with no rating
                if movie_rating == "Rating not available":
                    return False

                try:
                    # Convert rating to float and compare
                    movie_rating = float(movie_rating)
                    if movie_rating < float(criteria['min_rating']):
                        return False
                except (ValueError, TypeError):
                    # If rating can't be converted to float, skip this movie
                    return False

            # Handle year range check
            if 'year_range' in criteria:
                release_date = movie_info['release_date']
                if release_date == "Release date not available":
                    return False

                try:
                    release_year = int(release_date[:4])
                    year_range = criteria['year_range']
                    if not (year_range[0] <= release_year <= year_range[1]):
                        return False
                except (ValueError, TypeError, IndexError):
                    return False

            # Handle genre requirements
            if 'genres' in criteria:
                movie_genres = [g['name'] for g in movie_info.get('genres', [])]
                if not any(g in criteria['genres'] for g in movie_genres):
                    return False

            # Handle excluded genres
            if 'exclude_genres' in criteria:
                movie_genres = [g['name'] for g in movie_info.get('genres', [])]
                if any(g in criteria['exclude_genres'] for g in movie_genres):
                    return False

            return True

        except Exception as e:
            print(f"Error in matches_special_criteria: {str(e)}")
            return False

    def get_age_based_recommendations(self, age_data):
        """Get recommendations based on age rating"""
        recommendations = []
        for _, movie in self.movies_df.iterrows():
            movie_info = self.get_movie_info(movie.movie_id)
            movie_genres = [g['name'] for g in movie_info['genres']]

            if (any(g in age_data['genres'] for g in movie_genres) and
                    not any(g in age_data.get('exclude_genres', []) for g in movie_genres)):
                recommendations.append({
                    'title': movie.title,
                    'info': movie_info
                })

        return sorted(recommendations, key=lambda x: x['info']['rating'], reverse=True)

def display_recommendations(result):
    """Helper function to display movie recommendations with better error handling"""
    try:
        if result['type'] == 'error':
            st.error("An error occurred while getting recommendations. Please try again.")
            return

        if not result.get('recommendations'):
            st.warning(f"No recommendations found for {result.get('category', 'your request')}. Try another category!")
            return

        st.subheader(f"Here are your {result.get('category', '')} recommendations:")

        # Display recommendations in columns
        cols = st.columns(min(len(result['recommendations']), 5))
        for col, rec in zip(cols, result['recommendations']):
            with col:
                try:
                    if rec['info'].get('poster'):
                        st.image(rec['info']['poster'], width=150)
                    st.write(f"**{rec['title']}**")

                    # Safely handle rating display
                    rating_value = rec['info'].get('rating')
                    if rating_value == "Rating not available":
                        st.write("Rating: Not available")
                    elif rating_value and isinstance(rating_value, (int, float)):
                        st.write(f"Rating: {float(rating_value):.1f}/10")
                    else:
                        st.write(f"Rating: {rating_value}")

                    with st.expander("More Info"):
                        st.write("**Overview:**")
                        st.write(rec['info'].get('overview', 'No overview available'))

                        genres = rec['info'].get('genres', [])
                        if genres:
                            st.write("**Genres:**")
                            st.write(", ".join([g['name'] for g in genres]))

                        # Safely handle watch providers
                        providers = rec['info'].get('watch_providers', {})
                        if providers and providers.get('stream'):
                            st.write("**Where to Stream:**")
                            st.write(", ".join([p['provider_name'] for p in providers['stream']]))

                except Exception as e:
                    st.error(f"Error displaying movie: {str(e)}")
                    continue

    except Exception as e:
        st.error(f"Error displaying recommendations: {str(e)}")


st.markdown("---")
st.title("Smart Movie Recommendation Bot")

# Initialize bot once
bot = ComprehensiveMovieBot(movies, similarity)

# Create tabs for different recommendation types
tab1, tab2, tab3 = st.tabs(["Quick Recommendations", "Seasonal & Special", "Mood Based"])

with tab1:
    st.markdown("""
    ### Quick Movie Recommendations
    Tell me what kind of movies you're looking for:
    - Specific genre (e.g., "action movies")
    - Age group (e.g., "kids movies")
    - Special category (e.g., "classic movies")
    """)

    quick_input = st.text_input("What kind of movies would you like?", key="quick_input")
    if st.button("Get Quick Recommendations"):
        if quick_input:
            bot = ComprehensiveMovieBot(movies, similarity)
            result = bot.get_recommendations(quick_input)
            if result and result['recommendations']:
                display_recommendations(result)
            else:
                st.warning("No recommendations found. Try a different category!")

with tab2:
    st.markdown("""
    ### Seasonal & Special Recommendations
    Choose a category:
    """)

    season_options = [
        "Christmas Movies", "Halloween Movies", "Valentine's Day Movies",
        "Summer Blockbusters", "Family Movie Night", "Date Night Movies"
    ]

    selected_category = st.selectbox("Select a category:", season_options)
    if st.button("Get Seasonal Recommendations"):
        bot = ComprehensiveMovieBot(movies, similarity)
        result = bot.get_recommendations(selected_category.lower())
        if result and result['recommendations']:
            display_recommendations(result)
        else:
            st.warning("No recommendations found for this category.")

with tab3:
    st.markdown("""
    ### Mood Based Recommendations
    How are you feeling today?
    """)

    mood_options = [
        "Happy", "Sad", "Excited", "Relaxed", "Romantic",
        "Thoughtful", "Energetic", "Calm"
    ]

    selected_mood = st.selectbox("Select your mood:", mood_options)
    if st.button("Get Mood Recommendations"):
        bot = ComprehensiveMovieBot(movies, similarity)
        result = bot.get_recommendations(selected_mood.lower())
        if result and result['recommendations']:
            display_recommendations(result)
        else:
            st.warning("No recommendations found for this mood.")


# Add clear all button at the bottom
if st.button("Clear All", key="clear_all"):
    st.experimental_rerun()