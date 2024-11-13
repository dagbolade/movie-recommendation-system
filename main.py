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
from datetime import date, datetime
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px

import plotly.graph_objects as go


st.set_page_config(page_title="Recommender system", layout="wide")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'skipped_movies' not in st.session_state:
    st.session_state.skipped_movies = set()
if 'last_recommendations' not in st.session_state:
    st.session_state.last_recommendations = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Quick"
if 'recommendation_offset' not in st.session_state:
    st.session_state.recommendation_offset = 0
if 'last_recommendation' not in st.session_state:
    st.session_state.last_recommendation = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'movie_info_cache' not in st.session_state:
    st.session_state.movie_info_cache = {}

# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_data
def poster(movie_id):
    """Get movie poster with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        return ""
    except:
        return ""


@st.cache_data
def overview(movie_id):
    """Get movie overview with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('overview'):
            return data['overview']
        return "Overview not available"
    except:
        return "Overview not available"


@st.cache_data
def date(movie_id):
    """Get release date with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('release_date'):
            return data['release_date']
        return "Release date not available"
    except:
        return "Release date not available"


@st.cache_data
def genres(movie_id):
    """Get genres with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('genres'):
            return data['genres']
        return []
    except:
        return []


@st.cache_data
def rating(movie_id):
    """Get rating with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('vote_average') is not None:
            return float(data['vote_average'])
        return 0.0
    except:
        return 0.0


@st.cache_data
def trailer(movie_id):
    """Get trailer with proper error handling"""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        data = response.json()
        if response.status_code == 200 and data.get('results') and len(data['results']) > 0:
            return data['results'][0]['key']
        return ""
    except:
        return ""


@st.cache_data
def crew(movie_id):
    """Get crew with proper error handling"""
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


def get_movie_info(movie_id):
    """Get all movie information with proper error handling"""
    try:
        # Make a single API call to get most of the data
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
        if response.status_code != 200:
            return None

        data = response.json()

        # Get crew info separately
        crew_names, crew_images = crew(movie_id)
        trailer_key = trailer(movie_id)

        return {
            'movie_id': movie_id,
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


# Watchlist functions
def load_watchlist():
    if os.path.exists('watchlist.txt'):
        with open('watchlist.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []


def save_watchlist(watchlist):
    with open('watchlist.txt', 'w') as f:
        f.write('\n'.join(watchlist))


def add_to_watchlist(movie_title):
    watchlist = load_watchlist()
    if movie_title not in watchlist:
        watchlist.append(movie_title)
        save_watchlist(watchlist)
        return True
    return False


def remove_from_watchlist(movie_title):
    watchlist = load_watchlist()
    if movie_title in watchlist:
        watchlist.remove(movie_title)
        save_watchlist(watchlist)
        return True
    return False


def display_watchlist():
    st.sidebar.title("My Watchlist")
    watchlist = load_watchlist()

    if watchlist:
        for movie in watchlist:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(movie)
            with col2:
                if st.button("🗑️", key=f"remove_{movie}"):
                    remove_from_watchlist(movie)
                    st.rerun()
    else:
        st.sidebar.write("Your watchlist is empty")


# Base similarity recommendation function
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        cosine_angles = similarity[movie_index]
        recommended_movies = sorted(list(enumerate(cosine_angles)), reverse=True, key=lambda x: x[1])[0:7]

        final = []
        final_posters = []

        final_name, final_cast = crew(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        gen = genres(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        overview_final = overview(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        rel_date = date(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        ratings = rating(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        re4view = get_reviews(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        rev = rating(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
        trailer_final = trailer(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

        for i in recommended_movies:
            title = movies.iloc[i[0]].title
            if title not in st.session_state.skipped_movies:  # Skip movies user isn't interested in
                ratingg = rating(movies.iloc[i[0]].movie_id)
                score = round(i[1], 2)
                final.append(f"{title} (Rating: {ratingg}) - Similarity score: {score}")
                final_posters.append(poster(movies.iloc[i[0]].movie_id))

        return final_name, final_cast, rel_date, gen, overview_final, final, final_posters, ratings, re4view, rev, trailer_final

    except IndexError:
        return None



movies_dict = pickle.load(open('movies_dict.pkl' , 'rb' ))
movies = pd.DataFrame(movies_dict)
@st.cache_resource
def load_similarity():
    return pickle.load(open('similarity.pkl', 'rb'))

similarity = load_similarity()


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

# # Create a file to store the watchlist
# WATCHLIST_FILE = 'watchlist.txt'
# if not os.path.exists(WATCHLIST_FILE):
#     open(WATCHLIST_FILE, 'w').close()
#
# # Load the watchlist from the file
# with open(WATCHLIST_FILE, 'r') as f:
#     watchlist = f.read().splitlines()
#
# # Create a form to add a movie to the watchlist
# # Modify the form version to have a distinct key
# with st.form(key='manual_watchlist_form'):  # Changed the form key
#     movie_title = st.text_input(label='Add to Watchlist Manually', value=selected_movie)
#     add_movie = st.form_submit_button(label='Add to Watchlist manually')  # Changed the button label
#
# # If the add movie button is clicked and the movie title is not empty, add the movie to the watchlist
# if add_movie and movie_title:
#     if movie_title not in watchlist:
#         watchlist.append(movie_title)
#         with open(WATCHLIST_FILE, 'w') as f:
#             f.write('\n'.join(watchlist))
#         st.success(f'{movie_title} added to Watchlist!')
#     else:
#         st.warning(f'{movie_title} is already in the Watchlist!')
#
# # If the add movie button is clicked and the movie title is empty, show an error message
# if add_movie and not movie_title:
#     st.error('Please enter a movie title.')
#
# # Create a form to remove a movie from the watchlist
# with st.form(key='remove_movie_form'):
#     # Display the watchlist as a selectbox
#     selected_movie = st.selectbox('Select a movie to remove from Watchlist', watchlist)
#     remove_movie = st.form_submit_button(label='Remove Movie')
#
# # If the remove movie button is clicked, remove the selected movie from the watchlist
#
# if remove_movie and selected_movie:
#     watchlist.remove(selected_movie)
#     with open(WATCHLIST_FILE, 'w') as f:
#         f.write('\n'.join(watchlist))
#     st.success(f'{selected_movie} removed from Watchlist!')


# # Display the watchlist with movie details
# if watchlist:
#     movie_data = []
#     for movie in watchlist:
#         # Make a request to the TMDb API to get movie details
#         response = requests.get(
#             "https://api.themoviedb.org/3/search/movie?api_key=4158f8d4403c843543d3dc953f225d77&query={}".format(
#                 movie))
#         if response.status_code == 200:
#             results = response.json()['results']
#             if len(results) > 0:
#                 data = results[0]
#                 movie_data.append(data)
#     if movie_data:
#         df = pd.DataFrame(movie_data, columns=['title', 'overview'])
#         st.write(df)
#     else:
#         st.write('No movie details found.')
# else:
#     st.write('Watchlist is empty.')



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


def add_to_watchlist(movie_title, watchlist_file='watchlist.txt'):  # Changed from watchlist1.txt to watchlist.txt
    """Add a movie to the watchlist with proper feedback"""
    try:
        # Create file if it doesn't exist
        if not os.path.exists(watchlist_file):
            with open(watchlist_file, 'w') as f:
                f.write('')

        # Read existing watchlist
        with open(watchlist_file, 'r') as f:
            watchlist = [line.strip() for line in f.readlines()]

        # Add movie if not already in watchlist
        if movie_title not in watchlist:
            watchlist.append(movie_title)
            with open(watchlist_file, 'w') as f:
                f.write('\n'.join(watchlist))
            return True
        return False
    except Exception as e:
        print(f"Error adding to watchlist: {str(e)}")
        return False

class ComprehensiveMovieBot:
    def __init__(self, movies_df, similarity):
        self.movies_df = movies_df
        self.similarity = similarity
        self.recommendation_cache = {}
        self.movie_info_cache = {}


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
            'summer': {
                'keywords': ['summer', 'beach', 'vacation', 'summer break'],
                'genres': ['Action', 'Adventure', 'Comedy'],
                'mood': 'fun'
            }
        }

        self.genre_mappings = {
            'action': {
                'required_genres': ['Action'],
                'preferred_genres': ['Adventure', 'Thriller', 'Science Fiction'],
                'min_rating': 6.5
            },
            'comedy': {
                'required_genres': ['Comedy'],
                'preferred_genres': ['Romance', 'Family'],
                'min_rating': 6.5
            },
            'drama': {
                'required_genres': ['Drama'],
                'preferred_genres': ['Romance', 'History'],
                'min_rating': 7.0
            },
            'horror': {
                'required_genres': ['Horror'],
                'preferred_genres': ['Thriller', 'Mystery'],
                'min_rating': 6.5
            },
            'romance': {
                'required_genres': ['Romance'],
                'preferred_genres': ['Drama', 'Comedy'],
                'min_rating': 6.5
            },
            'family': {
                'required_genres': ['Family'],
                'preferred_genres': ['Animation', 'Adventure', 'Comedy'],
                'min_rating': 7.0
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
            'classic movies': {
                'year_range': (1900, 1980),
                'min_rating': 7.5
            }
        }

        # Mood mappings
        self.mood_mappings = {
            'happy': {
                'required_genres': ['Comedy', 'Animation', 'Family'],
                'preferred_genres': ['Adventure', 'Fantasy', 'Musical'],
                'exclude_genres': ['Horror', 'War', 'Thriller'],
                'min_rating': 7.0
            },
            'sad': {
                'required_genres': ['Drama'],
                'preferred_genres': ['Romance', 'War', 'History'],
                'exclude_genres': ['Comedy', 'Horror', 'Action'],
                'min_rating': 7.5
            },
            'excited': {
                'required_genres': ['Action', 'Adventure'],
                'preferred_genres': ['Science Fiction', 'Fantasy', 'Thriller'],
                'exclude_genres': ['Documentary', 'Drama'],
                'min_rating': 7.0
            },
            'relaxed': {
                'required_genres': ['Documentary', 'Family'],
                'preferred_genres': ['Comedy', 'Animation', 'Nature'],
                'exclude_genres': ['Horror', 'Thriller', 'War'],
                'min_rating': 7.0
            },
            'romantic': {
                'required_genres': ['Romance'],
                'preferred_genres': ['Drama', 'Comedy'],
                'exclude_genres': ['Horror', 'War', 'Thriller'],
                'min_rating': 7.0
            }
        }

    def get_movie_info(self, movie_id):
        """Get movie information with proper genre handling and error checking"""
        try:
            # Make a single API call to get most of the data
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US")
            if response.status_code != 200:
                return None

            data = response.json()

            # Safely handle genres - ensure it's always a list
            movie_genres = []
            if data.get('genres') and isinstance(data['genres'], list):
                movie_genres = data['genres']

            # Get crew info separately
            crew_names, crew_images = crew(movie_id)
            trailer_key = trailer(movie_id)

            movie_info = {
                'movie_id': movie_id,
                'genres': movie_genres,  # Now guaranteed to be a list
                'rating': float(data.get('vote_average', 0)),
                'poster': "https://image.tmdb.org/t/p/w500" + data['poster_path'] if data.get('poster_path') else "",
                'overview': data.get('overview', "No overview available"),
                'release_date': data.get('release_date', "Release date not available"),
                'trailer': trailer_key,
                'crew_names': crew_names,
                'crew_images': crew_images
            }

            # Cache the result
            self.movie_info_cache[movie_id] = movie_info
            return movie_info

        except Exception as e:
            print(f"Error getting movie info for {movie_id}: {str(e)}")
            return None

    def get_fresh_recommendations(self, category_type, category=None, input_movie=None):
        """Get fresh recommendations based on category type"""
        if category_type == 'similarity' and input_movie:
            return recommend(input_movie)
        elif category_type == 'mood':
            return self.get_recommendations(category.lower())
        elif category_type == 'seasonal':
            return self.get_seasonal_recommendations(category.lower())
        else:  # Quick categories
            return self.get_recommendations(category.lower())

    # Add a new function to refresh recommendations
    def refresh_recommendations(self, category_type, category=None):
        """Get fresh recommendations based on current offset."""
        try:
            # Increment offset
            current_offset = st.session_state.get('recommendation_offset', 0)
            new_offset = (current_offset + 5) % len(self.movies_df)
            st.session_state['recommendation_offset'] = new_offset

            if category_type == 'seasonal':
                return self.get_seasonal_recommendations(category)
            elif category_type == 'mood':
                return self.get_recommendations(category)
            else:
                return self.get_recommendations(category)

        except Exception as e:
            print(f"Error refreshing recommendations: {str(e)}")
            return None


    def get_recommendations_with_offset(self, category_type, category=None, offset=0):
        """Get recommendations with offset for refresh functionality"""
        if 'recommendation_offset' not in st.session_state:
            st.session_state.recommendation_offset = offset

        recommendations = self.get_recommendations(category)
        if recommendations and 'recommendations' in recommendations:
            # Rotate recommendations based on offset
            recs = recommendations['recommendations']
            rotated_recs = recs[offset:] + recs[:offset]
            recommendations['recommendations'] = rotated_recs

        return recommendations

    def detect_category(self, text):
        """Detect category from user input"""
        text = text.lower().strip()

        # Check for mood matches
        for mood in self.mood_mappings.keys():
            if mood in text:
                return {'type': 'mood', 'category': mood}

        # Check for seasonal matches
        for season, data in self.seasonal_mappings.items():
            if any(keyword in text for keyword in data['keywords']):
                return {'type': 'seasonal', 'category': season}

        # Check for special categories
        for category in self.special_categories.keys():
            if category in text:
                return {'type': 'special', 'category': category}

        # Check for general genres
        genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'family']
        for genre in genres:
            if genre in text:
                return {'type': 'genre', 'category': genre}

        return None



    def get_similar_movies(self, movie_title, n=5):
        """Get similar movies using similarity matrix"""
        try:
            # Get the index of the movie
            movie_index = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].index[0]

            # Get similarity scores
            similarity_scores = list(enumerate(self.similarity[movie_index]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            # Get the top N similar movies
            similar_movies = []
            for idx, score in similarity_scores[1:n + 1]:  # Skip first as it's the movie itself
                movie = self.movies_df.iloc[idx]
                movie_info = self.get_movie_info(movie.movie_id)
                if movie_info:
                    similar_movies.append({
                        'title': movie.title,
                        'info': movie_info,
                        'similarity_score': score
                    })

            return similar_movies

        except Exception as e:
            print(f"Error in get_similar_movies: {str(e)}")
            return []

    def get_seed_movie(self, category):
        """Get a good seed movie for the category"""
        category_seeds = {
            'valentine': ['Titanic', 'The Notebook', 'Pride and Prejudice'],
            'christmas': ['Home Alone', 'Elf', 'Miracle on 34th Street'],
            'halloween': ['The Shining', 'Halloween', 'A Nightmare on Elm Street'],
            'family': ['Toy Story', 'The Lion King', 'Finding Nemo'],
            'happy': ['The Hangover', 'Toy Story', 'Groundhog Day'],
            'sad': ['The Shawshank Redemption', 'Schindler\'s List', 'Life is Beautiful'],
            'classic': ['Casablanca', 'Gone with the Wind', 'The Godfather']
        }

        if category['type'] in ['mood', 'seasonal', 'special']:
            seeds = category_seeds.get(category['category'], [])
            for seed in seeds:
                if seed.lower() in self.movies_df['title'].str.lower().values:
                    return seed

        return None

    def matches_mood_criteria(self, movie_info, criteria):
        """Check if movie matches mood criteria"""
        try:
            if not movie_info or 'genres' not in movie_info:
                return False

            movie_genres = [g['name'] for g in movie_info['genres']]

            # Must have at least one required genre
            if not any(genre in movie_genres for genre in criteria['required_genres']):
                return False

            # Must not have any excluded genres
            if any(genre in movie_genres for genre in criteria.get('exclude_genres', [])):
                return False

            # Check rating
            rating = movie_info.get('rating')
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    return False

            if not rating or rating < criteria.get('min_rating', 7.0):
                return False

            return True

        except Exception as e:
            print(f"Error in matches_mood_criteria: {str(e)}")
            return False

    def matches_genre_criteria(self, movie_info, criteria):
        """Check if movie matches genre criteria"""
        try:
            if not movie_info or 'genres' not in movie_info:
                return False

            movie_genres = [g['name'] for g in movie_info['genres']]

            # Check if movie has required genre
            if not any(genre in movie_genres for genre in criteria['required_genres']):
                return False

            # Check rating
            rating = movie_info.get('rating')
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    return False

            if not rating or rating < criteria['min_rating']:
                return False

            return True

        except Exception as e:
            print(f"Error in matches_genre_criteria: {str(e)}")
            return False

    @st.cache_data
    def get_all_recommendations(self, category_type, category):
        """Get all possible recommendations for a category and cache them, with pre-filtering."""
        recommendations = []

        if category_type == 'mood':
            criteria = self.mood_mappings.get(category, {})
        else:
            criteria = self.genre_mappings.get(category, {})

        # Use pre-filtered DataFrame
        filtered_movies_df = self.filter_movies_by_criteria(
            required_genres=criteria.get('required_genres'),
            exclude_genres=criteria.get('exclude_genres'),
            min_rating=criteria.get('min_rating', 0)[0] if isinstance(criteria.get('min_rating', 0),
                                                                      list) else criteria.get('min_rating', 0)
        )

        # Now process only the filtered movies
        for _, movie in filtered_movies_df.iterrows():
            movie_info = self.get_movie_info(movie.movie_id)
            if movie_info and self.matches_genre_criteria(movie_info, criteria):
                recommendations.append({
                    'title': movie.title,
                    'info': movie_info
                })

        recommendations = sorted(
            recommendations,
            key=lambda x: float(x['info']['rating']) if isinstance(x['info']['rating'], (int, float)) else 0,
            reverse=True
        )

        return recommendations

    def get_seasonal_recommendations(self, season, page=0, page_size=5):
        """Get recommendations for seasonal movies with improved genre handling"""
        try:
            if season.lower() not in self.seasonal_mappings:
                return None

            # Get season criteria
            season_data = self.seasonal_mappings[season.lower()]
            required_genres = season_data['genres']
            exclude_genres = season_data.get('exclude_genres', [])
            recommendations = []

            # Get current offset from session state
            start_idx = st.session_state.get('recommendation_offset', 0)
            movies_processed = 0
            skipped_movies = st.session_state.get('skipped_movies', set())

            # Process movies starting from offset
            for _, movie in self.movies_df.iloc[start_idx:].iterrows():
                try:
                    # Skip if movie is in skipped list
                    if movie.title in skipped_movies:
                        continue

                    movie_info = self.get_movie_info(movie.movie_id)
                    if not movie_info:
                        continue

                    # Safely get movie genres
                    movie_genres = []
                    if movie_info.get('genres'):
                        movie_genres = [g.get('name', '') for g in movie_info['genres']]

                    # Check if movie matches criteria
                    if movie_genres and any(genre in required_genres for genre in movie_genres):
                        # Check excluded genres
                        if not any(genre in exclude_genres for genre in movie_genres):
                            recommendations.append({
                                'title': movie.title,
                                'info': movie_info
                            })

                    movies_processed += 1

                    # Break if we have enough recommendations
                    if len(recommendations) >= page_size:
                        break

                except Exception as e:
                    print(f"Error processing movie {movie.title}: {str(e)}")
                    continue

            # Update offset for next time
            new_offset = (start_idx + movies_processed) % len(self.movies_df)
            st.session_state['recommendation_offset'] = new_offset

            # If we don't have enough recommendations, reset offset and try again
            if len(recommendations) < page_size and movies_processed < len(self.movies_df):
                st.session_state['recommendation_offset'] = 0
                return self.get_seasonal_recommendations(season, page, page_size)

            return {
                'type': 'seasonal',
                'category': season.lower(),
                'recommendations': recommendations,
                'has_more': True,
                'total_pages': (len(self.movies_df) + page_size - 1) // page_size,
                'current_page': page
            }

        except Exception as e:
            print(f"Error in get_seasonal_recommendations: {str(e)}")
            return {
                'type': 'seasonal',
                'category': season.lower(),
                'recommendations': [],
                'has_more': False,
                'total_pages': 0,
                'current_page': 0
            }

    # Add this helper function to safely process genres
    def process_genres(self, genre_list):
        """Safely process genres from API response"""
        try:
            if not genre_list or not isinstance(genre_list, list):
                return []

            return [g.get('name', '') for g in genre_list if isinstance(g, dict) and 'name' in g]
        except Exception as e:
            print(f"Error processing genres: {str(e)}")
            return []

    def filter_movies_by_criteria(self, required_genres=None, exclude_genres=None, min_rating=0):
        """Filters movies based on genres and minimum rating."""
        filtered_df = self.movies_df

        if required_genres:
            filtered_df = filtered_df[filtered_df['genres'].apply(
                lambda genres: genres and any(genre in genres for genre in required_genres)
            )]

        if exclude_genres:
            filtered_df = filtered_df[~filtered_df['genres'].apply(
                lambda genres: genres and any(genre in genres for genre in exclude_genres)
            )]

        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]

        return filtered_df

    def get_recommendations(self, user_input):
        """Get movie recommendations based on user input"""
        try:
            # Get the current offset from session state
            offset = st.session_state.get('recommendation_offset', 0)
            user_input = user_input.lower().strip()
            print(f"Processing input: {user_input}")

            # Add a random offset to get different movies each time
            if 'recommendation_offset' not in st.session_state:
                st.session_state.recommendation_offset = 0

            # First check if it's a seasonal request
            seasonal_terms = {
                'christmas movies': 'christmas',
                'halloween movies': 'halloween',
                "valentine's day movies": 'valentine',
                'summer blockbusters': 'summer',
                'holiday movies': 'christmas',
                'family movie night': 'family',
                'date night movies': 'date'
            }

            # Check for seasonal match
            for term, season in seasonal_terms.items():
                if term in user_input:
                    result = self.get_seasonal_recommendations(season)
                    if result:
                        # Rotate recommendations based on offset
                        rotated_recs = result['recommendations'][st.session_state.recommendation_offset:] + \
                                       result['recommendations'][:st.session_state.recommendation_offset]
                        result['recommendations'] = rotated_recs
                    return result

            # Check if it's a mood
            mood = None
            for mood_key in self.mood_mappings.keys():
                if mood_key in user_input:
                    mood = mood_key
                    break

            # Check if it's a genre
            genre = None
            if not mood:  # Only check genre if no mood was found
                for key in self.genre_mappings.keys():
                    if key in user_input:
                        genre = key
                        break

            if not mood and not genre:
                # Check for special categories
                for category in self.special_categories.keys():
                    if category in user_input:
                        criteria = self.special_categories[category]
                        recommendations = []

                        # Use offset for movie selection
                        start_idx = st.session_state.recommendation_offset
                        for _, movie in self.movies_df.iloc[start_idx:].iterrows():
                            try:
                                movie_info = self.get_movie_info(movie.movie_id)
                                if not movie_info:
                                    continue

                                if self.check_special_criteria(movie_info, category):
                                    recommendations.append({
                                        'title': movie.title,
                                        'info': movie_info
                                    })

                                if len(recommendations) >= 10:
                                    break

                            except Exception as e:
                                print(f"Error processing movie {movie.title}: {str(e)}")
                                continue

                        recommendations = sorted(
                            recommendations,
                            key=lambda x: float(x['info']['rating']) if isinstance(x['info']['rating'],
                                                                                   (int, float)) else 0,
                            reverse=True
                        )[:5]

                        return {
                            'type': 'special',
                            'category': category,
                            'recommendations': recommendations
                        }

                return {'type': 'unknown', 'category': None, 'recommendations': []}

            # Get recommendations based on type
            recommendations = []

            if mood:
                print(f"Found mood: {mood}")
                criteria = self.mood_mappings[mood]
                matcher = self.matches_mood_criteria
                rec_type = 'mood'
                category = mood
            else:
                print(f"Found genre: {genre}")
                criteria = self.genre_mappings[genre]
                matcher = self.matches_genre_criteria
                rec_type = 'genre'
                category = genre

            # Process movies with offset
            start_idx = st.session_state.recommendation_offset
            for _, movie in self.movies_df.iloc[start_idx:].iterrows():
                try:
                    movie_info = self.get_movie_info(movie.movie_id)
                    if not movie_info:
                        continue

                    if matcher(movie_info, criteria):
                        recommendations.append({
                            'title': movie.title,
                            'info': movie_info
                        })

                    if len(recommendations) >= 10:
                        break

                except Exception as e:
                    print(f"Error processing movie {movie.title}: {str(e)}")
                    continue

            # If we don't have enough recommendations, reset offset and try again from start
            if len(recommendations) < 5:
                st.session_state.recommendation_offset = 0
                return self.get_recommendations(user_input)

            # # Sort by rating and take top 5
            # recommendations = sorted(
            #     recommendations,
            #     key=lambda x: float(x['info']['rating']) if isinstance(x['info']['rating'], (int, float)) else 0,
            #     reverse=True
            # )[:5]

            # Increment offset for next refresh
            st.session_state.recommendation_offset += 5
            if st.session_state.recommendation_offset >= len(self.movies_df):
                st.session_state.recommendation_offset = 0

            return {
                'type': rec_type,
                'category': category,
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return {'type': 'error', 'category': None, 'recommendations': []}

    def check_recommendation_criteria(self, movie_info, category):
        """Check if movie matches category criteria"""
        try:
            if not movie_info or 'genres' not in movie_info:
                return False

            if category['type'] == 'seasonal':
                season_data = self.seasonal_mappings.get(category['category'])
                if not season_data:
                    return False

                movie_genres = [g['name'] for g in movie_info['genres']]
                rating = movie_info.get('rating')

                # Convert rating to float if it's a string
                if isinstance(rating, str):
                    try:
                        rating = float(rating)
                    except ValueError:
                        return False

                # Check genre requirements
                has_required_genre = any(genre in season_data['genres'] for genre in movie_genres)
                no_excluded_genre = not any(genre in season_data.get('exclude_genres', [])
                                            for genre in movie_genres)

                # Check rating requirement (minimum 7.0 for seasonal movies)
                good_rating = rating and rating >= 7.0

                return has_required_genre and no_excluded_genre and good_rating

            elif category['type'] == 'special':
                return self.check_special_criteria(movie_info, category['category'])

            elif category['type'] == 'mood':
                return self.check_mood_criteria(movie_info, category['category'])

            return False

        except Exception as e:
            print(f"Error in check_recommendation_criteria: {str(e)}")
            return False

    def check_mood_criteria(self, movie_info, mood):
        """Check if movie matches mood criteria"""
        try:
            movie_genres = [g['name'] for g in movie_info['genres']]
            preferred_genres = self.mood_mappings.get(mood, [])

            rating = movie_info.get('rating')
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    return False

            return (
                    any(genre in preferred_genres for genre in movie_genres) and
                    rating and rating >= 7.0
            )
        except Exception as e:
            print(f"Error in check_mood_criteria: {str(e)}")
            return False

    def check_seasonal_criteria(self, movie_info, season):
        """Check if movie matches seasonal criteria"""
        try:
            season_data = self.seasonal_mappings.get(season)
            if not season_data or not movie_info or 'genres' not in movie_info:
                return False

            movie_genres = [g['name'] for g in movie_info['genres']]
            return (
                    any(genre in season_data['genres'] for genre in movie_genres) and
                    not any(genre in season_data.get('exclude_genres', []) for genre in movie_genres)
            )
        except Exception as e:
            print(f"Error in check_seasonal_criteria: {str(e)}")
            return False

    def check_special_criteria(self, movie_info, category):
        """Check if movie matches special criteria"""
        try:
            special_data = self.special_categories.get(category)
            if not special_data:
                return False

            # Check rating requirement
            rating = movie_info.get('rating')
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    return False

            if 'min_rating' in special_data and (not rating or rating < special_data['min_rating']):
                return False

            # Check year range for classics
            if 'year_range' in special_data:
                release_date = movie_info.get('release_date', '')
                try:
                    year = int(release_date[:4])
                    if not (special_data['year_range'][0] <= year <= special_data['year_range'][1]):
                        return False
                except (ValueError, IndexError):
                    return False

            # Check genre requirements
            if 'genres' in special_data:
                movie_genres = [g['name'] for g in movie_info['genres']]
                if not any(genre in special_data['genres'] for genre in movie_genres):
                    return False

            # Check excluded genres
            if 'exclude_genres' in special_data:
                movie_genres = [g['name'] for g in movie_info['genres']]
                if any(genre in special_data['exclude_genres'] for genre in movie_genres):
                    return False

            return True

        except Exception as e:
            print(f"Error in check_special_criteria: {str(e)}")
            return False

    def check_genre_criteria(self, movie_info, genre):
        """Check if movie matches genre criteria"""
        try:
            if not movie_info or 'genres' not in movie_info:
                return False

            movie_genres = [g['name'].lower() for g in movie_info['genres']]
            return genre in movie_genres
        except Exception as e:
            print(f"Error in check_genre_criteria: {str(e)}")
            return False

# Load data and initialize bot
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Define options
quick_categories = [
    "Action Movies", "Comedy Movies", "Drama Movies",
    "Horror Movies", "Romance Movies", "Family Movies"
]

season_options = [
    "Christmas Movies",
    "Halloween Movies",
    "Valentine's Day Movies",
    "Summer Blockbusters",
    "Family Movie Night",
    "Date Night Movies"
]

mood_options = [
    "Happy", "Sad", "Excited", "Relaxed",
    "Romantic", "Thoughtful"
]


def display_movie_details(movie_data):
    """Display main movie details"""
    name, cast, rel_date, gen, overview_final, ans, posters, ratings, re4view, rev, trailer_final = movie_data[:11]

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
        genres = process(gen)
        genres_str = " , ".join(genres)
        st.write("Genres : {}".format(genres_str))
        st.write("Release Date : {} ".format(rel_date))
        st.write("Ratings : {} ".format(ratings))

    return name, cast, ans, posters, re4view


def display_recommendations(recommendations, category_type=None):
    """Display recommendations with working Streamlit refresh functionality"""
    if not recommendations:
        st.warning("No recommendations found!")
        return

    # Create unique key for refresh button using current time
    refresh_key = f"refresh_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Refresh button centered
    cols = st.columns([1, 2, 1])
    with cols[1]:
        if st.button("🔄 Get New Recommendations", key=refresh_key):
            # Update offset in session state
            if 'recommendation_offset' not in st.session_state:
                st.session_state.recommendation_offset = 0

            # Increment offset
            current_offset = st.session_state.recommendation_offset
            st.session_state.recommendation_offset = (current_offset + 5) % len(movies)

            # Get fresh recommendations based on category
            if category_type == 'mood':
                st.session_state.last_recommendations = bot.get_recommendations(st.session_state.selected_mood.lower())
            elif category_type == 'seasonal':
                season_category = st.session_state.selected_seasonal_category.split()[0].lower()
                st.session_state.last_recommendations = bot.get_seasonal_recommendations(season_category)
            else:  # Quick recommendations
                search_term = (st.session_state.quick_input
                               if st.session_state.quick_input
                               else st.session_state.selected_quick_category)
                st.session_state.last_recommendations = bot.get_recommendations(search_term.lower())

            # Force streamlit to rerun
            st.rerun()

    # Display recommendations in grid
    for i in range(0, len(recommendations['recommendations']), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(recommendations['recommendations']):
                movie = recommendations['recommendations'][idx]

                # Skip if movie is in skipped list
                if movie['title'] in st.session_state.get('skipped_movies', set()):
                    continue

                with col:
                    st.subheader(movie['title'])
                    if movie['info'].get('poster'):
                        st.image(movie['info']['poster'], width=200)

                    with st.expander("More Info"):
                        st.write(f"Rating: {movie['info'].get('rating', 'N/A')}")
                        st.write(movie['info'].get('overview', 'No overview available'))

                    # Buttons in columns
                    button_cols = st.columns(2)
                    with button_cols[0]:
                        not_interested_key = f"not_interested_{refresh_key}_{idx}"
                        if st.button("Not Interested", key=not_interested_key):
                            if 'skipped_movies' not in st.session_state:
                                st.session_state.skipped_movies = set()
                            st.session_state.skipped_movies.add(movie['title'])
                            st.rerun()

                    with button_cols[1]:
                        watchlist_key = f"watchlist_{refresh_key}_{idx}"
                        if st.button("Add to Watchlist", key=watchlist_key):
                            if add_to_watchlist(movie['title']):
                                st.success(f"Added {movie['title']} to watchlist!")
                            else:
                                st.info("Already in watchlist!")


def display_cast(cast, names):
    """Display cast members"""
    st.title("Top Cast")
    if len(cast) >= 6 and len(names) >= 6:
        # First row
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                st.image(cast[i], width=225, use_column_width=225)
                st.caption(names[i])

        # Second row
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                st.image(cast[i + 3], width=225, use_column_width=225)
                st.caption(names[i + 3])
    else:
        st.warning("Not enough cast members to display.")


def display_reviews(re4view):
    """Display reviews and sentiment analysis"""
    if re4view:
        # Calculate sentiment counts
        pos_count = sum(1 for sentiment in re4view.values() if sentiment == 'Positive')
        neg_count = sum(1 for sentiment in re4view.values() if sentiment == 'Negative')

        # Plot sentiment analysis
        fig, ax = plt.subplots(dpi=50)
        ax.bar(['Positive', 'Negative'], [pos_count, neg_count])
        ax.set_title('Sentiment Analysis of Reviews')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Reviews')
        st.pyplot(fig)

        # Display reviews table
        st.write("Reviews:")
        df = pd.DataFrame.from_dict(re4view, orient='index', columns=['Sentiment'])
        styled_table = df.style.set_table_styles([
            {'selector': 'tr', 'props': [('background-color', 'white')]},
            {'selector': 'th', 'props': [('background-color', 'lightgrey')]},
            {'selector': 'td', 'props': [('color', 'black')]}
        ])
        styled_table.highlight_max(axis=0)
        st.table(styled_table)
    else:
        st.write("No reviews found for this movie.")

@st.cache_resource
def get_movie_bot():
    return ComprehensiveMovieBot(movies, similarity)

# Use the cached bot
bot = get_movie_bot()

# Main UI code
st.title('Movie Recommendation System')


# Show watchlist in sidebar
display_watchlist()

# Main content
tab1, tab2, tab3 = st.tabs(["Quick Recommendations", "Seasonal & Special", "Mood Based"])

with tab1:
    if 'quick_input' not in st.session_state:
        st.session_state.quick_input = ""

    st.session_state.quick_input = st.text_input("What kind of movies would you like?",
                                                 value=st.session_state.quick_input)
    st.session_state.selected_quick_category = st.selectbox("Or choose from common categories:", quick_categories)

    if st.button("Get Quick Recommendations", key="quick_rec_button"):
        st.session_state.current_tab = 'Quick'
        st.session_state.recommendation_offset = 0
        st.session_state.current_page = 0
        search_term = st.session_state.quick_input if st.session_state.quick_input else st.session_state.selected_quick_category
        st.session_state.last_recommendation = bot.get_recommendations(search_term.lower())

    if st.session_state.last_recommendation and st.session_state.current_tab == 'Quick':
        display_recommendations(st.session_state.last_recommendation, 'quick')

with tab2:
    st.session_state.selected_seasonal_category = st.selectbox("Select a category:", season_options)

    if st.button("Get Seasonal Recommendations", key="seasonal_button"):
        st.session_state['current_tab'] = 'Seasonal'
        season_category = st.session_state.selected_seasonal_category.split()[0].lower()
        recommendations = bot.get_seasonal_recommendations(season_category)

    if st.session_state.last_recommendation and st.session_state.current_tab == 'Seasonal':
        display_recommendations(st.session_state.last_recommendation, 'seasonal')

with tab3:
    st.session_state.selected_mood = st.selectbox("Select your mood:", mood_options)

    if st.button("Get Mood Recommendations", key="mood_button"):
        st.session_state.current_tab = 'Mood'
        st.session_state.recommendation_offset = 0
        st.session_state.current_page = 0
        st.session_state.last_recommendation = bot.get_recommendations(st.session_state.selected_mood.lower())

    if st.session_state.last_recommendation and st.session_state.current_tab == 'Mood':
        display_recommendations(st.session_state.last_recommendation, 'mood')


# Footer
st.markdown("---")
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Clear All", key="clear_all", use_container_width=True):
        # Reset all session state
        for key in ['current_page', 'last_quick_category', 'last_seasonal_category',
                    'last_mood', 'last_search', 'recommendation_offset', 'skipped_movies']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()