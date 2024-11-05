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

# Movie Chatbot Section
st.title("Movie Recommendation Chatbot")


class SimpleMovieBot:
    def __init__(self, movies_df, similarity):
        self.movies_df = movies_df
        self.similarity = similarity
        self.moods = {
            'happy': ['comedy', 'animation', 'family'],
            'sad': ['drama', 'romance'],
            'excited': ['action', 'adventure', 'sci-fi'],
            'relaxed': ['documentary', 'family'],
            'scared': ['horror', 'thriller']
        }

    def get_recommendations(self, input_text):
        input_text = input_text.lower()

        # Check for mood-based recommendations
        for mood, genres in self.moods.items():
            if mood in input_text:
                recommendations = []
                for _, movie in self.movies_df.iterrows():
                    movie_genres = genres(movie.movie_id)
                    if any(g['name'].lower() in genres for g in movie_genres):
                        movie_rating = rating(movie.movie_id)
                        if movie_rating > 7.0:  # Only recommend highly-rated movies
                            recommendations.append({
                                'title': movie.title,
                                'rating': movie_rating,
                                'poster': poster(movie.movie_id)
                            })
                            if len(recommendations) == 5:  # Limit to 5 recommendations
                                break
                return {
                    'type': 'mood',
                    'mood': mood,
                    'recommendations': recommendations
                }

        # Check for movie similarity recommendations
        if "like" in input_text:
            try:
                # Extract movie name after "like"
                movie_name = input_text.split("like")[1].strip()
                if movie_name in self.movies_df['title'].values:
                    idx = self.movies_df[self.movies_df['title'] == movie_name].index[0]
                    sim_scores = list(enumerate(self.similarity[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                    recommendations = []
                    for i, score in sim_scores:
                        recommendations.append({
                            'title': self.movies_df.iloc[i].title,
                            'rating': rating(self.movies_df.iloc[i].movie_id),
                            'poster': poster(self.movies_df.iloc[i].movie_id)
                        })
                    return {
                        'type': 'similar',
                        'movie': movie_name,
                        'recommendations': recommendations
                    }
            except:
                pass

        return None


# Initialize bot
bot = SimpleMovieBot(movies, similarity)

# Chat interface
st.markdown("""
### How to use:
1. Tell me your mood (e.g., "I'm feeling happy")
2. Or tell me a movie you like (e.g., "I like The Dark Knight")
""")

# Create chat input
user_input = st.text_input("What kind of movies would you like?", key="chat_input")
submit_button = st.button("Get Recommendations")

if submit_button and user_input:
    recommendations = bot.get_recommendations(user_input)

    if recommendations:
        if recommendations['type'] == 'mood':
            st.write(f"Based on your {recommendations['mood']} mood, here are some recommendations:")
        else:
            st.write(f"If you like {recommendations['movie']}, you might enjoy:")

        # Display recommendations in columns
        cols = st.columns(len(recommendations['recommendations']))
        for col, movie in zip(cols, recommendations['recommendations']):
            with col:
                if movie['poster']:
                    st.image(movie['poster'], width=150)
                st.write(f"**{movie['title']}**")
                st.write(f"Rating: {movie['rating']}/10")
    else:
        st.write("I couldn't understand your request. Try:\n"
                 "- 'I'm feeling [happy/sad/excited/relaxed/scared]'\n"
                 "- 'I like [movie name]'")

# Add to watchlist option
if st.button("Add recommended movies to watchlist"):
    if 'recommendations' in locals() and recommendations and recommendations.get('recommendations'):
        for movie in recommendations['recommendations']:
            if movie['title'] not in watchlist:
                watchlist.append(movie['title'])
        with open(WATCHLIST_FILE, 'w') as f:
            f.write('\n'.join(watchlist))
        st.success("Added recommended movies to your watchlist!")