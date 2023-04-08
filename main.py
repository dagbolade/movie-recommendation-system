import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
from bs4 import BeautifulSoup
import urllib.request
import pickle
import requests
from datetime import date, datetime
import matplotlib.pyplot as plt





def crew(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{0}/credits?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    crew_name = []
    final_cast = []
    k = 0
    for i in data["cast"]:
        if(k!=6):
            crew_name.append(i['name'])
            final_cast.append("https://image.tmdb.org/t/p/w500/" + i['profile_path'])
            k+=1
        else:
            break
    return crew_name , final_cast



def date(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    return data['release_date']


def genres(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    return data['genres']

def overview(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(
            movie_id))
    data = response.json()
    return data['overview']
def poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def rating(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(movie_id))
    data = response.json()
    return data['vote_average']


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

def get_reviews(movie_id):
    url = 'https://api.themoviedb.org/3/movie/{}/reviews?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US&page=1'.format(movie_id)
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json().get('results')
        if not results:
            print('No reviews found for this movie.')
            return None
        reviews_list = []
        reviews_status = []
        for review in results:
            review_text = review.get('content')
            if review_text:
                reviews_list.append(review_text)
                # passing the review to our model
                movie_review_list = np.array([review_text])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Positive' if pred else 'Negative')
        # combining reviews and comments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}








        for review, status in movie_reviews.items():
            print('{} - {}'.format(status, review))
        return movie_reviews
    except requests.exceptions.HTTPError as e:
        print('Error retrieving reviews: {}'.format(e))
        return None



def trailer(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}/videos?api_key=4158f8d4403c843543d3dc953f225d77&language=en-US".format(movie_id))
    data = response.json()
    return data['results'][0]['key']



def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    cosine_angles = similarity[movie_index]
    recommended_movies = sorted(list(enumerate(cosine_angles)), reverse=True, key=lambda x: x[1])[0:7]


    final = []
    final_posters = []
    final_name , final_cast = crew(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
    gen = genres(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
    overview_final = overview(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
    rel_date = date(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
    ratings = rating(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

    re4view = get_reviews(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
    rev = review(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)
    trailer_final = trailer(movies.iloc[movies[movies['title'] == movie].index[0]].movie_id)

    for i in recommended_movies:

        final.append(movies.iloc[i[0]].title)
        final_posters.append(poster(movies.iloc[i[0]].movie_id))

    return final_name , final_cast , rel_date , gen , overview_final , final , final_posters, ratings, re4view, rev, trailer_final




movies_dict = pickle.load(open('movies_dict.pkl' , 'rb' ))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl' , 'rb'))
st.title('Movie Recommendation System')

filename = 'finalized_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))


selected_movie = st.selectbox(
    'Which Movie Do you like?',
     movies['title'].values)



def process(genre):
    final = []
    for i in genre:
        final.append(i['name'])

    return final

if st.button('Search'):

    name , cast , rel_date , gen , overview_final , ans , posters, ratings, re4view, rev, trailer_final = recommend(selected_movie)




    st.header(selected_movie)
    col_1 , col_2 = st.columns(2)


    with col_1:
        st.image(posters[0] , width=  325 , use_column_width= 325)

    with col_2:
        st.write("Title : {} ".format(ans[0]))

        st.write("Overview : {} ".format(overview_final))
        gen = process(gen)
        gen = " , ".join(gen)
        st.write("Genres : {}".format(gen))
        st.write("Release Date {} : {} ".format(" " , rel_date))
        st.write("Ratings : {} ".format(ratings))









    st.title("Top Casts")

    c1 , c2 , c3 = st.columns(3)
    with c1:
        st.image(cast[0] , width=  225 , use_column_width= 225)
        st.caption(name[0])
    with c2:
        st.image(cast[1] , width=  225 , use_column_width= 225)
        st.caption(name[1])
    with c3:
        st.image(cast[2], width=  225 , use_column_width= 225)
        st.caption(name[2])


    c1 , c2 ,c3 = st.columns(3)
    with c1:
        st.image(cast[3], width=  225 , use_column_width= 225)
        st.caption(name[3])

    with c2:
        st.image(cast[4], width=  225 , use_column_width= 225)
        st.caption(name[4])

    with c3:
        st.image(cast[5], width=225, use_column_width=225)
        st.caption(name[5])



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
        fig, ax = plt.subplots()
        ax.bar(['Positive', 'Negative'], [pos_count, neg_count])
        ax.set_title('Sentiment Analysis of Reviews')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Reviews')
        st.pyplot(fig)

        # Create a dataframe from the reviews dictionary
        df = pd.DataFrame.from_dict((re4view), orient='index', columns=['Sentiment'])

        # Display the dataframe in Streamlit
        st.write("Reviews:")
        st.table(df.style.highlight_max(axis=0))
    else:
        st.write("No reviews found for this movie.")







    st.title("")


    st.title("   Similar Movies You May Like")

    c1, c2, c3 = st.columns(3)
    with c1:

         st.image(posters[1], width=225, use_column_width=225)
         st.write(ans[1])




    with c2:
        st.image( posters[2], width=225, use_column_width=225)
        st.write(ans[2])
    with c3:
        st.image(posters[3], width=225, use_column_width=225)
        st.write(ans[3])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(posters[4], width=225, use_column_width=225)
        st.write(ans[4])

    with c2:
        st.image(posters[5], width=225, use_column_width=225)
        st.write(ans[5])

    with c3:
        st.image(posters[6], width=225, use_column_width=225)
        st.write(ans[6])


    import streamlit as st

    # Add a social sharing menu
    def add_share_menu():
        url = selected_movie # Replace with your app URL
        twitter_text = 'Check out this cool movie' # Replace with your Twitter message
        whatsapp_text = 'Check out this cool movie: {}'.format(url) # Replace with your WhatsApp message

        st.sidebar.subheader('Share')
        st.sidebar.write('Share this app with your friends and colleagues:')
        tweet_btn = st.sidebar.button(label='Twitter')
        if tweet_btn:
            tweet_url = 'https://twitter.com/intent/tweet?text={}&url={}'.format(twitter_text, url)
            st.sidebar.markdown('[![Tweet](https://img.shields.io/twitter/url?style=social&url={})]({})'.format(tweet_url, tweet_url), unsafe_allow_html=True)

        whatsapp_btn = st.sidebar.button(label='WhatsApp')
        if whatsapp_btn:
            whatsapp_url = 'https://wa.me/?text={}'.format(whatsapp_text)
            st.sidebar.markdown('[![WhatsApp](https://img.shields.io/badge/WhatsApp-Chat-green?style=social&logo=whatsapp)]({})'.format(whatsapp_url), unsafe_allow_html=True)

    # Add the social sharing menu to your app
    add_share_menu()

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
    st.success(f'{selected_movie} removed from Watchlist!')



# Display the watchlist with movie details
if watchlist:
    movie_data = []
    for movie in watchlist:
        # Make a request to the OMDb API to get movie details
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie?api_key=4158f8d4403c843543d3dc953f225d77&query={}".format(
                movie))
        if response.status_code == 200:
            data = response.json()['results'][0]
            movie_data.append(data)
    if movie_data:
        df = pd.DataFrame(movie_data, columns=['title', 'overview'])
        st.write(df)
    else:
        st.write('No movie details found.')
else:
    st.write('Watchlist is empty.')






