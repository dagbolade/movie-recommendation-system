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

        st.write("Reviews : {} ".format(re4view))
        st.write("Review : {} ".format(rev))


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

    st.title(" Top Reviews")

    st.image("https://image.flaticon.com/icons/png/512/25/25231.png", width=225, use_column_width=225)
    for i in range(len(rev)):
        st.write("Review {}: {}".format(i+1, rev[i]))


        # Create a dataframe from the reviews dictionary
    df = pd.DataFrame.from_dict((re4view), orient='index', columns=['Sentiment'])

    # Display the dataframe in Streamlit
    st.dataframe(df)





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






