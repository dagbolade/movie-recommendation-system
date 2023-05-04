# Sentiment Based Movie-recommendation-system
The project is a movie recommendation engine that uses machine learning techniques to provide personalized movie recommendations to users. The engine is integrated into a web application built using Streamlit, a Python library for creating data apps. In addition to recommending movies, the application also provides a search function for users to search for movies by title, genre, or cast. The application uses a dataset of movie information and user ratings to generate recommendations for each user. The machine learning model is trained on this dataset to predict which movies a user is likely to enjoy based on their viewing history and preferences. Overall, the project demonstrates the power of machine learning in developing personalized recommendations for users and showcases the efficiency of Streamlit in deploying such applications.


## How to Run the Movie Recommendation System
1.Clone the project repository from Github to your local machine using the following command:
```
https://github.com/dagbolade/Movie-recommendation-system.git
```
2.Navigate to the project directory on your local machine using the following command:
```
cd Movie-recommendation-system
```
3.Install the required packages using the following command:
```
pip install -r requirements.txt
```
4.Run the following command to launch the web application:
```
streamlit run main.py
```
5.The web application should now be running on a local server. Navigate to the URL provided in the terminal to view the application in your browser.

## How to Use the Movie Recommendation System
1.Once the web application is running, you should see the following home page:
![image](https://user-images.githubusercontent.com/71558720/126080401-9b6b9b9d-2b9a-4b0a-8b0a-9b0b6b2b2b0b.png)

2.To get movie recommendations, enter your movie in the text box and click the 'Search' button. 

3.The application will then display the movie poster, title, genre, release date, and a brief description of the movie and also the ratings and the sentiment reviews of the movie.

4.It will the display the top 6 casts, the trailer of the movie and the top 6 recommended movies based on the movie you searched for.



