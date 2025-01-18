import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model



# define paths to the data files
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_PATH=os.path.join(BASE_DIR,'recommender/static/data/movies.csv')
RATINGS_PATH=os.path.join(BASE_DIR,'recommender/static/data/ratings.csv')



# Load datasets
movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)



# create mappings for user ids and movie ids to indices for embedding layers
user_map={id:idx for idx,id in enumerate(ratings['userId'].unique())}
movie_map={id:idx for idx,id in enumerate(ratings['movieId'].unique())}
reverse_movie_map={v:k for k,v in movie_map.items()} # reverse mapping to get original movie ids



# replace user ids and movie ids in the ratings dataframe with their mapped indices
ratings['userId']=ratings['userId'].map(user_map)
ratings['movieId']=ratings['movieId'].map(movie_map)



# split the data into training and test sets
train,test=train_test_split(ratings,test_size=0.2,random_state=42)



# define the ncf model
def build_ncf_model(num_users,num_movies,embedding_dim=50):

    # create user embedding layer
    user_input=Input(shape=(1,),name='user_input')
    user_embedding=Embedding(num_users,embedding_dim,name='user_embedding')(user_input)
    user_vec=Flatten(name='flatten_user')(user_embedding)

    # create movie embedding layer
    movie_input=Input(shape=(1,),name='movie_input')
    movie_embedding=Embedding(num_movies,embedding_dim,name='movie_embedding')(movie_input)
    movie_vec=Flatten(name='flatten_movie')(movie_embedding)

    # concatenate user and movie vectors and pass through dense layers
    concat=Concatenate()([user_vec,movie_vec])
    dense=Dense(128,activation='relu')(concat)
    dense=Dropout(0.3)(dense)
    dense=Dense(64,activation='relu')(dense)
    output=Dense(1,activation='sigmoid')(dense)  # prediction, changed to sigmoid

    # scale the output to match the rating range (0.5 to 5.0)
    scaled_output=tf.keras.layers.Lambda(lambda x:x*4.5+0.5)(output) # ?

    # compile the model
    model=Model(inputs=[user_input,movie_input],outputs=scaled_output)
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    return model



# build and train the ncf model
num_users=len(user_map)
num_movies=len(movie_map)
embedding_dim=50



ncf_model=build_ncf_model(num_users,num_movies,embedding_dim)



# prepare input data for training
train_user=train['userId'].values
train_movie=train['movieId'].values
train_rating=train['rating'].values
test_user=test['userId'].values
test_movie=test['movieId'].values
test_rating=test['rating'].values



# train the model
ncf_model.fit(
    [train_user,train_movie],
    train_rating,
    validation_data=([test_user,test_movie], test_rating),
    epochs=10,
    batch_size=64
)



# save the trained model
ncf_model.save('ncf_model.h5') # this throws warning



# function to get movie recommendations for a user
def get_recommendations(user_ratings):
    new_user_id=1 # arbitrary for us 

    # map user ratings to internal indices
    rated_movie_ids={movie_map[movie_id]:rating for movie_id,rating in user_ratings.items() if movie_id in movie_map}

    # get all movie indices except those already rated
    all_movie_ids=set(movie_map.values())
    unrated_movie_ids=list(all_movie_ids-set(rated_movie_ids.keys()))

    # predict scores for unrated movies
    user_input=np.array([new_user_id]*len(unrated_movie_ids))
    movie_input=np.array(unrated_movie_ids)
    predictions=ncf_model.predict([user_input,movie_input])

    # combine predictions with movie ids and sort them
    predicted_scores=list(zip(unrated_movie_ids,predictions.flatten()))
    predicted_scores.sort(key=lambda x:x[1],reverse=True)
    top_3=predicted_scores[:3]

    # fetch movie titles and scores for the top 3 recommendations
    recommendations=[]
    for movie_id,score in top_3:
        if movie_id in reverse_movie_map:
            movie_title=movies[movies['movieId']==reverse_movie_map[movie_id]]
            if not movie_title.empty:
                recommendations.append({
                    "title":movie_title.iloc[0]['title'],
                    "movieId":reverse_movie_map[movie_id],
                    "score":round(score,2)
                })

    return recommendations
