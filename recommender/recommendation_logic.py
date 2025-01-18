import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

# custom scaling layer
class RatingScale(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs*4.5+0.5



# global settings
MODEL_PATH="ncf_model.keras"
EMBED_DIM=32
BATCH_SIZE=64
EPOCHS=10
TOP_K_CANDIDATES=30



# load data
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH=os.path.join(BASE_DIR, "static/data/movies.csv")
RATINGS_PATH=os.path.join(BASE_DIR, "static/data/ratings.csv")



movies=pd.read_csv(MOVIES_PATH)
ratings=pd.read_csv(RATINGS_PATH)



unique_user_ids=ratings["userId"].unique()
unique_movie_ids=ratings["movieId"].unique()



user_map={uid: idx for idx, uid in enumerate(unique_user_ids)}
movie_map={mid: idx for idx, mid in enumerate(unique_movie_ids)}
reverse_movie_map={v: k for k, v in movie_map.items()}



ratings["userId"]=ratings["userId"].map(user_map)
ratings["movieId"]=ratings["movieId"].map(movie_map)



num_users=len(user_map)
num_movies=len(movie_map)



# build a fresh model every time
def build_ncf_model(num_users, num_movies, embed_dim=32):
    """
    build a neural collaborative filtering model
    - create input layers for users and movies
    - add embedding layers for users and movies
    - flatten the embeddings
    - concatenate the user and movie vectors
    - add dense and dropout layers
    - output a single rating prediction
    - compile the model with adam optimizer and mse loss
    """
    user_input=Input(shape=(1,), name="user_input")
    movie_input=Input(shape=(1,), name="movie_input")

    user_emb=Embedding(input_dim=num_users, output_dim=embed_dim, embeddings_regularizer=l2(1e-6), name="user_embedding")(user_input)
    movie_emb=Embedding(input_dim=num_movies, output_dim=embed_dim, embeddings_regularizer=l2(1e-6), name="movie_embedding")(movie_input)

    user_vec=Flatten(name="flatten_user")(user_emb)
    movie_vec=Flatten(name="flatten_movie")(movie_emb)
    concat_vec=Concatenate(name="concat_user_movie")([user_vec, movie_vec])

    x=Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(concat_vec)
    x=Dropout(0.4)(x)
    x=Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x=Dropout(0.4)(x)
    x=Dense(1, activation='sigmoid')(x)
    out=RatingScale(name="rating_scale")(x)

    model=Model(inputs=[user_input, movie_input], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model



print("building a fresh model and training it on startup...")



ncf_model=build_ncf_model(num_users, num_movies, EMBED_DIM)

# train on startup
train_df, test_df=train_test_split(ratings, test_size=0.2, random_state=42)

train_users=train_df["userId"].values
train_movies=train_df["movieId"].values
train_ratings=train_df["rating"].values

test_users=test_df["userId"].values
test_movies=test_df["movieId"].values
test_ratings=test_df["rating"].values

early_stop=EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
ncf_model.fit([train_users, train_movies], train_ratings, validation_data=([test_users, test_movies], test_ratings), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=1)

ncf_model.save(MODEL_PATH)
print(f"model was trained and saved to: {MODEL_PATH}")



# build a sub-model for new users
def build_uservec_submodel(orig_model, embed_dim=32):
    """
    build a sub-model for user vector inference
    - create input layers for user vector and movie id
    - get the movie embedding layer from the original model
    - flatten the movie embedding
    - concatenate the user vector and movie vector
    - add dense and dropout layers
    - output a single rating prediction
    - set weights from the original model to the new model
    """
    from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Concatenate

    user_vec_input=Input(shape=(embed_dim,), name="user_vec")
    movie_id_input=Input(shape=(1,), name="movie_id_sub")

    movie_emb_layer=orig_model.get_layer("movie_embedding")
    movie_emb_sub=movie_emb_layer(movie_id_input)
    movie_vec_sub=Flatten()(movie_emb_sub)

    merged_vec=Concatenate()([user_vec_input, movie_vec_sub])

    dense_layers=[]
    rating_layer=None
    for layer in orig_model.layers:
        if isinstance(layer, Dense):
            dense_layers.append(layer)
        elif isinstance(layer, RatingScale):
            rating_layer=layer

    new_dense_1=Dense(dense_layers[0].units, activation=dense_layers[0].activation, kernel_regularizer=dense_layers[0].kernel_regularizer)
    new_drop_1=Dropout(0.4)
    new_dense_2=Dense(dense_layers[1].units, activation=dense_layers[1].activation, kernel_regularizer=dense_layers[1].kernel_regularizer)
    new_drop_2=Dropout(0.4)
    new_out=Dense(dense_layers[2].units, activation=dense_layers[2].activation)
    new_scale=RatingScale()

    x=new_dense_1(merged_vec)
    x=new_drop_1(x, training=False)
    x=new_dense_2(x)
    x=new_drop_2(x, training=False)
    x=new_out(x)
    out=new_scale(x)

    sub_model=Model(inputs=[user_vec_input, movie_id_input], outputs=out)

    _=sub_model([np.zeros((1, embed_dim), dtype=np.float32), np.zeros((1,1), dtype=np.int32)])

    new_dense_1.set_weights(dense_layers[0].get_weights())
    new_dense_2.set_weights(dense_layers[1].get_weights())
    new_out.set_weights(dense_layers[2].get_weights())

    return sub_model



uservec_inference_model=build_uservec_submodel(ncf_model, EMBED_DIM)



# recommendation function
def get_recommendations(user_ratings_dict, top_n=6):
    """
    get movie recommendations for a new user
    - get the movie embedding matrix from the model
    - calculate the weighted sum of movie embeddings based on user ratings
    - normalize the user vector
    - exclude already rated movies from candidates
    - predict ratings for candidate movies
    - sort and select top-n recommendations
    - map movie ids back to original ids and get movie titles
    """
    movie_emb_matrix=ncf_model.get_layer("movie_embedding").get_weights()[0]
    weighted_sum=np.zeros((EMBED_DIM,), dtype=np.float32)
    total_rating=0.0

    for orig_m_id, rating in user_ratings_dict.items():
        if orig_m_id in movie_map:
            mapped_m_id=movie_map[orig_m_id]
            weighted_sum+=movie_emb_matrix[mapped_m_id]*rating
            total_rating+=rating

    if total_rating==0:
        user_vec=np.zeros((EMBED_DIM,), dtype=np.float32)
    else:
        user_vec=weighted_sum/total_rating

    rated_mapped_ids={movie_map[m] for m in user_ratings_dict if m in movie_map}
    all_mapped_ids=set(range(num_movies))
    candidates=list(all_mapped_ids-rated_mapped_ids)

    user_vec_batch=np.tile(user_vec, (len(candidates), 1))
    movie_batch=np.array(candidates).reshape(-1, 1)

    preds=uservec_inference_model.predict([user_vec_batch, movie_batch], verbose=0).flatten()
    idx_score_pairs=list(zip(candidates, preds))
    idx_score_pairs.sort(key=lambda x: x[1], reverse=True)

    top_30=idx_score_pairs[:TOP_K_CANDIDATES]

    random.shuffle(top_30)

    chosen=top_30[:top_n]
    chosen.sort(key=lambda x: x[1], reverse=True)

    results=[]
    for mapped_m, score in chosen:
        orig_id_candidates=[k for k, v in movie_map.items() if v==mapped_m]
        if not orig_id_candidates:
            continue
        orig_id=orig_id_candidates[0]
        row=movies[movies["movieId"]==orig_id]
        if not row.empty:
            results.append({
                "title": row.iloc[0]["title"],
                "movieId": int(orig_id),
                "score": round(float(score), 2)
            })
    return results

# test demo
'''
if __name__=="__main__":
    example_user_ratings={
        1: 5.0,
        50: 1.5,
        32: 4.0,
        260: 3.5,
        589: 1.0
    }
    recs=get_recommendations(example_user_ratings, top_n=10)
    print("recommended for new user:", recs)
'''
