# from predictor import Predictor

import numpy as np
import keras
"""
class NNPredictor(Predictor):
    def __init__(self) -> None:
        self.model = None

    @property
    def name(self):
        return "Neural Network"
    
    def train(self, ratings:np.ndarray) -> bool:
        pass

    def predict(self, user, movie):
        pass
"""

def create_model(num_users, num_movies, num_factors):
    # two input layers for movies and users
    user_input = keras.layers.Input(shape=(1,))
    movie_input = keras.layers.Input(shape=(1,))

    # two embedding layers for movies and users (transforms indices into dense vectors)
    user_embedding = keras.layers.Reshape((num_factors,))(keras.layers.Embedding(num_users,num_factors,embeddings_initializer='he_normal', embeddings_regularizer = keras.regularizers.l2(1e-16))(user_input))
    movie_embedding = keras.layers.Reshape((num_factors,))(keras.layers.Embedding(num_movies,num_factors,embeddings_initializer='he_normal', embeddings_regularizer = keras.regularizers.l2(1e-16))(movie_input))

    # dense hidden layer
    # ...
    
    # output layer
    out = keras.layers.Dot(axes=1)([user_embedding, movie_embedding])
    
    # Assembling & compiling model
    model = keras.models.Model(inputs = [user_input,movie_input], outputs=out)
    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model