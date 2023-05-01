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

def create_model(num_users, num_movies, num_factors, structure='dense3', lr=5e-3):
    if structure not in ('dense1', 'dense2', 'dense3', 'dot'):
        raise ValueError('structure must be dense1, dense2, or dot')

    # two input layers for movies and users
    user_input = keras.layers.Input(shape=(1,))
    movie_input = keras.layers.Input(shape=(1,))

    # two embedding layers for movies and users (transforms indices into dense vectors)
    user_embedding = keras.layers.Reshape((num_factors,))(keras.layers.Embedding(num_users,num_factors,embeddings_initializer='he_normal', embeddings_regularizer = keras.regularizers.l2(1e-16))(user_input))
    movie_embedding = keras.layers.Reshape((num_factors,))(keras.layers.Embedding(num_movies,num_factors,embeddings_initializer='he_normal', embeddings_regularizer = keras.regularizers.l2(1e-16))(movie_input))

    # output (and hidden) layers:
        
    if structure == 'dense1':
        # hidden:
        out = keras.layers.Concatenate()([user_embedding, movie_embedding])
        out = keras.layers.Dense(100, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('relu')(out)
        
        out = keras.layers.Dense(10, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('relu')(out)
        #out:
        out = keras.layers.Dense(1, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('sigmoid')(out)
    elif structure == 'dense2':
        # hidden:
        out = keras.layers.Concatenate()([user_embedding, movie_embedding])
        out = keras.layers.Dense(10, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('relu')(out)
        
        out = keras.layers.Dense(10, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('relu')(out)
        #out:
        out = keras.layers.Dense(1, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('sigmoid')(out)
    elif structure == 'dense3':
        # hidden:
        out = keras.layers.Concatenate()([user_embedding, movie_embedding])
        out = keras.layers.Dense(10, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('relu')(out)
        #out:
        out = keras.layers.Dense(1, kernel_initializer='he_normal')(out)
        out = keras.layers.Activation('sigmoid')(out)
    else:
        out = keras.layers.Dot(axes=1)([user_embedding, movie_embedding])
    
    # normalizer (to rating structure)
    out = keras.layers.Lambda(lambda x: x*(5-0.5)+0.5)(out)
    
    # Assembling & compiling model
    model = keras.models.Model(inputs = [user_input,movie_input], outputs=out)
    adam = keras.optimizers.Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model