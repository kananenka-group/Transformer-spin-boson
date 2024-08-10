import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools
import util
import os
import random
#import keras_nlp
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def get_position_encoding(time, memory, d, n=1000):
        P = np.zeros((memory, d))
        #time = time.reshape(memory)
        #time = time.numpy()
        for k in range(memory):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(time[k]/denominator)
                P[k, 2*i+1] = np.cos(time[k]/denominator)
        return P

def positional_encoding(position_indices, memory, projection_dim):
        #position_indices = tf.range(start=0, limit=memory, delta=1)
        #position_indices = np.random.randint(100, size=memory)

        position_embedding_matrix = get_position_encoding(position_indices,memory, projection_dim)
        position_embedding_layer = layers.Embedding(
                                    input_dim=memory, output_dim=projection_dim,
                                    weights=[position_embedding_matrix],
                                    trainable=False)
        print("_______________--indices shape_______________- ",position_indices.shape)
        embedded_indices = position_embedding_layer(position_indices[:,:,0])
        #embedded_indices = tf.squeeze(embedded_indices)
        print("p encoding shape", embedded_indices.shape )
        return embedded_indices

def inp_transform(projection_dim):
    return tf.keras.layers.Dense(projection_dim)


def mlp_gelu(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def mlp_relu(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles_original(times, d_model, base=1000):
    dim_indices = tf.range(d_model, dtype=tf.float32)
    
    exponent = tf.divide(tf.multiply(2., dim_indices),
                         tf.cast(d_model, tf.float32))

    angle_rates = tf.pow(tf.cast(base, dtype=tf.float32), exponent)
    angle_rates = tf.math.reciprocal(angle_rates)
    angle_rates = times * angle_rates
    return angle_rates

def fn(x):
    if x[1] % 2 == 0:
        return (tf.sin(x[0]), x[1])
    else:
        return (tf.cos(x[0]), x[1])

class MapLayer(layers.Layer):
    def call(self, input,indices):
        return tf.map_fn(lambda x: fn(x),  (input, indices))[0]


def positional_encoding2(times, d_model, base=1000, mjd=False):
    indices = times
    #indices = tf.range(tf.shape(times)[1], dtype=tf.float32)
    #indices = tf.expand_dims(indices, 0)
    #indices = tf.tile(indices, [tf.shape(times)[0], 1])
    #indices = tf.expand_dims(indices, 2)

    angle_rads = get_angles_original(indices, d_model)

    # SIN AND COS
    def fn(x):
        if x[1] % 2 == 0:
            return (tf.sin(x[0]), x[1])
        else:
            return (tf.cos(x[0]), x[1])

    x_transpose = tf.transpose(angle_rads, [2,1,0])
    
    indices = tf.range(0, tf.shape(x_transpose)[0])
    #x_transpose = tf.map_fn(lambda x: fn(x),  (x_transpose, indices))[0]
    x_transpose = MapLayer()(x_transpose,indices)
    pos_encoding = tf.transpose(x_transpose, [2, 1, 0])
    return tf.cast(pos_encoding, dtype=tf.float32)
