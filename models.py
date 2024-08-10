import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn import preprocessing
import keras_tuner
from sklearn.metrics import confusion_matrix
import itertools
import util
import os
import random

#import keras_nlp
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from pos_encoding import positional_encoding, positional_encoding2
from pos_encoding import inp_transform ,mlp_relu , mlp_gelu, point_wise_feed_forward_network


ntest = 50
num_epochs = 500
learning_rate = 0.0001
cur_dir  = os.getcwd()
 

num_features = 1 ## predciting 1 feature i.e sigma_z at next time step ##                  
memory = 41      ## memory lenght of the trajecotry ##                                             
input_shape = (41,1)  ## input for keras model (memory,features)                                                
projection_dim = 512  ## embeding dimesion for represtation, now each time step of sigma_z ##                
                      ## is encoded in a vector of size projection_dim ##                                                  
num_heads = 4         ## Number of head for the transformer                                                 
transformer_units = [ projection_dim * 2, projection_dim, ]  #number of neurons for the dense                   
                                                             #layer after the SA, each element                    
                                                             # of the list coorespond to a dense layer            
transformer_layers = 2     # Number of layer of tnasfomer
dff = 512
mlp_head_units = [512, 512]  # Size of the dense layers of the final output (decoder)  

step = 0
sym = "sym"
asym = "asym"

def simple_model():
    input0 = layers.Input(shape=input_shape)
    input1 = layers.Input(shape=input_shape)

    data = {'input':input0 ,
            'times':input1}

    x_pe = positional_encoding(data['times'], memory,projection_dim) ####

    x_transformed = inp_transform(projection_dim)(data['input'])                 ###

    transformed_input = x_transformed + x_pe
    x = Dropout(0.2)(transformed_input, training=False) ####

    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    # Create the Keras model.
    model = keras.Model(inputs=[input0,input1], outputs=out)
    return model


def Trans_encoder_sym():
    ### Parameters ###                                                                                                                                                    
    num_heads = 2         ## Number of head for the transformer                                                        
    val_dim = 32  ## embeding dimesion for represtation SA                                                           
    projection_dim = int(val_dim*num_heads) #Concadenated embeding dimesion for represtation after the MHA                                                                
    transformer_layers = 2      # Number of layer of tnasfomer                                                         
    dff = 1920             ## point wise FFN                                                                           
    n1 = 128    ##Dense n1 decoder                                                                                     
    n2 = 448     ##Dense n2 decoder  


    #####################
    
    input0 = layers.Input(shape=input_shape)
    input1 = layers.Input(shape=input_shape)
    
    data = {'input':input0 ,
            'times':input1}

    print(data['input'].shape)
    
    #x_pe = positional_encoding(data['times'], memory,projection_dim) ####                                                                                                
    x_pe = positional_encoding2(data['times'], projection_dim) ####                                                                                                       
    
    x_transformed = inp_transform(projection_dim)(data['input'])                 ###                                                                                      
    
    transformed_input = x_transformed + x_pe
    x_input = transformed_input #Dropout(0.2)(transformed_input, training=False) ####                                                                                     
    
    for _ in range(transformer_layers):
        # Layer normalization 1.                                                                                                                                          
        #x1  = layers.LayerNormalization(epsilon=1e-6)(x)                                                                                                                 
        # Create a multi-head attention layer.                                                                                                                            
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, value_dim =val_dim  ,
            dropout=0.1)( x_input, x_input, use_causal_mask=True  )  ####Masked#### use_causal_mask=True                                                                  
        # Skip connection 1.                                                                                                                                              
        #print("_______attention_output________",attention_output.shape)                                                                                                  
        x = layers.Add()([attention_output, x_input])
        # Layer normalization 2.                                                                                                                                          
        x  = layers.LayerNormalization(epsilon=1e-6)(x)
        # MLP.                                                                                                                                                            
        x2 = point_wise_feed_forward_network(projection_dim,dff)(x)
        
        #x3 = mlp_relu(x3, hidden_units=transformer_units, dropout_rate=0.1)                                                                                              
        #print("_______mpl_relu________",attention_output.shape)                                                                                                          
        # Skip connection 2.                                                                                                                                              
        
        x = layers.Add()([x2, x])
        x_input = layers.LayerNormalization(epsilon=1e-6)(x)


    print(x_input.shape)
    x_out = layers.Flatten()(x_input)
    x_out = layers.Dense(n1, activation="relu")(x_out)
    x_out = layers.Dense(n2, activation="relu")(x_out)
    out = layers.Dense(num_features, activation="linear")(x_out)
    # Create the Keras model.                                                                                                                                             
    model = keras.Model(inputs=[input0,input1], outputs=out)
    
    return model


def Trans_encoder():
    input0 = layers.Input(shape=input_shape)
    input1 = layers.Input(shape=input_shape)

    data = {'input':input0 ,
            'times':input1}

    print(data['input'].shape)

    x_pe = positional_encoding(data['times'], memory,projection_dim) ####

    x_transformed = inp_transform(projection_dim)(data['input'])                 ###

    transformed_input = x_transformed + x_pe
    x_input = transformed_input #Dropout(0.2)(transformed_input, training=False) ####

    for _ in range(transformer_layers):
        # Layer normalization 1.
        #x1  = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
                              num_heads=num_heads, key_dim=projection_dim, value_dim =int(projection_dim/num_heads)  , dropout=0.1 
                              )(x_input, x_input, use_causal_mask=True  )  ####Masked#### use_causal_mask=True
        # Skip connection 1.
        #print("_______attention_output________",attention_output.shape)

        x = layers.Add()([attention_output, x_input])
        # Layer normalization 2.
        x  = layers.LayerNormalization(epsilon=1e-6)(x)
        # MLP.
        x2 = point_wise_feed_forward_network(projection_dim,dff)(x)
        
        #x3 = mlp_relu(x3, hidden_units=transformer_units, dropout_rate=0.1)
        #print("_______mpl_relu________",attention_output.shape)

        # Skip connection 2.
        x = layers.Add()([x2, x])
        x_input = layers.LayerNormalization(epsilon=1e-6)(x)
        
    #representation = layers.LayerNormalization(epsilon=1e-6)(x)
    #representation = layers.Flatten()(representation)
    #representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    #features = mlp_relu(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    # Classify outputs.
    #logits = layers.Dense(num_classes, activation=tf.nn.softmax)(features)
    #features = layers.Dense(1024, activation="relu")(representation)
    print(x_input.shape)
    x_out = layers.Flatten()(x_input)
    x_out = layers.Dense(2048, activation="relu")(x_out)
    x_out = layers.Dense(1024, activation="relu")(x_out)
    out = layers.Dense(num_features, activation="linear")(x_out)
    # Create the Keras model.
    model = keras.Model(inputs=[input0,input1], outputs=out)
    return model



class HyperTrans(keras_tuner.HyperModel):
    def build(self,hp):

        ### Parameters ###
        num_heads = hp.Int("num_head", min_value=1, max_value=8, step=1)         ## Number of head for the transformer
        val_dim = hp.Int("val_dim", min_value=32, max_value=128, step=32)  ## embeding dimesion for represtation SA
        projection_dim = int(val_dim*num_heads) #Concadenated embeding dimesion for represtation after the MHA            
        transformer_layers = hp.Int("SA layer", min_value=1, max_value=3, step=1)      # Number of layer of tnasfomer                  
        dff = hp.Int("pwFF", min_value=128, max_value=2048, step=128)             ## point wise FFN
        n1 = hp.Int("n1", min_value=128, max_value=2048, step=128)     ##Dense n1 decoder
        n2 = hp.Int("n2", min_value=128, max_value=2048, step=128)     ##Dense n2 decoder


        ##################

        
        input0 = layers.Input(shape=input_shape)
        input1 = layers.Input(shape=input_shape)

        data = {'input':input0 ,
                'times':input1}

        print(data['input'].shape)

        #x_pe = positional_encoding(data['times'], memory,projection_dim) ####
        x_pe = positional_encoding2(data['times'], projection_dim) ####                                     
        #x_pe = inp_transform(projection_dim)(data['times'])                 ###                              
        x_transformed = inp_transform(projection_dim)(data['input'])                 ###                              

        transformed_input = x_transformed + x_pe
        x_input = transformed_input #Dropout(0.2)(transformed_input, training=False) ####                             

        for _ in range(transformer_layers):
            # Layer normalization 1.                                                                              
            #x1  = layers.LayerNormalization(epsilon=1e-6)(x)                                                     
            # Create a multi-head attention layer.                                                                
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, value_dim =val_dim  ,
                dropout=0.1)( x_input, x_input, use_causal_mask=False  )  ####Masked#### use_causal_mask=True     
            # Skip connection 1.                                                                                  
            #print("_______attention_output________",attention_output.shape)                                     
            x = layers.Add()([attention_output, x_input])
            # Layer normalization 2.                                                                              
            x  = layers.LayerNormalization(epsilon=1e-6)(x)
            # MLP.
            x2 = point_wise_feed_forward_network(projection_dim,dff)(x)

            #x3 = mlp_relu(x3, hidden_units=transformer_units, dropout_rate=0.1)                         
            #print("_______mpl_relu________",attention_output.shape)
            # Skip connection 2.
        
            x = layers.Add()([x2, x])
            x_input = layers.LayerNormalization(epsilon=1e-6)(x)

    
        print(x_input.shape)
        x_out = inp_transform(1)(x_input)
        x_out = layers.Flatten()(x_out)
        x_out = layers.Dense(n1, activation="relu")(x_out)
        x_out = layers.Dense(n2, activation="relu")(x_out)
        out = layers.Dense(num_features, activation="linear")(x_out)
        # Create the Keras model.                                                                                 
        model = keras.Model(inputs=[input0,input1], outputs=out)


        optimizer = Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()],
                      )

        return model



    def fit(self, hp, model, X_train0, X_train1, Y_train, X_val0, X_val1 ,Y_val ,traj_test, ntimes, **kwargs):
        model.fit(x=[X_train0, X_train1],
                  y=Y_train,
                  batch_size=hp.Int("batch_size", min_value=64, max_value=512, step=64), ###                      
                  epochs=num_epochs,     ###                                                                     
                  validation_data=([X_val0, X_val1],Y_val),
                  verbose=0,
                  **kwargs)
        #error_t = util.error(model,traj_test, memory, ntimes, ntest, cur_dir)
        error_t = util.error(model,traj_test, memory, ntimes, ntest, cur_dir, step, asym , plot = False)
        return error_t
