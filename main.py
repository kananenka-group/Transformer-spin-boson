import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
import random
import keras_nlp
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
#############

from models   import simple_model, Trans_encoder
from pos_encoding import positional_encoding
from pos_encoding import inp_transform ,mlp_relu , mlp_gelu

#####    Random seen   #####
##### Current directory ####

SEED = 0
util.set_global_determinism(seed=SEED)
cur_dir  = os.getcwd()

#### Select 1 gpu ####

os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu

#### Parameters #####

ntest = 50   ## Numebr of test files to hyperparameters ##
#num_features = 1 ## predciting 1 feature i.e sigma_z at next time step ## 
memory = 41      ## memory lenght of the trajecotry ##
#input_shape = (41,1)  ## input for keras model (memory,features)

#projection_dim = 512  ## embeding dimesion for represtation, now each time step of sigma_z ##
                      ## is encoded in a vector of size projection_dim ##

#num_heads = 1         ## Number of head for the transformer

#transformer_units = [ projection_dim * 2, projection_dim, ]  #number of neurons for the dense
                                                             #layer after the SA, each element
                                                             # of the list coorespond to a dense layer  
#transformer_layers = 1     # Number of layer of tnasfomer 

#mlp_head_units = [512, 512]  # Size of the dense layers of the final output (decoder)


#### Import data ####

## Training ###
x_train = np.load("/home/leherrer/Documents/ML_QD/Pavlo_data/Transformer_data/Asymetric/asym_x.npy")
y_train = np.load("/home/leherrer/Documents/ML_QD/Pavlo_data/Transformer_data/Asymetric/asym_y.npy")


## out of training data (testing, hyperparameters) ##
traj_test = np.load("/home/leherrer/Documents/ML_QD/Pavlo_data/Transformer_data/test_sym/traj_test.npy")
ntimes = traj_test.shape[1]

traj_test_as = np.load("/home/leherrer/Documents/ML_QD/Pavlo_data/Transformer_data/test_asym/traj_test_as.npy")

## Split Train and validation ##

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
print(f"x_train shape: {X_train.shape} - y_train shape: {Y_train.shape}")
print(f"x_val shape: {X_val.shape} - y_val shape: {Y_val.shape}")


## This data have the shape X = (None, memory, 9) and Y =(None,1,9) where the 9 elements corrposnds to
## #0 sigma_z #1 time #2 rho_00 #3 rho_11 #4-8 ep,delta,lamb,wc,beta.
## we need 2 imputs for the mode, we select only the sigma_z and the time ##

X_train0, X_train1, = X_train[:,:,0], X_train[:,:,1]
X_val0, X_val1, = X_val[:,:,0], X_val[:,:,1]

Y_train = Y_train[:,0]
Y_val = Y_val[:,0]

#### Def run the model ###

def run_experiment(model,
                    learning_rate = 0.0001,
                    batch_size = 256,
                    num_epochs =1500,
                    step = 0):

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                    loss=keras.losses.MeanSquaredError(),
	                metrics=[keras.metrics.MeanSquaredError()],
                    )

    history = model.fit(
	x=[X_train0, X_train1],
	y=Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
	    validation_data=([X_val0, X_val1],Y_val),
        verbose=1
    )
    print(model.summary())
    model.save("model"+str(step)+".keras")
    return history , model

### Run the model #####

model_test =  Trans_encoder()
history , model = run_experiment(model_test)
util.validation_curve(history)  ## plot learing curve



### Test model on test data ###
step = 0
sym = "sym"
asym = "asym"
util.error(model,traj_test, memory, ntimes, ntest, cur_dir, step, sym,  plot = True)
util.error(model,traj_test_as, memory, ntimes, ntest, cur_dir, step, asym , plot = True)

