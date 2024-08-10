import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import keras_tuner 
import itertools
import util 
from sklearn.model_selection import train_test_split


from models import HyperTrans
from models   import simple_model, Trans_encoder
from pos_encoding import positional_encoding
from pos_encoding import inp_transform ,mlp_relu , mlp_gelu

#####    Random seen   #####
keras.utils.set_random_seed(0)

#tf.config.experimental.enable_op_determinism()


##### Current directory ####                                                                                       

SEED = 1
util.set_global_determinism(seed=SEED)
cur_dir  = os.getcwd()

#### Select 1 gpu ####

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Specify GPU to avoid randomness due to different GPU utilization
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu  

#####################

ntest = 50
memory = 41
step = 0
num_epochs = 500

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

asym = "asym"





tuner = keras_tuner.BayesianOptimization(
    hypermodel=HyperTrans(),
    # Specify the name and direction of the objective.                                                      
    #objective=keras_tuner.Objective("val_custom_metric", direction="min"),                                 
    max_trials=50,
    seed=SEED,
    overwrite=True,
    directory="Test2",
    project_name="50_iteration_Fix",
)



tuner.search(X_train0 = X_train0, X_train1 = X_train1, Y_train = Y_train, X_val0 = X_val0, X_val1 = X_val1 ,Y_val= Y_val, ntimes = ntimes  ,traj_test = traj_test_as)
tuner.results_summary()

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

### Get 2 best model 
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()
best_model.save("best_model.keras")
error_t = util.error(best_model,traj_test_as, memory, ntimes, ntest, cur_dir, 0, asym , plot = True)




