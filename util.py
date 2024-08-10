import numpy as np
import matplotlib.pyplot as plt
import random as rm
import glob
from scipy.interpolate import CubicSpline
import os
import random 
import tensorflow as tf


def plot_pd(base_dir, tgrid, r00, tgrid2, _r00, ids, step, sym):

    fontsize=16

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.axis([0,20,-1,1])

    ax.plot(tgrid, r00, color='g')
    ax.plot(tgrid2, _r00, color='g', ls=':')


    plt.tight_layout()
    ax.set_xlabel("Time",fontsize=fontsize)
    ax.set_ylabel("Rho",fontsize=fontsize)
    fn = base_dir + "/plots_"+ str(sym)   + "/" + str(ids) + "_" +  str(step)  +".pdf"
    plt.savefig(fn,dpi=1200,bbox_inches='tight')
    plt.close(fig)

def save(cur_dir, tgrid, traj, tgrid2, traj1,  ids, step, sym):

    fname = cur_dir+ "/plots_"+ str(sym)   + "/" + str(ids) + "_" + str(step)   + ".dat"
    f = open(fname, "w")

    ntime = tgrid.shape[0]

    for n in range(ntime):
       f.write(" %7.5f %7.5f %7.5f %7.5f \n"%
               (tgrid[n], traj[n], tgrid2[n], traj1[n]  ) )

    f.close()





    
def compare(t_ex, traj_ex, t_pred, traj_pred):
    #cs = CubicSpline(t_ex, traj_ex)
    #new_points = cs(t_pred)
    #diference = np.abs(new_points - traj_pred)
    diference = np.abs(traj_ex - traj_pred)
    error = np.sum(diference)
    return error



def error(model,traj_test, memory, ntimes, ntest, cur_dir, step ,sym ,plot = False):

    traj1=np.zeros_like(traj_test)
    error_t=0.0

    for i in range(ntest): 
        error=0.0
        traj1[i,:,:]=traj_test[i,:,:]
        for n in range(ntimes-memory):
            x_inp=traj1[i,n:n+memory,:].reshape(1,memory,9)
            x0= x_inp[:,:,0].reshape(1,memory)
            x1= x_inp[:,:,1].reshape(1,memory)    
            yhat = model.predict([x0,x1], verbose=False)
            traj1[i,n+memory,0]=yhat
        if plot:
            plot_pd(cur_dir, traj_test[i,:,1] , traj_test[i,:,0], traj_test[i,:,1], traj1[i,:,0], i,step, sym)
            save(   cur_dir, traj_test[i,:,1] , traj_test[i,:,0], traj_test[i,:,1], traj1[i,:,0] , i,step, sym)
            
        #error = np.sum(np.abs(traj_test[i,memory:,0]- traj1[i,memory:,0]))/len(np.abs(traj_test[i,memory:,0]))                        
        error = compare(traj_test[i,:,1], traj_test[i,:,0], traj_test[i,:,1] , traj1[i,:,0])/len(np.abs(traj_test[i,memory:,0]))

        #error /= ntest                                                                                                                
        error_t += error
        print (" Errors %d : %10.5f "%(i,error))
    print (" Total and average errors %10.5f %10.5f \n"%(error_t,error_t/ntest))

    return error_t



def validation_curve(history):
    ##### curves accuracy                                                                                   
    plt.figure(figsize=(4, 4))
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig("Training_curve_acuracy.pdf")


    plt.figure(figsize=(4,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig("Training_curve_loss.pdf")



def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

#############
def set_global_determinism(seed):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
