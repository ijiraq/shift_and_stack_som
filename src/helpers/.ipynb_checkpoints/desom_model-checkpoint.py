
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#-------------------------------------------Imports

#Import a class
from src.DESOM.DESOM import DESOM
#from src.DESOM.SOM import SOMLayer

# #Load model
from keras.models import load_model
from keras.models import Model
# from timeit import default_timer as timer
import os
import numpy as np
from glob import glob
import pathlib
import sys
import matplotlib.pyplot as plt


# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# we can explicitly make assignments on it 
#this.desom = None

#------------------------------------------Load model
flag = False
desom = None  #Initialize as none



#--------------------------------------------Specify paths
current_path = os.getcwd()
#Where are Images Stored?
data_path =   os.path.join(current_path, 'data/im')

#Specify which model you wanna load
#--- 1 is best so far
pretrained_autoencoder = os.path.join(current_path,'results/tmp/different_DESOMS/DESOM1/ae_weights-epoch100.h5')
pretrained_model = os.path.join(current_path,'results/tmp/different_DESOMS/DESOM1/DESOM_model_final.h5')




# 100K dataset
saved_X = os.path.join(current_path,'data/X_train_desom2/X_train_copy.npy')
saved_y = os.path.join(current_path,'data/X_train_desom2/y_train_copy.npy')







'''
Return String Status of model training
'''
def get_flag():
    global flag
    return str(flag)




'''
Load X_train from saved_X

Returns X_train
'''
def load_X_train(saved_X):
    print("Loading X_train from: "+ saved_X)
    X_train = np.load(saved_X)
    print("X_train dimensions: {}".format(X_train.shape))
    return X_train


'''
Load X_train from saved_X

Returns X_train
'''
def load_y_train():
    print("Loading y_train from: "+ saved_y)
    y_train = np.load(saved_y)
    print("y_train dimensions: {}".format(y_train.shape))
    return y_train




'''
- Defines Architecture
- loads  Pretrained Model and AE
- Compiles model
- Returns model

'''
def load_trained_model(): 
    #Access global
    global flag
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
    filters = 8 #flat-32
    #i = pool_size
    i = 2
    ks = 3
    #desom is 15x15 with 8 filters
    desom = DESOM(encoder_dims= [np.power(32,2), 
                                [filters, ks, i],
                                [filters, ks, i], 
                                [filters, ks, i],
                                [filters,ks, i], 
                                                    [],
                                 [filters, ks, i],
                                 [filters, ks, i],
                                 [filters, ks, i], 
                                 [filters, 5, i],

                                 [1, ks, 0]],
                        ae_type = 'cnn',
                        map_size = (15,15) )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
   
    
    #-----------------Load AE Weights and Trained Model weights
    desom.load_ae_weights(pretrained_autoencoder)
    desom.load_weights(pretrained_model)
    
    flag = "Model Compiled" #Update
    
    return desom





'''
-----------------------LARGER MAPS-----------------


- Defines Architecture
- loads  Pretrained Model and AE FROM SPECIFIED Directories
- Compiles model
- Returns model

'''
def load_som1(pretrained_ae, pretrained_model): 
    #Access global
    global flag
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
    filters = 16 
    #i = pool_size
    i = 2
    ks = 3
    #desom is 15x15 with 8 filters
    desom = DESOM(encoder_dims= [np.power(32,2), 
                                [filters, ks, i],
                                [filters, ks, i], 
                                [filters, ks, i],
                                [filters,ks, i], 
                                                    [],
                                 [filters, ks, i],
                                 [filters, ks, i],
                                 [filters, ks, i], 
                                 [filters, 5, i],

                                 [1, ks, 0]],
                        ae_type = 'cnn',
                        map_size = (25,25) )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
#     #Specify directories
#     pretrained_autoencoder_ping = os.path.join(current_path,'ping_work/ae_weights-epoch100.h5')
#     pretrained_model_ping = os.path.join(current_path,'ping_work/DESOM_model_final.h5')
   
    
    #-----------------Load AE Weights and Trained Model weights
    desom.load_ae_weights(pretrained_ae)
    desom.load_weights(pretrained_model)
    
    flag = "Model Compiled" #Update
    
    return desom




'''
PARAMETERS:
    - DATA (X, y)
    - # Filters
    - Map Size
    - AE Epochs
    - SOM/DESOM Iterations
    - Save directories




- Defines Architecture
- Trains AE according to specified pre_train epochs
- Trains Model and SOM according to specified epochs

- SAVES to specified files

- Compiles model
- Returns model

'''
def train_specific_model(X_train, y_train,
                          #Architecture
                          num_filters, map_size,
                          #Epochs
                          ae_epochs, som_iters, model_iters,
                          #Save directories
                          ae_savepath,
                          model_savepath): 
    
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
    filters = num_filters #flat-32
    #i = pool_size
    i = 2
    ks = 3
    desom = DESOM(encoder_dims= [np.power(32,2), 
                                [filters, ks, i],
                                [filters, ks, i], 
                                [filters, ks, i],
                                [filters,ks, i], 
                                                    [],
                                 [filters, ks, i],
                                 [filters, ks, i],
                                 [filters, ks, i], 
                                 [filters, 5, i],

                                 [1, ks, 0]],
                        ae_type = 'cnn',
                        map_size = map_size )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
   
    
    #-----------------Pretrain AE and Train Model
    #Pretrain AE
    desom.pretrain(X_train, 
                  optimizer='adam',
                     #Epochs
                     epochs = ae_epochs,
                     batch_size=256,
                     save_dir= ae_savepath)
    
    
    #Train model
    desom.init_som_weights(X_train)
    desom.fit(X_train, 
              y_train,              
              Tmax = 10.0,
              #Iterations
              iterations = model_iters,
              som_iterations= som_iters,
              save_dir = model_savepath)
           
    flag = "Model Compiled" #Update
    
    return desom
    
    
    
'''
Plot DESOM Map and Save it to directory
Optional Parameter: Filename
'''
def save_grid_plot(fname = 'DESOM_Grid'):
    #Access global 
    global desom
    
    #Find Decoded prototypes
    decoded_prototypes = desom.decode(desom.prototypes)

    #Setup Map Size
    map_size = desom.map_size

    #Set up 32x32 size
    img_size = 32
    #img_size = int(np.sqrt(X_train.shape[1]))

    #Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))
    #Iterate over each and every prototype
    for k in range(map_size[0]*map_size[1]):
        ax[k // map_size[1]][k % map_size[1]].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap='gray')
        ax[k // map_size[1]][k % map_size[1]].axis('off')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    #Filename
    filename = fname +'.png'
    img_savepath = os.path.join(current_path,'data/plots/',filename)
    plt.savefig(img_savepath)
    
    print("DESOM Map saved as {} at {}.".format(filename, img_savepath))
    
    
    
'''
This function uses the LOADED DESOM to generate a Distance Map

saved_X specifies location

RETURNS the Distance map
'''    
def get_distance_map(saved_X):
    #Refer to DESOM (loaded)
    global desom
    
    #Throw error if NONE?
    
    #Load X-train to make predictions
    X_train = load_X_train(saved_X)

    #Get Predicted Labels
    y_pred = desom.predict(X_train)

    #Distance map
    # i - point in X(data)
    # j - Assigned CELL on SOM
    distance_map = desom.map_dist(y_pred)
    
    return distance_map


'''
Simply returns map size

VERY HELPFUL
'''
def get_map_size():
    global desom
    return desom.map_size


'''
Find DECODED Prototypes

'''
def get_decoded_prototypes():
    global desom
    decoded_prototypes = desom.decode(desom.prototypes)
    
    return decoded_prototypes
    




def print_save_directories():
    #Trained Model directories
    print("Pretrained AE at: " + pretrained_autoencoder)
    print("Pretrained DESOM (15x15) at: " + pretrained_model)

    
    
'''
Check if model is not None.
If it has been defined, print map size and architecture.
'''
def print_model_summary():
    #Refer to global
    global desom
    
    if(desom == None):
        print("Model has not been defined!")
        return
    else:
        print("Model has been defined...")
        print("Map Size: {}".format(get_map_size()))
        print(model.summary())
        
        
        
    



































