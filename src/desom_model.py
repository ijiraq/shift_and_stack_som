#-------------------------------------- DEFAULT TRAINING VERSION---------------------


import numpy as np
#Import a class
from DESOM import DESOM
# #Load model
from keras.models import load_model
from keras.models import Model

import os
import matplotlib.pyplot as plt




current_path = os.getcwd()



'''
Default training-adjusted

- Defines Architecture
- loads  Pretrained Model and AE
- Compiles model
- Returns model

PARAMS:
    - Map Size
    - AE
    - Model

'''
def load_trained_SOM1(map_size, pretrained_ae, pretrained_model): 
    #Access global
    global flag
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
  
    es = 16
    encoder_dims = [np.power(32,2),#1024
                    [es, 3, 2], [es, 3, 2], [es, 3, 2], [es, 3, 2],
                    [],
                    [es, 3, 2], [es, 3, 2], [es, 3, 2], [es, 3, 2], 
                    [1, 3, 0]]
   
    desom = DESOM(encoder_dims = encoder_dims,
                        ae_type = 'cnn',
                        map_size = map_size )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
   
    
    #-----------------Load AE Weights and Trained Model weights
    desom.load_ae_weights(pretrained_ae)
    desom.load_weights(pretrained_model)
    
    
    return desom





    
'''
Plot DESOM Map and Save it to directory
Optional Parameter: Filename

If multiple, save all epochs in different folder
'''
def save_grid_plot(multiple = False, fname = 'DESOM_Grid'):
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
    img_savepath = os.path.join(current_path,'plots/',filename)
    
    if(multiple):
        #Filename
        filename = fname +'.png'
        img_savepath = os.path.join(current_path,'plots/multiple_epochs/',filename)
        
    plt.savefig(img_savepath)
    print("DESOM Map saved as {} at {}.".format(filename, img_savepath))
    
    return
    
    
    
    
    
    





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