'''
Date of creation: 2020-04-14 (Week 15)
- This module contains utility functions for Deep SOM-2
'''

#--------------------------------------------------- Imports ---------------------------------------------------
import pandas as pd
import os
from timeit import default_timer as timer
import numpy as np
from random import sample
from sklearn.preprocessing import MinMaxScaler

from helpers import desom_model as dm
from helpers import utilities as uti
from helpers import stat_analysis as stan
from DESOM import DESOM

import seaborn as sn
import matplotlib.pyplot as plt
from pickle import (load, dump)

#--------------------------------------------------- Reusable components ---------------------------------------------------
map_size = None


#Some reusable components
lookup_class  = {"0":"t", "1":"g", 
                 "2":"r" , "3":"f" , "4":"b"}
ramp_map = {0: 1.2, 
            1: 1.4 , 
            2: 1.6, 
            3: 1.8,
            4: 2.0}
        



#--------------------------------------------------- Paths ---------------------------------------------------
current_path = os.getcwd()
# ae_savepath  = os.path.join(current_path, 'results/deep_som2/')
# model_savepath = os.path.join(current_path, 'results/deep_som2/')







#--------------------------------------------------- Training ---------------------------------------------------




'''
This function fits a MMS to the data  and saves the MMS.
    -Normalizes the input data

PARAMS:
    - Data Matrix
    - Directory to save file to
RETURNS:
    - MMS-Transformed X  
'''
def preprocess(X, mms_savepath):
    
    #Initialize MMS
    mms = MinMaxScaler(feature_range=(0, 1))  
    #Preprocess
    X = mms.fit_transform(X)

    #ave the MMS
    mms_savename = os.path.join(mms_savepath,'mms_scaler.pkl')
    print("Saving fitted-MMS as {}".format(mms_savename))
    dump(mms, open(mms_savename, 'wb'))

    return X







'''
This function loads a FITTED MMS

PARAMS:
    - Directory to load MMS FROM
RETURNS:
    - Loaded MMS  
'''
def load_mms(mms_savepath):
    
    mms_savename = os.path.join(mms_savepath,'mms_scaler.pkl')
    print("Loading fitted-MMS FROM {}".format(mms_savename))
    mms = load(open(mms_savename, 'rb'))

    return mms






'''
This function normalizes the given count vector.
THE VECTOR MUST BE SHAPED 

PARAMS:
    - input vector
    - Loaded MMS
RETURNS:
    - Normalized Vector  
'''
def normalize_input_with_mms(x, mms):
    #Reshape each BMU as a row
    #x = x.reshape(-1, 1)
    #Normalize
    x = mms.transform(x)

    return x








'''
This function trains a fresh DEEP SOM-2

PARAMS:
    - Data
    - Map dimensions TUPLE
    - z (Latent dimensions)
    - SAVEPATHS
RETURNS:
    - Fully trained SOM-2
'''
def train_som(X, map_size,
               #Hyperparameters
               z, Tmax, Tmin,
               #Training duration
               ae_epochs, iters, som_iters,
               model_savepath, ae_savepath):
    
    #Set GLOBAL
    global map_dims
    map_dims = map_size
   
    #Define architecture
    encoder_dims = [X.shape[-1], #625
                    500, 256, 150,
                    z]   #z


    som2 = DESOM(encoder_dims = encoder_dims,
                        ae_type = 'fc',
                        map_size = map_size)
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    som2.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    som2.compile(gamma, optimizer)
    som2.model.summary()
   
    #-----------------Pretrain AE and Train Model
    #Pretrain AE
    som2.pretrain(X, 
                  optimizer='adam',
                     #Epochs
                     epochs = ae_epochs,
                     batch_size = 256,
                     save_dir= ae_savepath)
    #Train model
    som2.init_som_weights(X)
    som2.fit(X,              
              Tmax = Tmax,
              Tmin = Tmin,
              #Iterations
              iterations = iters,
              som_iterations= som_iters,
              save_epochs = 100,
              save_dir = model_savepath)
    
    return som2

    

    
'''
This function loads a TRAINED DEEP SOM-2

PARAMS:
    - Map dimensions TUPLE
    - z (Latent dimensions)
    - SAVEPATHS
RETURNS:
    - Fully-loaded trained SOM-2
'''
def load_som(input_dims, map_size, z,
               pretrained_model, pretrained_ae):
    
    #Set GLOBAL
    global map_dims
    map_dims = map_size
   
    #Define architecture
    encoder_dims = [input_dims, #625
                    500, 256, 150,
                    z]   #z


    som2 = DESOM(encoder_dims = encoder_dims,
                        ae_type = 'fc',
                        map_size = map_size)
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    som2.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    som2.compile(gamma, optimizer)
    som2.model.summary()
   
    #-----------------Load AE Weights and Trained Model weights
    som2.load_ae_weights(pretrained_ae)
    som2.load_weights(pretrained_model)
    
    return som2











#--------------------------------------------------- Utilities ---------------------------------------------------

'''
This function accumulates Euclidean distance to each nearest neighbour

PARAMETERS:
    - MAP_SIZE 
RETURNS:
    - IDX_MAP and NODE_MAP
'''
def generate_conversion_dicts(map_size):
    
    grid_coordinates = uti.generate_list_of_coords(map_size) #All Nodes in a list

    #STEP 1: Generate IDX MAP
    idx_map = {}
    for k in grid_coordinates:
        w = map_size[0] #Width of MAP
        #Convert Grid NODE to IDX
        arr_i = k[0] + w * k[1]
        #Initialize
        idx_map[k] = arr_i  
    
    #STEP 2: Generate NODE MAP    
    node_map = {}   
    for k in range(map_size[0] * map_size[1]): 
                 #Convert to grid NODE
                x = k // map_size[1]
                y = k % map_size[1]
                #Form coordinate
                node = (x,y)
                #IDX -> Node
                node_map[k] = node
                
                
    return idx_map, node_map

    




'''
This function generates the appropriate MATRICES

PARAMETERS:
    - SOM 
RETURNS:
    - Prototypes and DECODED Prototypes
'''
def get_prototypes(som):
    
    prototypes = som.prototypes
    decoded_prototypes = som.decode(prototypes) 
    
    print("Prototypes(shape): {} ".format(prototypes.shape))
    print("DECODED Prototypes(shape): {} \n".format(decoded_prototypes.shape))
    
    return prototypes, decoded_prototypes



'''
This function generates the MiniSom-esque HEATMAP

PARAMETERS:
    - SOM 
    - IDX MAP
RETURNS:
    - HEATMAP MATRIX, TRANSPOSED as needed
'''
def distance_map(som, idx_map):
    
    # global map_dims

    #Generate M
    print("Generating Prototypes and DECODED Prototypes...")
    M, _ = get_prototypes(som)
    
    print("Generating EUCLIDEAN DISTANCE DICT... \n")    
    euc_map = distmap_dict_flexible(M, idx_map, som.map_size)
    
    #Container array
    map_dims = som.map_size #FIX
    A = np.zeros(shape = map_dims[0] * map_dims[1])
    #Populate
    for key,euc_distance in euc_map.items():
        #Generate idx
        k = idx_map[key]
        #Cumulative Distance to neighbours
        A[k] = euc_distance

    #Normalize
    print("Normalizing dist_map... \n")
    A = A/A.max()
    
    print("Reshaping dist_map and taking transpose... \n")
    A = A.reshape(map_dims).T
    
    print("Map shape: {}".format(A.shape))
    
    return A

    



'''
This function PLOTS the MiniSom-esque HEATMAP

PARAMETERS:
    - Heatmap matrix
RETURNS:
    - 
'''
def plot_deepsom_heatmap(heatmap_matrix):

    plt.figure(figsize = (8,8))
    sn.heatmap(heatmap_matrix)
    plt.show()

    return







#--------------------------------------------------- Nearest Neighbours ---------------------------------------------------

    
    
    
    
'''
This function accumulates Euclidean distance to each nearest neighbour

PARAMETERS:
    - Weights Matrix from PROTOTYPES(M)
    - IDX MAP
RETURNS:
    - Distmap mapping (NODE)=>(TOTAL Euclidean distance to NEIGHBOURS)

'''
def distmap_dict_flexible(M, idx_map, map_dims):
    
    #Setup thresholds
    #global map_dims
    max_i, max_j = map_dims[0], map_dims[1]   
    print("Thresholds: \t i:{} \t j:{}".format(max_i, max_j))

    #Initialize Dictionary
    dist_map = {}  
    
    #Iterate over FULL weight matrix, since limit 
    #        IDX will be discarded as invalid later anyway
    for i in range(0, max_i):       
        for j in range(0, max_j):           
            
            #Generate k
            node = (i, j)
            k = idx_map[node]
            
            #Generate all neighbours for this node
            neighbours = get_all_neighbours(node)
            
            #Filter out WRT Tresholds
            valid_neighbours = filter_out_invalid(M, neighbours, max_i, max_j)
            #print("Valid neighbours for NODE({}): {} \n".format(node, valid_neighbours))
            
            #Find Eulcidean distance FROM k to valid neighbours
            distance_to_neighbours = find_euclidean_dist(k, idx_map, M, valid_neighbours)
            
            #Store results
            dist_map[node] = distance_to_neighbours
           
    return dist_map
 
    
    
    
'''
This function generates all possible 8 neighbours for
    a given node
PARAMETERS:
    - Node TUPLE
RETURNS:
    - List of ALL nearest Neighbours
'''    
def get_all_neighbours(node):
    #Split
    i, j = node[0],node[1]
    
    all_nbrs = [(i-1, j-1), (i, j-1), (i+1, j-1), #1,2,3
                
                (i+1, j), (i-1, j), #4,8
                
                (i-1, j+1), (i, j+1), (i+1, j+1)] #5,6,7
    
    return all_nbrs




'''
This function simply filters out INVALID Neighbours

PARAMETERS:
    - Weights Matrix (M)
    - List of ALL NEIGHBOURS
    - i/j THRESHOLDS
RETURNS:
    - Filtered list of Valid neighbours

'''
def filter_out_invalid(M, neighbours, max_i, max_j):
    #Container for empty nodes
    ret_list = []
    
    #Analyze each neighbouring node
    for node in neighbours:        
        if(is_valid(M, node, max_i, max_j)):
            #Keep it
            ret_list.append(node)    
    return ret_list
            
        
        

        
'''
Determines whether a given node is VALID/INVALID
INVALID NEIGHBOUR-NODE indices are
        - Outside the Weight Matrix(ie. MAP)
        - NEGATIVE
        
PARAMETERS:
    - Weights Matrix
    - i/j THRESHOLDS
RETURNS:
    - TRUE if valid, FALSE otw
'''        
def is_valid(M, node, max_i, max_j):
    global map_dims
    
    #Split that node
    i, j = node[0], node[1]
    
#     #max_dim = 20 #ROUGH(20 for 400)
#     max_i, max_j = map_dims[0], map_dims[1]
    
    #print("Thresholds: \t i:{} \t j:{}".format(max_i, max_j))

    
    if(i < 0 or j < 0 or i >= max_i or j >= max_j):
        return False #Outside matrix
    else:
        return True
        
    

    
'''
This function iterates over nearest VALID neighbours

PARAMETERS:
    - k (CURSOR NODE)- where we are
    - IDX MAP
    - weights is the M(WEIGHTS) Matrix
    - List of Valid Neighbours
RETURNS:
    - Total Distance to all valid neighbours of given CURSOR NODE
'''
def find_euclidean_dist(k, idx_map,  weights, valid_neighbours):
    
    total_dist = 0.0
    #Find total distance to all neighbours
    for node in valid_neighbours:
        
        #Convert Neighbour to INDEX(kk)
        kk = idx_map[node]

        #CRUCIAL
        #print("k: {}".format(k))      
        #Reshape weights
        a = weights[k].reshape(1,-1)  #CURRENT CURSOR NODE
        b = weights[kk].reshape(1,-1) #Distance to VALID NEIGHBOUR      
        euclidean_distance = np.linalg.norm(a - b, axis = 1)
        
        #Add this 
        total_dist = total_dist + np.sum(euclidean_distance)

   
    return total_dist
    


# -----------------------------------------   Prior Knowledge ------------------------------


'''
This function generates the DIST_MAP (proximity to data points)

PARAMETERS:
    - som2
    -  Histovectors used to TRAIN this dataset

RETURNS:
    - proximity_map 
'''
def generate_distmap(som, X):

    y_pred = som.predict(X)
    proximity_map = som.map_dist(y_pred)
    print("Dist_map Shape: {}".format(proximity_map.shape))

    return proximity_map













   
'''
This function generates the Clusters based on the 
    Binning-algorithm using
    the Prior Knowledge about our dataset.

PARAMETERS:
    - som
    - Distance Map

RETURNS:
    - M (Ramped heatmatrix)
    - L (Labelled heatmatrix)
    - D (Density Matrix)
'''
def accumulate_bin_clusters(som, dist_map):
    # To accumulate clusters
    BINS = [x*1000 for x in range(0,(1+10),2)]  

    map_size2 = som.map_size

    #Setup arrays
    M = np.zeros(shape = (map_size2[0] * map_size2[1]))
    L = np.repeat('', map_size2[0] * map_size2[1])
    D = np.zeros(shape = (map_size2[0] * map_size2[1]))  #Densities


    #Iterate over entire SOM-2 map
    for bmu in range(dist_map.shape[1]):   
        distances = dist_map[:, bmu]
        #Minimum distance value
        min_dist = np.min(distances)
        #Specify indices of data points
        closest_idx = np.where(distances == min_dist)[0]
        BINS = [x*1000 for x in range(0, 11, 
                                        #Jump between classes
                                         2)]
        
        #Bin them
        counts, bin_edges = np.histogram(closest_idx, bins = BINS)

        #Which bin(ie.class-idx) has the MOST NEAR-SAMPLES?
        max_class_idx = np.argmax(counts)
        imputation_value = ramp_map[max_class_idx]
        M[bmu] = imputation_value
        L[bmu] = lookup_class[str(max_class_idx)]
        D[bmu] = len(closest_idx)  #Number of nearby samples
        
    #Normalize
    D = D/D.max()

    #Reshape matrices before plotting and returning
    M = M.reshape(map_size2)
    L = L.reshape(map_size2)
    D = D.reshape(map_size2)

    fig, (a1,a2) = plt.subplots(1,2,figsize = (20,8))
    sn.heatmap(M, annot = L, linewidths = 0, fmt ='', cmap='jet', ax = a1)
    sn.heatmap(D, annot = L, linewidths = 0, fmt ='', cmap='jet', ax = a2)
    plt.show()

    return M, L, D





#------------------------------------------------------- Full Predictions---------------------------------------------

'''
PARAMETERS:
    - Histovec
    - Cutoff Percentile
RETURNS:
    - IDX of Maximum Activation
'''
def find_maximum_activation_loc(histovec, cutoff = 95):
    
    #Find maximum activation
    percentile = np.percentile(histovec, cutoff)
    maximum_activation_idx = np.where(histovec > percentile)

    return maximum_activation_idx[0]




'''
This is essentially the main prediction function which utilizes both
    of the trained models.

Function to Predict:
    - spot on SOM-2 for given Bin of Images
    - spots of MAXIMUM ACTIVATION ON SOM-2
    
PARAMETERS:
    - BIN of FLATTENED 28x28 images
    - SOMs
    - pre-fitted MMS
    - GRID COORDINATES LIST for SOM-1
    - MAPS for SOM-1 

RETURNS:    
    - Activated spots on SOM-1
    - SPOT on SOM-2
'''
def compute_prediction_stats(x, som1, som2,
                            mms,
                        som1_grid_coords,
                        idx_map, node_map):

    #Predicted locations on SOM-1 for all images in the bin
    yhat_som1 = som1.predict(x)
    #----Accumulate counts across the map using SOM-1 Traversal
    count_vec = stan.get_bmu_counts(yhat_som1, 
                                    som1_grid_coords,
                                    som1.map_size, 
                                    #SOM-1 maps
                                    node_map, idx_map)
    print("Accumulated Counts:{}... \t SHAPE:{}".format(count_vec[:8],count_vec.shape))
    
    #Find most-activate spots on SOM-1
    k1_spots = list(find_maximum_activation_loc(count_vec, cutoff = 99.0))
    print("Most-activated spots on SOM-1 : {}".format(k1_spots))
    
    #---- Pass through trained SOM-2

    #Normalize

    mms = MinMaxScaler(feature_range=(0,1))
    x_vec = mms.fit_transform(count_vec.reshape(-1, 1))
    #x_vec = mms.transform(count_vec.reshape(-1, 625))


    #Make prediction
    k2 = som2.predict(x_vec.reshape(1, -1))[0]
    print("Predicted SPOT on SOM-2: {}".format(k2))
    
    return k1_spots, k2

  



'''
This function PLOTS the predicted spot on SOM-2

PARAMETERS:
    - Heatmap matrix
    - Labels matrix
    - Predicted spot on SOM-2
    - SOM-2

RETURNS:
    - 
'''
def plot_prediction_spot(M, L, k_spot, som):

    plt.figure(figsize =(10,8))

    M_prime = M.flatten().copy()
    L_prime = L.flatten().copy()

    #Highlight predicted spot
    M_prime[k_spot] = 1.5 * M_prime[k_spot]
    L_prime[k_spot] = 'X'


    sn.heatmap(M_prime.reshape(som.map_size), annot = L_prime.reshape(som.map_size), 
                                        linewidths = 0, fmt ='', cmap='jet')
    plt.show()

    return







'''
This function PLOTS the Decoded Prototypes from SOM-1

PARAMETERS:
    - Decoded Prototypes
    - HIgh Activation IDX
RETURNS:
    - 
'''
def plot_similar_dps(decoded_prototypes, high_idx):
    img_size = 32
    max_similar = 4
    
    #If you don't have enough...
    if(len(high_idx) < max_similar):
        max_similar = len(high_idx)
        
    #Find {MAX-SIMILAR} appropriate DPs
    activated_dps = decoded_prototypes[high_idx][:max_similar]
    
    #Setup plot
    fig, ax = plt.subplots(1, max_similar, figsize = (10, 10), squeeze = False)
    #Iterate over each 
    for k, dp in enumerate(activated_dps):
        #Extract DECODED SOM-1 Prototype
        image = dp.reshape(img_size, img_size)
        ax[0][k].imshow(image, cmap='inferno')
        ax[0][k].axis('off')
        
    plt.show()



'''
Plots Histovector of SOM-1 accumulated counts

PARAMETERS:
    - Histovector of counts across SOM-1
'''
def plot_histovector(count_vec):
    plt.figure(figsize =(25,5))
    plt.plot(count_vec)

    plt.ylabel('Counts')
    plt.xlabel('SOM-1 BMU')

    plt.show()









