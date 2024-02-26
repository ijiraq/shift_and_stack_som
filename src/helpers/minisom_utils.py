'''
This module contains the utilities for MiniSOM
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sn
import pickle

from helpers import deep_som as deepsom
from helpers import histo_som as histosom
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter







'''
This function loads a specified Lookup File

PARAMETERS:
    - Filename
RETURNS:
    - Loaded object
'''
def load_object(fname):
    with open(fname + '.pkl', 'rb') as f:
        val_lookup = pickle.load(f)
        return val_lookup




'''
This function normalizes the data as required...

PARAMETERS:
    - Data
RETURNS:
    - Normalized data
    - FITTED MMS
'''
def normalize_data(X, by_bmu=True):
    #Initialize MMS
    mms = MinMaxScaler(feature_range = (0,1))

    if(by_bmu):
        print("Normalizing by BMU...(ie.with transpose)")
        X_t = mms.fit_transform(X.T)
        X = X_t.T   #Transpose again
    else:
        print("Normalizing each sample")
        X = mms.fit_transform(X)
        
    print("Shape: {}".format(X.shape))
    return X, mms







'''
This function generates the conversion DICTS

PARAMETERS:
    - Data
RETURNS:
    - idx_map, node_map
    - MAP SIZE
'''
def generate_conversion_dicts(som):
    
    map_size = som._weights.shape[:2]
    print("Map-size: {}".format(map_size))

    idx_map, node_map = {}, {}

    # idx_map, node_map = deepsom.generate_conversion_dicts(map_size)
    for kk in range(map_size[0]*map_size[1]):
        node = np.unravel_index(kk, map_size)
        #Setup dicts
        idx_map[node] = kk
        node_map[kk] = node
        
    return idx_map, node_map, map_size







'''
This function loads the .npy file with string IDs and maps them 
to their corresponding pre-computed values

PARAMETERS:
    - Val lookup (DICT): OBS-ID =>[label, pr(good)]
RETURNS:
    - labels
    - probabilities
'''
def extract_priors(val_lookup, ids):
    
    #Extract only the labels
    tuple_arrs = np.vectorize(val_lookup.get)(ids)
    y = tuple_arrs[0]
    probabilities = tuple_arrs[1]
    
    return y, probabilities



'''
Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.

PARAMETERS:
    - data
    - Trained MiniSOM
RETURNS:
    - Matrix
'''
def euclidist_map(X, som):
    return som._distance_from_weights(X)







'''
** Nearest Neighbors Method**
Returns the LABELLED and RAMPED SOM

PARAMETERS:
    - OBS map
    - String-IDs array
    - Y
    - probabilities
    - map_size
    - N : Number of nearest samples to use in calculations
    -
RETURNS:
    - Dict mapping [BMU-IDX => STATS]
'''
def generate_node2stats(obs_map, map_size,
                        ids, y, probabilities,
                        N=10):
    
    ret_dict = {}

    #Iterate over map in usual way (left to right)
    for bmu in range(map_size[0]*map_size[1]):

        #N nearest IDs for ACCESSING
        min_idx = np.argsort(obs_map[:, bmu])[:N]

        #Observation IDs
        nearest_idx = ids[min_idx]
        #Euclidean distances from this BMU
        distances = obs_map[min_idx, bmu]
        
        #Expected labels and probabilities
        labels = y[min_idx]
        g_probs = probabilities[min_idx]
        
        #Find mean probability
        mean_prob = np.around(np.mean(g_probs), 3)
        #-------------- Binning-----------------
        expected_labels, counts = np.unique(labels, return_counts=True)
        #Get the highest value LOC-IDX(ie. the last of the sorted indices array)
        majority_voted_label = expected_labels[np.argsort(counts)[-1]]
        
        #Store (idx=> stats)
        ret_dict[bmu] = [nearest_idx, mean_prob, majority_voted_label]
        
    return ret_dict


        
        
        
'''
This function produces the RAMPED SOM

PARAMETERS:
    - node2stats dict
    - MAP SIZE
RETURNS:
    - Ramped up SOM
'''
def generate_ramped_SOM(node2stats, map_size):
    
    L = np.repeat('', map_size[0]*map_size[1])
    M = np.zeros(map_size[0]*map_size[1])


    for bmu in range(map_size[0]*map_size[1]):
        #---- Retrieve
        _, mean_prob, majority_voted_label = node2stats[bmu]
        
        #----------------- Impute
        M[bmu] = mean_prob
        L[bmu] = majority_voted_label


    #--------------- PLOT ----------------------   
#     plt.figure(figsize= (8,8))  
#      #Extract distance map
#     jet_reversed = cm.get_cmap('jet_r')
#     print("Plotting transpose...")
#     ax = sn.heatmap(M.reshape(map_size).T, annot=L.reshape(map_size).T, fmt='', linewidths = 0.0, cmap=jet_reversed)
#     #Correct structure
#     bottom, top = ax.get_ylim()
#     ax.set_ylim(bottom + 0.5, top - 0.5)

#     plt.show()
    plot_ramped_map(M, L, map_size)
    
    return M, L


'''
This function obtains the similar Observation IDs for a given node

PARAMETERS:
    - node2stats dict
    - node
    - IDX_MAP
RETURNS:
    - List of Observation IDs
'''
def get_similar_idx(node2stats, node, idx_map):
    
    #Convert to BMU(IDX)
    bmu = idx_map[node]
    #Lookup
    idx_list, _, _ = node2stats[bmu]
        
    return idx_list
    

    
    
#----------------------------------------------------------  WinMAP functionalities------------------------    

    
'''
Returns a dictionary where we map the FLAT_BMU-ID to a Counter mapping classes to their counts

PARAMETERS:
    - SOM
    - Data, corresponding LABELS
    - IDX_MAP
RETURNS:
    - MAP [BMU-flat => Counter]
'''
def get_class_map(som, data, labels, idx_map):
    
        #Check for proper ordering of labels
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
            
        #Initialize
        winmap = defaultdict(list)
        
        for xx, obs_idx in zip(data, labels):
            
            #Find winner for the data point in SOM system
            x, y = som.winner(xx)
            
            #Find corresponding point on FLAT
            kk = idx_map[(x,y)]
            
            node = (x,y)
            
            #Setup correct order for Node
            #    and store that NODE as a KEY
            winmap[node].append(obs_idx)
         
        #For each BMU(FLAT-idx), we have a list of Observation IDs that were mapped to it
        for node in winmap:
            #print("Winmap ele: {} \t Winmap VAL:{}".format(flat_bmu, winmap[flat_bmu]))
            winmap[node] = Counter(winmap[node])

        return winmap
    
    
    
    
    
    
'''
Returns a dictionary where we map the FLAT_BMU-ID to a Counter mapping classes to their counts

PARAMETERS:
    - SOM
    - Data, corresponding OBS-IDX
    - IDX_MAP
RETURNS:
    - MAP [BMU-flat => OBS-ID LIST]
'''    
def get_obs_idx_map(som, data, labels, idx_map):
        
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
            
        winmap = defaultdict(list)
        
        for xx, obs_idx in zip(data, labels):
            #Find winner for the data point
            x, y = som.winner(xx)
            node = (x,y)
            
            #Setup correct order for Node
            #    and store that NODE as a KEY
            kk = idx_map[node]
            
            winmap[node].append(obs_idx)

        return winmap
    
    

'''
Returns the ramped M-matrix, L-matrix and DICT

PARAMETERS:
    - map_size
    - CLASS_MAP [BMU NODE => Counter]
    - OBS_ID_MAP [BMU NODE => List of IDs]
    - Node MAP
    - Val Lookup
RETURNS:
    - MAP [BMU => Results]
    - M(ramped SOM)
    - L (labels)
'''        
def generate_bmu2results(map_size, class_map, obs_idmap, node_map, val_lookup):
    
    #Initialize
    L = np.repeat('', map_size[0]*map_size[1])
    M = np.zeros(map_size[0] * map_size[1])

    #Initialize
    bmu_map ={}

    #Iterate over flat
    for kk in range(map_size[0]*map_size[1]):

        #Lookup appropriate node
        bmu = node_map[kk]
        bmu_results = []

        #Check if we actually have the BMU in the map
        if bmu in class_map.keys():

            #Find the Counter
            class_counter = class_map[bmu]

            if len(class_counter) > 0:
                #Find majority vote class
                yhat = class_counter.most_common()[0][0]
                #Find a list of IDs nearest to this node
                nearest_idx = list(obs_idmap[bmu])

                #Collect probabilities
                idx_probabilities = []
                #Accumulate probability values for 
                for obs_id in nearest_idx:
                    #Unpack lookup ground truth values
                    y0, p0 = val_lookup[obs_id]
                    idx_probabilities.append(p0)
                    #Find the class and append as tuple
                    bmu_results.append((y0, obs_id))

                #Take mean to impute RAMPED MATRIX
                mean_prob = np.mean(idx_probabilities)

                M[kk] = mean_prob
                L[kk] = yhat
                #Store results for Grand DICT
                bmu_map[bmu] =  bmu_results
        else:
            #print("BMU not in keys!")
            M[kk] = 0.0
            #Store empty list
            bmu_map[bmu] =  []
            
            
    #Done
    return bmu_map, M, L





    
'''
Plots the labelled SOM

PARAMETERS:
    - M
    - L
    - map_size
RETURNS:
    -
''' 
def plot_ramped_map(M, L, map_size):
    
    plt.figure(figsize= (8,8)) 
    
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    jet_reversed = cm.get_cmap('jet_r')
    ax = sn.heatmap(M.reshape(map_size).T, annot=L.reshape(map_size).T, fmt='', linewidths = 0.0, cmap=jet_reversed)
    
    #Correct structure
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    return





