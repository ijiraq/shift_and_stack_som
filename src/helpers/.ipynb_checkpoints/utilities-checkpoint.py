# -*- coding: utf-8 -*-



from timeit import default_timer as timer
import os
import numpy as np
import matplotlib.pyplot as plt

#Random sampling
from random import sample


#-----------------------------Global Variables(used in plenty of places throughout code)
idx_map = {}
node_map = {}

#Maybe? distance_map = None



#--------------------------------------------Specify paths
current_path = os.getcwd()
img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/closest_samples/')





'''
Takes in Map size

RETURNS a list of cords for whole map
'''
def generate_list_of_coords(map_size): 
    #Find map size
    #map_size = desom.map_size
    
    coords = [] #List of Grid Coords
    for k in range(map_size[0] * map_size[1]):
        x = k // map_size[1]
        y = k % map_size[1]

        coords.append((x,y))
    return coords



'''
Requires Grid Coordinates(From generate_list_of_coords)
and map_size



Create a Mapping Dict for (Node => IDX)   CALLED idx_map
'''
def get_idx_map(grid_coordinates, map_size):
    #Refer to global
    global idx_map 
    #Populate
    for k in grid_coordinates:
            w = map_size[0] #Width of MAP
            #Convert Grid NODE to IDX
            arr_i = k[0] + w * k[1]

            #Initialize
            idx_map[k] = arr_i
    print("idx_map: maps from NODE to IDX")
    return idx_map
     
    
    
    
'''
REQUIRES map size


Creates and RETURNS a Mapping Dict for (IDX => Node)   CALLED node_map

also SETS global ref
'''
def get_node_map(map_size): 
    global node_map 
    for k in range(map_size[0] * map_size[1]): 
             #Convert to grid NODE
            x = k // map_size[1]
            y = k % map_size[1]
            #Form coordinate
            node = (x,y)
            #IDX -> Node
            node_map[k] = node
    print("node_map: maps from IDX to NODE")
    
    return node_map





    
'''
Requires Distance-Map Result

Higlight NODE for a given COORDINATE PAIR

#PASS IN : DECODED PROTOTYPES
            map_size

            NODE_MAP dict and IDX_MAP for fast lookup

'''
def highlight_node(grid_coords, map_size ,
                   decoded_prototypes,
                   idx_map, node_map):  
    
    #Get width
    w = map_size[0]
    
    #Array index
    #arr_i = grid_coords[0] + w * grid_coords[1]
    arr_i = idx_map[grid_coords]
    
    #---------------------Plot
    #Set up 32x32 size
    img_size = 32
    #Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))

    #Iterate over each and every prototype
    for k in range(map_size[0]*map_size[1]):
        
        #Extract coordinates
        coords = node_map[k]
        #Find coordinates
        x = coords[0]
        y = coords[1]
   
        ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap='gray')
        ax[x][y].axis('off')

        #Highlight the one we need
        if(k==arr_i):
            ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, 
                                                                img_size),
                                                         cmap='inferno')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    #---------Filename
    filename = 'highlighted_grid.png'
    #Slightly alter
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/')
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, 
                               filename)
    
    plt.savefig(im_savepath)
    print("Highlighted Grid saved as {} at {}".format(filename, im_savepath))
    
    
    
  


    
'''
Takes in BMU Coords (0-indexed)
and Distance Map
and MAP_SIZE for formatting
and X_train

Returns ----------------closest IMAGES (32x32 each)

'''
def find_closest_samples(grid_coords,
                         distance_map,
                         map_size,
                         X_train,
                        verbose = True):
    #Setup Map Size
    #map_size = desom.map_size
    
    #Get width
    w = map_size[0]
    
    #Array index(USE DICT LATER!)
    arr_i = grid_coords[0] + w * grid_coords[1]
    
    
    #Access
    A = distance_map[:, arr_i]

    #Indices of location with closest nodes
    closest_idx = np.asarray((np.where(A == np.min(A)))).flatten()
    
    #Collect samples from original data
    closest_samples = []
    for idx in closest_idx:
        #Extract sample from data and reshape
        closest_samples.append(X_train[idx].reshape(32,32))
        
    if(verbose):
        print("{} matching samples found.".format(len(closest_samples)))
    
    return closest_samples





'''
REQUIRES:  Samples with Minimum Manhattan Distance
         Coordinates (for filename)

Plot them

'''
    
def plot_closest_samples_and_save(closest_samples, coords):
    #How many nearby samples?
    num_samples = len(closest_samples)

    if(num_samples > 20):
        #Select only 20 randomly
        closest_img_list = sample(closest_samples, 20)
        num_samples = 20
    else:
        closest_img_list = closest_samples
    #Setup plot
    plt.figure(figsize=(20, 4))

    for i in range( 1, num_samples):
        ax = plt.subplot(1, num_samples, i)
        plt.imshow(closest_img_list[i].reshape(32,32), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #Filename
    filename = 'closest_'+ str(coords[0]) + '_' + str(coords[1]) +'.png'
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, filename)   
    plt.savefig(im_savepath)
    
    print("Nearest samples saved as {} at {}.".format(filename, im_savepath))




'''
Highlight BMU for a given idx-th element from dataset

Requires Distance-Map Result

# '''
# def highlight_bmu(dist_map,
#                   sample_idx,
#                   desom,  #trained desom
#                   X_train, #dataset
#                  decoded_prototypes): #pass in
    
#     #Extract approrpiate element of X_train
#     A = dist_map[sample_idx]
#     #Index on flat SOM -100
#     j = np.argmin(A)
    
#     #Plot
#     #Setup Map Size
#     map_size = desom.map_size

#     #Set up 32x32 size
#     img_size = int(np.sqrt(X_train.shape[1]))
#     #Setup plot
#     fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))
#     #Iterate over each and every prototype
#     for k in range(map_size[0]*map_size[1]):
#         ax[k // map_size[1]][k % map_size[1]].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap='gray')
#         ax[k // map_size[1]][k % map_size[1]].axis('off')

#         #Highlight the one we need
#         if(k==j):
#             ax[k // map_size[1]][k % map_size[1]].imshow(decoded_prototypes[k].reshape(img_size, img_size))
#     plt.subplots_adjust(hspace=0.05, wspace=0.05)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








































