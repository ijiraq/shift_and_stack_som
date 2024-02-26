#---WEEK 7
# Modules for loading numpy arrays from folders in order to prep for SOM-1 and SOM-2


import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from random import sample
from timeit import default_timer as timer



from src.helpers import stat_analysis as stan


#---------------------------------------------------PATHS------------------------------------------
current_path = os.getcwd()
save_path = os.path.join(current_path, 'data/new_data/')


#Filenames
X_tr_name = os.path.join(save_path, 'X_tr.npy')
X_val_name = os.path.join(save_path, 'X_val.npy')
y_tr_name = os.path.join(save_path, 'y_tr.npy')
y_val_name = os.path.join(save_path, 'y_val.npy')




#New data
saved_data = os.path.join(current_path, 'new_data/')
print("New data is being loaded from: {}".format(saved_data))






'''
Forms DICT and returns it

KEY: Class name
VALUE: Tuple (Image, class label)
'''

def create_file_dict():

    #Setup mapping dict
    labels_dict = {}

    filenames = glob(saved_data + '/' + '*.npy' )
    for idx, name in enumerate(filenames):
            #Store image data
            image_data = np.load(name)
            #Reshape
            image_data = image_data.reshape(image_data.shape[0],
                           np.power(image_data.shape[1],2))#Ensure each row is flattened
            #Setup dict
            name = name.split('/')[-1][:-len(".npy")]
            labels_dict[name] = (image_data, idx)

            print('ID:'+str(idx)+ ' \t Name of file: '+name+ '\n Shape: {}'.format(image_data.shape))
     
    
    return labels_dict
        
        
        
        
'''
Simply prints the k/v pairs of the dict
'''
def print_dict(labels_dict):        
    for k,v in labels_dict.items():
        print("File name: {} \t Data Shape: {} \t y_label: {}".format(k,v[0].shape,
                                                                        v[1]))
    

    
'''
Creates Training and Validation sets
PARAMETERS:
    - set_size -- TOTAL SIZE of Training Set/Val Set each
    - labels_dict  
RETURNS: 
    -X_tr, y_tr   X_val, y_val
    
EFFECTS:
    - SAVES datasets
'''
def create_datasets(labels_dict, set_size = 100000):

    #How many images to retain for each class?
    #num_ims = 20000 (default, essentially)    
    num_ims = set_size//5
       
    #Final size of training/val set each
    subset_size = (num_ims//2)*5

    print("Retain {} images from  2 Million Images".format(num_ims*5))
    print("Training and Validation will each contain {} images".format(subset_size))


    #initialization
    X_tr = np.zeros((subset_size,np.power(32,2)), dtype = np.float32)
    X_val = np.zeros((subset_size,np.power(32,2)), dtype = np.float32)


    y_tr = np.zeros(subset_size, dtype = int) #Just a vector 1D
    y_val = np.zeros(subset_size, dtype = int) #Just a vector 1D


    print("X_tr INITIAL shape: {}".format(X_tr.shape))
    print("X_val INITIALshape: {}".format(X_val.shape))
    print("y_tr INITIAL shape: {}".format(y_tr.shape))
    print("y_val INITIALshape: {}".format(y_val.shape))

    #Deal with each class
    for c_name,v in labels_dict.items():  
        #Deconstruct the tuple
        data = v[0]
        #Randomly shuffle rows
        np.random.shuffle(data)
        print("-------------------------Shuffled in-place-------------- \n")

        #--------Number of rows to retain from each class
        #num_ims = 200000 (Defined above)
        sample_size = num_ims//2
        #Extract first num_ims rows
        data = data[:num_ims]
        print("-------------------------Retained top {} rows-------------- \n".format(num_ims))


        #Create labels vector
        y_label  = np.repeat(v[1],#Label
                            sample_size) #Repeat for train/val size


        #Setup indices for mutation
        factor = data.shape[0]//2 #200K for half
        lower_row_lim = factor * v[1] #200K * lim
        upper_row_lim = factor * (v[1] + 1) #200K * lim

        #Mutate
        X_tr[lower_row_lim:upper_row_lim] = data[:sample_size]  #Top half of shuffled data
        X_val[lower_row_lim:upper_row_lim] = data[-sample_size:]  #Bottom half of shuffled data

        #Y data
        y_tr[lower_row_lim:upper_row_lim] = v[1]  #Set label
        y_val[lower_row_lim:upper_row_lim] = v[1] 


        print("---------------------------------Filled with data ---------------------- \n")

        print("View between idx {} and {}".format(lower_row_lim, upper_row_lim))
        print("ORIGINAL Data: \n{}".format(data))
        print("X_train: \n{}".format(X_tr[lower_row_lim:upper_row_lim]))
        print("X_val: \n{}".format(X_val[lower_row_lim:upper_row_lim]))

        print("y_tr:{} \t Shape:{} \n".format(y_tr[lower_row_lim:upper_row_lim],y_tr.shape))
        print("y_val:{} \t Shape:{}\n".format(y_val[lower_row_lim:upper_row_lim],y_val.shape))


        print("Just addded {}".format(c_name))
        print("X_train shape: {}".format(X_tr.shape))
        print("X_val shape: {}".format(X_val.shape))

        print("y_tr -labels: {}".format(np.unique(y_tr)))
        print("y_val -labels: {}".format(np.unique(y_val)))


        print("\n")
        
    #----Save datasets
    # Mind the order!
    save_datasets(X_tr, y_tr, X_val, y_val)
        
    #Return values
    return (X_tr, y_tr), (X_val, y_val)


'''
Saves datasets
'''
def save_datasets(X_tr, y_tr, X_val, y_val):
    
    #Declare Filenames
    X_tr_name = os.path.join(save_path, 'X_tr.npy')
    X_val_name = os.path.join(save_path, 'X_val.npy')
    y_tr_name = os.path.join(save_path, 'y_tr.npy')
    y_val_name = os.path.join(save_path, 'y_val.npy')
       
    #Save
    np.save(X_tr_name, X_tr)
    np.save(X_val_name, X_val)
    np.save(y_tr_name, y_tr)
    np.save(y_val_name, y_val)
    
    print("Saved datasets at {}...".format(save_path))
        
 
'''
Loads and returns validation datasets
'''
def load_val_sets():
    
    #Declare Filenames
    X_val_name = os.path.join(save_path, 'X_val.npy')
    y_val_name = os.path.join(save_path, 'y_val.npy')
       
    #Load
    X_val = np.load(X_val_name)
    y_val = np.load(y_val_name)
    
    print("Loaded datasets from {}...".format(save_path))
    
    return X_val, y_val
        



'''
FAKE ID Generation Algorithm

PARAMETERS:
    -num_fakes: Number fake_IDs (ie. fake BINS) per class
    - validation datasets (X,y)
RETURNS:
    - List of Fake Bins   
'''
def generate_fake_IDs(X_val, y_val,
                      num_fakes = 1000):
    
    #Collection of bins
    bins = []

    #how many fake_IDs per class?
    #num_fakes = 1000

    #How many images have we sampled so far?
    num_ims = 0

    start = timer()

    #Generate fake_IDs for each class
    for class_label in range(0,5):

        #Obtain images for a given class
        print("Sampling images from class {}".format(class_label))
        X_sub = X_val[np.where(y_val == class_label)]
        print("X-sub shape: {}".format(X_sub.shape))

        #Ranodmly sample n_images
        num_cutouts = [x*100 for x in range(1,5,
                                            1)]
        print("Possible number of cutouts: {}".format(num_cutouts))

        #For 100 iteration, repeat sampling and bin creation
        for i in range(num_fakes):
            #Randomly sample one number
            n0 = sample(num_cutouts, 1)[0]

            #Generate n0 random IDX for sampling from X_SUB
            print("Randomly selecting images from X_sub, with replacement...")
            sample_idx = np.random.randint(X_sub.shape[0], 
                                               size = n0)

            #Generate fake_ID
            fake_bin = X_sub[sample_idx, :]
            print("Dimensions of generated fake bin: {}".format(fake_bin.shape))

            #Update num
            num_ims += fake_bin.shape[0]

            #Add to collection
            bins.append(fake_bin)

            print("Number of fake_IDs: {} \n".format(len(bins)))

    end = timer() 
    #Log
    print("Time taken: {} seconds".format(np.around(end - start,2)))
    print("#Images across the 5 classes: {}".format(num_ims))  
    
    return bins

        

    
'''
PARAMETERS:
    -List of Bins (Fake IDs)
    - MORE...
RETURNS:
    - Matrix of dimensions:   (num_IDs ,(map_size[0]*map_size[1]))


Basically takes each Bin, finds a Vector of Counts for it,
        stacks on main_vector_collection
        
'''
def create_bin_counts_matrix(bins, desom,
                             grid_coordinates,
                           map_size, 
                           node_map, idx_map):
    
    #Create Vector of Correct Size
    predicted_bmu_counts = np.zeros(map_size[0]*map_size[1])
    
    #Iterate over each bin
    for X in bins:
        print("Loaded bin has shape: {}".format(X.shape))
        
        #Bin is an X-matrix that we can pass into our trained model
        #Use trained DESOM model (desom) to get Vector of Predictions
        #Predictions
        y_pred = desom.predict(X)

        #count_vec stores the counts for each BMU in range (0, n_nodes-1)
        count_vec = stan.get_bmu_counts(y_pred, grid_coordinates,
                                   map_size, 
                                   node_map, idx_map)
        
        #print("Formed vector of length: {}".format(count_vec.shape))
        #print(count_vec)
        
        #Stack
        predicted_bmu_counts = np.vstack([predicted_bmu_counts, count_vec])
        
    #Remove first row
    predicted_bmu_counts = predicted_bmu_counts[1:, :] 

    print("Final Matrix of Predicted counts has dimensions: {}".format(predicted_bmu_counts.shape))
    
    return predicted_bmu_counts

        
        
        
        
        
        
        
        
        
        








