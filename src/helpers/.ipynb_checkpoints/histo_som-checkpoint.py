#---------------------------------------------Module for SOM-2 and annalyzing results
import os
from glob import glob
import pickle
import io


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from random import sample
from timeit import default_timer as timer

from minisom import MiniSom
from src.DESOM.DESOM import DESOM  #Constructor


from src.helpers import preprocessor as prep
from src.helpers import stat_analysis as stan

from sklearn.preprocessing import MinMaxScaler


#---------------------------------------------------PATHS------------------------------------------
current_path = os.getcwd()
img_savepath = os.path.join(current_path,'data/plots/')


#Saving Recommender
rec_savepath = os.path.join(current_path,'data/recommender/')  #Recommender objects




#------------------------------------------CHANGE!!!------------------
#Specify where AEs should be saved
ae_dir = os.path.join(current_path,'results/tmp/different_SOM2s/deep_AE/')
#For training procedure
ae_dir_train = os.path.join(current_path,'results/tmp/different_SOM2s/deep_AE/')
#Specify where SOM-2's should be saved
som2_dir = os.path.join(current_path,'results/tmp/different_SOM2s/som2_0/')
#Load table from...
id_names = os.path.join(current_path, 'CADC_Images/result_v2.txt')




#Dict for fast lookup
lookup_class  = {"0":"t", "1":"g", 
                 "2":"r" , "3":"f" , "4":"b"}



#Saving and loading Recommender
def save_rec(obj, name):
    with open(rec_savepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_rec(name):
    with open(rec_savepath  + name + '.pkl', 'rb') as f:
        return pickle.load(f)

    
#For app loading
def load_rec_app(name):  
    
    rec_savepath = "C:\\Users\\Ahnaf Tazwar Ayub\\Documents\\NRC-Work\\project\\cadc-im\\data\\recommender\\"  
    with open(rec_savepath  + name + '.pkl', 'rb') as f:
        return pickle.load(f)


'''
Data Preprocessing Function

Must normalize BY COLUMN(ie.by BMU)
'''
def normalize_by_bmu(X):  
    for col in range(X.shape[1]):
        #Initialize scaler
        mms = MinMaxScaler()        
        #Normalize column
        X[:,col] = mms.fit_transform(X[:,col].reshape(1,-1))
        
        
'''
Fits a MMS to the matrix and transfrms it and returns the mms


PARAMETERS:
    - Matrix of Histogram Count Vectors
RETURNS:
    - X(Transformed)
    - MinMaxScaler(fitted to data)
'''
def mms_preprocess(X):
    #Initialize MMS
    mms = MinMaxScaler(feature_range=(0, 1))  
    #Preprocess
    X = mms.fit_transform(X)
    
    return X, mms
    
    
    
    
'''
Initializes a MiniSom model and trains it with the data

PARAMETERS:
    - data matrix
    - x/y -- map dimensions
    - learning rate
    - number of  training iterations
RETURNS:
    - trained SOM
'''
def train_minisom(X,
                  x,y, #Map dimensions
                 sigma = 3.2, 
                 learning_rate = 0.01,
                 num_iterations = 10000):  
    
    #Initialize SOM
    som = MiniSom(x, y,
                  #Number of features
                  X.shape[1],
                  sigma, 
                  learning_rate)

    #Initialize weights
    som.pca_weights_init(X)
    #Train the SOM Randomly
    som.train_random(X, num_iterations, verbose = True)
    
    
    return som




'''
Saves a trained SOM-2 with the appropriate name

PARAMETERS:
    - Trained SOM-2
    - Path directory
RETURNS:
    - 
'''
def save_minisom(som, save_dir):  
    
    #print("Model name: {}".format(model_name))
    
    #Specify where to save
    save_to_file = save_dir + 'som.p'   
    
    print("Saving trained SOM-2 at {}.".format(save_to_file))    
    #Save
    with open(save_to_file, 'wb') as outfile:
        pickle.dump(som, outfile)
        
    print("Model Saved!")
    
    return



'''
loads a trained SOM-2 from the provided location

PARAMETERS:
    - Directory of saved SOM-2 model  
RETURNS:
    - loaded SOM-2
'''
def load_minisom(trained_som_dir):  
    
    print("Loading trained SOM-2 from {}.".format(trained_som_dir))    
    with open(trained_som_dir, 'rb') as infile:
        som = pickle.load(infile)
    
    return som






#----------------------------------------------------- DESOM AutoEncoder----------------------------


'''
AutoEncoder for Dimensionality Reduction
- Define architecture (z - latent dims)
- use SAME Map Size as MiniSom


PARAMETERS:
    - input_dims: 625 for SOM-1 (25x25)
    - z
RETURNS:
    - trained fully-connected AE (compiled)
'''
def compile_AE(input_dims, z):
 
    #------------------------------------AE Architecture--------------------------
    #Latent dimensions
    print("Latent dimensions (z): {}".format(z))

    map_size = (20,20)
    print("Map Size(SOM Layer): {}".format(map_size))


    #625 dimensions coming in
    desom = DESOM(encoder_dims= [input_dims,
                                 500, 500, 
                                 300, 
                                 z],
                        ae_type = 'fc',
                        map_size = map_size )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    print(desom.model.summary())
    
    return desom





'''
AutoEncoder for Dimensionality Reduction
- Define architecture (z - latent dims)
- use SAME Map Size as MiniSom


PARAMETERS:
    - X (full-dimensional dataset)--must be Normalized
    - z
    - #Epochs for training
    - Directory to save weights
RETURNS:
    - trained fully-connected AE
'''
def pretrain_AE(z, 
               epochs, 
               save_dir):
    
    #Use helper
    desom = compile_AE(input_dims, z)      
    #----Pretrain--------
    desom.pretrain(X, 
                     optimizer='adam',
                     #Epochs
                     epochs = epochs,
                     batch_size = 256,
                     save_dir = save_dir)
    
    return desom



'''
AutoEncoder for Dimensionality Reduction
- Define architecture (z - latent dims)
- use SAME Map Size as MiniSom


PARAMETERS:
    - z
    - Path to pretrained AE (pretrained_autoencoder)
RETURNS:
    - LOADED trained fully-connected AE
'''
def load_trained_AE(input_dims, z, 
               pretrained_autoencoder):
 
    #Use helper
    desom = compile_AE(input_dims, z)
    
    #Load from appropriate location
    print("Loading trained AE from {}".format(pretrained_autoencoder))
    desom.load_ae_weights(pretrained_autoencoder)
    

    return desom



'''
Compresses data(X), NORMALIZES and returns Xz,mms


PARAMETERS:
    - Pretrained Deep AE
    - X (full-dimensional dataset) 
RETURNS:
    - Compressed dataset
    - MMS (Fitted)
'''
def compress_data(desom, X):
    
    print("Compressing data... \n")
    #Compressed version
    X_z = desom.encode(X)
 
    print("Normalizing data (Dimensions : {})... \n".format(X_z.shape[1]))
    #Normalize
    X_z, mms = mms_preprocess(X_z)
    
    print("X_z MIN: {} \n".format(np.min(X_z)))
    print("X_z MAX: {} \n".format(np.max(X_z)))

    return X_z, mms












'''
Plot SOM-2 using Seaborn

PARAMETERS:
    -trained SOM

PLOTS the resulting SOM
'''
def plot_minisom(som,
                img_savepath = img_savepath): #Default
    
    #------------------------PLOT-----------------
    plt.figure(figsize= (8,8))  
    
     #Extract distance map
    M = som.distance_map().T
    ax = sn.heatmap(M, linewidths = 0.01)
    
    #Correct structure
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
   
    
    #---------Filename
    filename = 'miniSOM.png'
    im_savepath = os.path.join(img_savepath,filename)
    plt.savefig(im_savepath)



'''
TODO: Improve splitting of classes. Current version is hard-coded 
        for 10,000 rows with even 20% split among each

Highlights unique clusters for given SOM and data 

PARAMETERS:
    - X
    -trained SOM
    - img_savepath (Folder to save plots)

RETURNS:
    - M (essentially the matrix which contains the ramped-up class cells)
    - labels(appropriately labelled cells on SOM-2)
    - df (key: class label --> value: imputation val)
    
'''
def highlight_classes(X, som,
                      img_savepath = img_savepath):
    
    #Key: IMPUTATION VALUE, VALUE: Label
    df = {}
    M = som.distance_map().T


    #Setup string labels
    total_units = M.shape[0] * M.shape[1]
    labels = np.repeat('',total_units).reshape(M.shape[0], M.shape[1])

    #Extract classes(VERY HARD-CODED)
    lower_bounds = [0,2,4,6,8]
    upper_bounds = [2,4,6,8,10]
    
    
    for idx, (lb,ub) in enumerate(zip(lower_bounds, upper_bounds)):    
        #Setup lower and upper bounds for proper class segments
        low = lb *1000
        high = ub *1000

        for i in range(low,high):

            #Extract winning node
            winning_node = som.winner(X[i])
            #Highlight winner
            #REVERSE!
            y = winning_node[0]
            x = winning_node[1]
            
            #print(x,y)

            #Imputation value
            imp_value = (low+high)/20000

            #Store the approrpriate imputation value for this class
            df[idx] = imp_value   

            M[x,y] = imp_value #Ramp up
            labels[x,y] = lookup_class[str(idx)] #Label with class

    
    #Call helper to plot
    plot_heatmap(M, labels,  'classes', 0.0,  img_savepath)
    
    return M, labels, df




'''
Plot Summary,ie. MiniSOM and Labelled classes
Very useful for Quality assessment

PARAMETERS:
    - trained SOM
    - training set (X_z)
    - Color MAP
    
PLOTS Result
'''
def summarize_minisom(som, X, cmap="cubehelix"):
    #Key: IMPUTATION VALUE, VALUE: Label
    df = {}
    M = som.distance_map().T
    #Original MiniSOM
    M_hmap = M.copy()


    #Setup string labels
    total_units = M.shape[0] * M.shape[1]
    labels = np.repeat('',total_units).reshape(M.shape[0], M.shape[1])

    
    #Extract classes(VERY HARD-CODED)
    lower_bounds = [0,2,4,6,8]
    upper_bounds = [2,4,6,8,10]
    
    #Use PRIOR knowledge
    for idx, (lb,ub) in enumerate(zip(lower_bounds, upper_bounds)):    
        #Setup lower and upper bounds for proper class segments
        low = lb *1000
        high = ub *1000

        #Label according to PRIOR knowledge
        for i in range(low,high):
            #Extract winning node
            winning_node = som.winner(X[i])
            #Extract x,y
            #print("Reverse x,y!!!")

            #REVERSE!
            y = winning_node[0]
            x = winning_node[1]
            #print(x,y)
            

            #Imputation value
            imp_value = (low+high)/20000

            #Store the approrpriate imputation value for this class
            df[idx] = imp_value   

            M[x,y] = imp_value #Ramp up
            labels[x,y] = lookup_class[str(idx)] #Label with class

    
    #PLOT
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (20,8))  
    
    #Plot original MiniSOM with CMAP
    sn.heatmap(M_hmap, 0.01, ax = ax1, cmap = cmap)
    #Correct structure
    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    
    #Plot labelled classes
    sn.heatmap(M, 0.0, annot = labels, fmt = '', ax = ax2, cmap = cmap)
    #Correct structure
    bottom, top = ax2.get_ylim()
    ax2.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.show()
    
    return M, labels #Just plot
   



'''
Highlights 1 class for given SOM and data 

PARAMETERS:
    - class label(to highlight)
    - M (ramped-up heatmap matrix)
    - df (key: class label --> value: imputation val)

PLOTS 1 class
'''
def plot_highlighted_class(class_label, M, df, img_savepath = img_savepath):
    
    print("Highlighting BMUs for class :{}".format(class_label))
     
    #Setup string labels
    total_units = M.shape[0] * M.shape[1]
    labels = np.repeat('',total_units).reshape(M.shape[0], M.shape[1])

    #Setup placeholder
    A = np.zeros([M.shape[0], M.shape[1]])

    #Find class values and ramp up
    A[M == df[class_label]] = 1
    labels[M == df[class_label]] = lookup_class[str(class_label)]


    #Call helper to plot
    fname = 'class_' + str(class_label)
    plot_heatmap(A, 
                 labels, fname, 0.0, img_savepath)
    
    return



    
'''
Helper to plot heatmap of given matrix

PARAMETERS:
    -matrix shaped like Map
    - labels
    -filename
    -linewidth
    - filepath(Folder to save plots)

'''
def plot_heatmap(M, labels,  fname , lwd,
                filepath):
    
    #------------------------PLOT-----------------
    plt.figure(figsize= (8,8))  
    
    #Plot
    ax = sn.heatmap(M, lwd, annot = labels, fmt = '')
    
    #Correct structure
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
   
    
    #---------Filename
    filename = fname + '.png'
    im_savepath = os.path.join(filepath,filename)
    plt.savefig(im_savepath)
    
    return




'''
Plot where validation ID lands on UNLABELLED SOM-2 
                as well as labelled SOM-2 for reference

PARAMETERS:
    - M_initial and labels_initial (unramped up)
    - M, L (UN-labelled and basically empty)
    - winning_node
    - Name to save plot as
    
PLOTS Result
'''
def highlight_validation(M_initial, labels_initial,
                         M, L, 
                         plot_name,
                         cmap="jet"):
    
    #PLOT
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (20,8))  
    
    #Plot original MiniSOM with CMAP and NO LABELS
    #Highlight on EMPTY
    
    sn.heatmap(M_initial, 0.01, annot = labels_initial,   fmt = '', ax = ax1, cmap = cmap)
    #Correct structure
    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    
    #Plot labelled classes
    sn.heatmap(M, 0.0, annot = L, fmt = '', ax = ax2, cmap = cmap)
    #Correct structure
    bottom, top = ax2.get_ylim()
    ax2.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.savefig(plot_name)   
    plt.show()
    
    return 


'''
Helper to SAVE Flask Map to static/plots

PARAMETERS:
    - matrix shaped like Map
    - labels
    - obs ID
    
RETURN
    - saved directory

'''
def save_flask_map(M, labels, obs_ID):
    
    #------------------------PLOT-----------------
    plt.figure(figsize= (8,8))  
    
    #Plot
    ax = sn.heatmap(M, 0.0, annot = labels, fmt = '')
    
    #Correct structure
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
   
    
    #---------Filename
    filename = str(obs_ID) + '.png'
    filepath = os.path.join(os.getcwd(),'static/plots/')
    im_savepath = os.path.join(filepath,filename)
    plt.savefig(im_savepath)
    
    return






    
'''
Downloads file and Runs Source Detection-->Passes through SOM-1-->preprocesses
                            -->passes to SOM-2 --> Highlights PREDICTED BMU on SOM-2

PARAMETERS:
    -obs_ID
    -#CCDs to randomly sample
    -SOM 1
    -SOM 2
    -M (ramped-up class matrix for SOM-2)
    -labels (matrix with class labels)
    -MMS (has already been fit to dataset)
    
    -SOM1 grid components
    
    - OPTIONAL Visualization


PLOTS 

RETURNS:
    - Predicted Cluster Label
'''    
def highlight_ID_SOM2(obs_ID, num_CCDs,
                      #Models
                      som1, som2, deep_AE,
                      #MinMaxScaler
                      mms,
                      M, labels,
                      #SOM-1 grid utilities
                     grid_coordinates,node_map, idx_map,
                      #Where to save plot
                     img_savepath = img_savepath,
                     visualize = False):
    
    
    _, ccd_counts, _ = stan.summarize_file(obs_ID,
                                           som1,
                                           grid_coordinates,
                                           node_map, idx_map,
                                           
                                           num_CCDs,
                                          visualize)
    
    #Preprocess
    print("Encode using pretrained Deep-AE ==> Normalize ==> Find winner")
    x = mms.transform(deep_AE.encode(ccd_counts.reshape(1,-1)))
    
    print("Predicting winner on SOM-2...")
    winning_node = som2.winner(x)
    
    #Highlight winner
    #REVERSE!
    y = winning_node[0]
    x = winning_node[1]
    
    #Find predicted label
    y_pred = labels[x,y]
    
    if(y_pred == ''):
        print("Confusion!")
    else:
        print("Predicted label: {}".format(y_pred))
    
    #Make copies and alter them
    labels = labels.copy()
    M = M.copy()
    
    
    labels[x,y] = 'X'
    M[x,y] = M[x,y]* 2.9  #Ramp up
    
    #Call helper to plot
    fname = 'highlighted_' + str(obs_ID)
    plot_heatmap(M, 
                 labels, fname, 0.0, img_savepath)
    
    return y_pred

  
    
    
    
#--------------------------------------------  Sampling IDs--------------------------------------------------    
    
    
    
'''
Reads in table of ID's and returns it as a DataFrame

PARAMETERS:
        - Directory where IDs are stored as text file
RETURNS:
        - Fully loaded DataFrame
'''
def get_dataframe():
  
    #Read using pandas
    table = pd.read_csv(id_names, sep=" ", header=None)
    table.drop(table.columns[[0,8]], axis=1, inplace = True)
    table.columns = ["ID","good", "rbt", "bt", "ts" ,"bgf" ,"dead_CCDS"]
    
    #Cast IDs appropriately
    table.ID = table.ID.astype(int)

    print(table.head())
    print("{} rows in full DataFrame.".format(table.shape[0]))
    
    return table




'''
Reads PREVIOUSLY SAVED SAMPLED DF

PARAMETERS:
        -
RETURNS:
        - Fully loaded SAMPLED DataFrame
'''
def load_sampled_df():
    
    sampled_df_loc = os.path.join(current_path, 'CADC_Images/sample.csv')
    print("Loading sampled data from {}".format(sampled_df_loc))
  
    #Read using pandas
    table = pd.read_csv(sampled_df_loc, index_col ='ID')
    
    print(table.head())
    print("{} rows in full DataFrame.".format(table.shape[0]))
    
    return table


    
'''
Reads in table of ID's and samples according to conditions

PARAMETERS:
        - Full DataFrame
        - Number of samples per condition
RETURNS:
        - Sampled DataFrame

'''
def get_sample(table, num_samples):
    
    #Initialize return dataframe
    ret_df = pd.DataFrame(columns = list(table.columns))
   
    #Good variants
    cond1 = np.logical_and(table.good > 0.8,table.good < 0.9)
    cond2 = np.logical_and(table.good > 0.7,table.good < 0.8)
    cond3 = np.logical_and(table.good > 0.6,table.good < 0.7)
    cond4 = np.logical_and(table.good > 0.1,table.good < 0.2)
    
    #Other classes
    rbt_max = np.logical_and(table.rbt > 0.8,table.rbt < 0.9)
    bt_max = np.logical_and(table.bt > 0.8,table.bt < 0.9)
    ts_max = np.logical_and(table.ts > 0.8,table.ts < 0.9)
    bgf_max = np.logical_and(table.bgf > 0.8,table.bgf < 0.9)

    
    conditions = [cond1, cond2, cond3, cond4,
                 rbt_max, bt_max, ts_max , bgf_max]
    
    for cond in conditions:
        sub_df = table.loc[cond]
        print("#Rows satisfying criteria: {}".format(sub_df.shape[0]))

        #Sample a row
        sampled_obs = sub_df.sample(n = num_samples)
        #Extract fields
        print("Observation: {} ".format(sampled_obs))
        sampled_id = sampled_obs.ID.astype(int)
        print("Sampled IDs: {} \n".format(sampled_id))

        #Append to DF
        ret_df = ret_df.append(sampled_obs, ignore_index=True)
        
        
    #Format dataframe
    ret_df.set_index('ID', inplace = True)
    #Apply dict mapping
    true_labels = ret_df.idxmax(axis = 1).apply(map_class)
    #Create column
    ret_df['y_true'] = true_labels.astype(str)
        
    return ret_df




'''
Reads in table of ID's and samples according to conditions

PARAMETERS:
        - Full DataFrame
        - Class label [g, r,b, t, f]
RETURNS:
        - List of CONDs(rows satisfying conditions)

'''
def get_sampled_class(table, c_label):
    
    #Find appropriate Series
    if(c_label == "g"):
        table_series = table.good
    elif(c_label == "r"):
        table_series = table.rbt
    elif(c_label == "b"):
        table_series = table.bt   
    elif(c_label == "t"):
        table_series = table.ts
    elif(c_label == "f"):
        table_series = table.bgf
    else:
        table_series = None
        print("No such class!")
   
    #Find variations
    cond1 = np.logical_and(table_series > 0.9,table_series < 1.0)
    cond2 = np.logical_and(table_series > 0.8,table_series < 0.9)
    cond3 = np.logical_and(table_series > 0.7,table_series< 0.8)
    #Lower probabiities
    cond4 = np.logical_and(table_series > 0.6,table_series < 0.7)
    cond5 = np.logical_and(table_series> 0.1,table_series < 0.2)

    conditions = [cond1, cond2, cond3, cond4, cond5]
    
    return conditions
    


'''
Reads in table of ID's and samples according to conditions

PARAMETERS:
        - Full DataFrame
        - c_label is the specified class KEY [g, r, b, ..]
        - Number of samples per condition
RETURNS:
        - Sampled DataFrame with RBTs

'''
def get_specific_sample(table,c_label, num_samples):
    
    #Initialize return dataframe
    ret_df = pd.DataFrame(columns = list(table.columns))
        
    conditions = get_sampled_class(table, c_label)
    
    for cond in conditions:
        sub_df = table.loc[cond]
        print("#Rows satisfying criteria: {}".format(sub_df.shape[0]))

        #Sample a row
        sampled_obs = sub_df.sample(n = num_samples)
        #Extract fields
        print("Observation: {} ".format(sampled_obs))
        sampled_id = sampled_obs.ID.astype(int)
        print("Sampled IDs: {} \n".format(sampled_id))

        #Append to DF
        ret_df = ret_df.append(sampled_obs, ignore_index=True)
        
        
    #Format dataframe
    ret_df.set_index('ID', inplace = True)
    #Apply dict mapping
    true_labels = ret_df.idxmax(axis = 1).apply(map_class)
    #Create column
    ret_df['y_true'] = true_labels.astype(str)
        
    return ret_df



'''
Reads in table of ID's and samples GOOD Observations according to conditions

PARAMETERS:
        - Full DataFrame
        - Number of samples per condition
        - lower threshold
        - upper threshold
RETURNS:
        - Sampled DataFrame
'''
def find_tricky_samples(table, num_samples, low, hi):
    #Initialize return dataframe
    ret_df = pd.DataFrame(columns = list(table.columns))
    
    #Good variants
    cond = (table['good'] > low) & (table['good'] < hi)
    
    #Boolean sampling
    sub_df = table.loc[cond]
    print("#Rows satisfying criteria: {}".format(sub_df.shape[0]))

    #Sample a row
    sampled_obs = sub_df.sample(n = num_samples)
    #Extract fields
    print("Observation: \n {} ".format(sampled_obs))
    sampled_id = sampled_obs.ID.astype(int)
    print("Sampled IDs: \n {} ".format(sampled_id))
    #Append to DF
    ret_df = ret_df.append(sampled_obs, ignore_index=True)
        
        
    #Format dataframe
    ret_df.set_index('ID', inplace = True)
    #Apply dict mapping
    true_labels = ret_df.idxmax(axis = 1).apply(map_class)
    #Create column
    ret_df['y_true'] = true_labels.astype(str)
    
    return ret_df





















   
    
'''
Mapping function to convert full class names to letter labels

PARAMETERS:
        - Full class name
RETURNS:
        - Single letter name
'''    
def map_class(class_name):
    #Map class to code
    class_dict = {'good':'g',
                  'bt':'b',
                  'rbt':'r',
                  'ts':'t',
                  'bgf':'f'}
    
    return class_dict[class_name]
    
    

    
    
'''
Prints rough accuracy measure
PARAMETERS:
        - DataFrame with columns: y_true, y_pred
RETURNS:
        - 
'''
def rough_accuracy(df):   
    print("Accuracy (Rough): {}".format(np.sum(df['y_true'] == df['y_pred'])/len(df)))    
    
    
    
    
    
    
    
    
    
    
    
    








