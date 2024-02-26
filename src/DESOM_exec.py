import os
import matplotlib.pyplot as plt
import numpy as np


from sklearn.preprocessing import StandardScaler
from DESOM import DESOM
from datasets import load_spectra_data

from helpers import utilities as uti
from helpers import desom_model as dm


#-------------------------------------------------- Paths -------------------------------------------------
root = os.getcwd()
#Model weights stored at
saved_weights = os.path.join(root, 'results/tmp/')

#Filenames
ae_path = os.path.join(saved_weights, 'ae_weights-epoch1.h5')
model_path = os.path.join(saved_weights, 'DESOM_model_final.h5')


# Load the Train SET(required to analyze nearest samples)
X = np.load("X_test.npy")
#X= [:,0:10000]



#---------------------------------------------- Model setup -------------------------------
#Load the DESOM model
som = dm.load_desom_trained((15,15), 'cnn2D', X.shape[1], ae_path, model_path)


#Generate components
map_size = som.map_size

#A list of NODES
nodes_list = uti.generate_list_of_coords(map_size)
#Disctionaries for FAST conversion
node2idx = uti.get_idx_map(nodes_list, map_size)
idx2node = uti.get_node_map(map_size)



#These are the BMU's you can see on the SOM...
decoded_prototypes = som.decode(som.prototypes)



#This map gives you the relation with the train set
#Please refer to the function for the logic.
dist_map = dm.get_distance_map(som, X)


#Nearest Samples
nearest_dict = {}
for bmu in range(dist_map.shape[1]):
    
    distances = dist_map[:, bmu]
    #Minimum distance value
    min_dist = np.min(distances)
    #Specify indices of data points
    closest_idx = np.where(distances == min_dist)[0]
    
    nearest_dict[bmu] = closest_idx
    
#nearest_dict



#Find nearest samples for node BMU
bmu = 0
nearest_samples = nearest_dict[bmu]
#Randomly sample 5 samples
sampled_idx = np.random.choice(nearest_samples, 5)

#Plot
fig, ax = plt.subplots(len(sampled_idx), 1, figsize = (10,10),  squeeze=False)

for i in range(len(sampled_idx)):
    ax[i][0].plot(X[sampled_idx[i]])
    ax[i][0].set_ylim(bottom=-10, top=10)
     
plt.savefig("nearest_samples_{}".format(bmu))





#-------------------------- Predictions ---------------


# If we want to analyze the first row in the dataset
#          and use the AE on it
x, x_rec = dm.reconstruct_sample_from_data(X, som, 0)
dm.plot_reconstruction(x, x_rec)

#Given a sample(x) from the dataset...
k, predicted_bmu = dm.predict_bmu(x, som, decoded_prototypes)
print("Predicted spot on SOM: {}".format(k))

#Plot the BMU
plt.plot(predicted_bmu)
plt.savefig("BMU.png")





