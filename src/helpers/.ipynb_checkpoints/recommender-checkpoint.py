
import os
import src.helpers.histo_som as histosom
import src.helpers.desom_model as dm
import src.helpers.stat_analysis as stan
from sklearn.preprocessing import MinMaxScaler




class Recommender(object):
    """A final full model with the following components

    Attributes:
        models_loaded :  Boolean Flag: TRUE if all 3 Deep Models have been loaded
        input_dims:  Dimensions coming in from SOM-1 Vectors (eg. 625)
        latent_dims :   z, ie. the compression level
        mms:            MinMaxScaler (trained on training set)
        M:              Heatmap Matrix
    labels:             Cluster labels
        
        grid_coordinates,
        node_map, 
        idx_map:       SOM-1 utilities
       
    """
    
    def __init__(self, input_dims,
                      mms, latent_dims,
                      M, labels,
                      #SOM-1 grid utilities
                     grid_coordinates,node_map, idx_map):
        self.som1 = None
        self.som2 = None
        self.deep_AE = None
        #FLAG
        self.models_loaded = False

        #For quick 
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.mms = mms
        self.M = M
        self.labels = labels


        self.grid_coordinates = grid_coordinates
        self.node_map = node_map
        self.idx_map = idx_map
        
        
        
    '''
    Retrieve M, labels
    '''
    def get_labelled_heatmap(self):
        
        return self.M, self.labels
    
    
    
    
    '''
    Initialize ALL MODELS
    '''
    
    def initialize_models(self,
                         current_path = os.getcwd()):
        
        print("W.D. for Loading Models is: {}".format(current_path))
        
        #----Saved models
        #SOM-1
        pretrained_autoencoder = os.path.join(current_path,'data/som1/ae_weights-epoch100.h5')
        pretrained_model = os.path.join(current_path,'data/som1/DESOM_model_final.h5')
        
        #Fully-connected Deep Autoencoder for data compression
        pretrained_deepAE = os.path.join(current_path,'data/deepAE/ae_weights-epoch100.h5')
        
        #Trained MiniSOM (SOM-2)
        pretrained_som2 = os.path.join(current_path,'data/som2/practice/som.p')
        
        
        #Load models
        self.som1 = dm.load_som1(pretrained_autoencoder,  pretrained_model)
        
        #Load the fully-connected AE
        self.deep_AE = histosom.load_trained_AE(self.input_dims, 
                                                self.latent_dims,
                                                pretrained_deepAE)
        
        #Decent pre-trained SOM-2
        self.som2 = histosom.load_minisom(pretrained_som2)
        print("Matrix stats: \n")
        print("M shape: {} \n".format(self.M.shape))
        print("L shape: {} \n".format(self.labels.shape))

        if(self.som1 and self.deep_AE and self.som2):
            print("Loaded all 3 successfully")
            self.models_loaded = True
            return
        else:
            print("ERROR")
            return
     
         
    '''
    Main Prediction function 
    
    PARAMETERS:
        - Observation ID
        - #CCDs to use
    RETURNS: 
        - Predicted label
        - Winning Node
        -M/L for plotting
    '''           
    def predict(self, obs_ID, num_CCDs):
        #Path?
        current_path = os.getcwd()
 
        print("Starting")
        _, ccd_counts, _ = stan.summarize_file(obs_ID,
                                           #Use instance variables
                                           self.som1,
                                           self.grid_coordinates,
                                           self.node_map,
                                           self.idx_map,
                                           
                                           num_CCDs,
                                           visualize = False)
         
        print("CCD Counts(BEFORE MMS) : \n {}".format(ccd_counts))

        
         #Preprocess
        print("Encode using pretrained Deep-AE ==> Normalize ==> Find winner")
        
        #--Transform and print--
        encoded = self.deep_AE.encode(ccd_counts.reshape(1,-1))
        print("Encoded representation: \n {}".format(encoded)) 
        print("Encoded SHAPE: {} \t  RESHAPED: {}".format(encoded.shape,encoded.reshape(-1,1).shape)) 

        
        #----------------------------------Normalization-------------------
        #x = self.mms.transform(encoded)  #Weird Bug?
        
        #--Option 2: FRESH MinMaxScaler on RESHAPED
        self.mms = MinMaxScaler(feature_range = (0,1))
        x  = self.mms.fit_transform(encoded.reshape(-1,1)) #With reshaped!(AVOID ZEROS)
        print("MMS-Transformed: \n {}".format(x))
       
    
        print("Reshape before passing into MiniSOM...")
        x = x.reshape(1,-1)
        print("SHAPE : \n {}".format(x.shape))

            
        #Copy
        L = self.labels.copy()
        M = self.M.copy() 
        
        print("Matrix stats: \n")
        print("M shape: {} \n".format(M.shape))
        print("L shape: {} \n".format(L.shape))

        
        print("Predicting winner...")
        winning_node = self.som2.winner(x)
        print("Reversing node...")
        winning_node = (winning_node[1],winning_node[0])
        
        print(winning_node)
        
        print("Finding label from labels MATRIX...")
        #Find predicted label
        y_pred = self.labels[winning_node]
        
        if(y_pred == ''):
            print("Confusion!")
        else:
            print("Predicted label: {}".format(y_pred))
        
        #Highlight winner
        L[winning_node] = 'X'
        M[winning_node] = M[winning_node]* 2.9  #Ramp up
        

        #histosom.plot_heatmap(M, L, fname, 0.0, img_savepath)
        histosom.save_flask_map(M, L, obs_ID)
       
        
        return y_pred, winning_node, M, L
    
    
   
 
         
         
        
        
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        