'''
------------- Module for Variational Autoencoder-----------
'''
import os
from keras.layers import (Input, Dense, Lambda)
from keras.models import Model
from keras.objectives import binary_crossentropy
#For sampling
from keras import backend as K
from keras.models import load_model


current_path = os.getcwd()
default_savedir = os.path.join(current_path,'data/vae/')
 



class VAE(object):
    
    '''
    Defines Encoder, Decoder architecture and connects to form VAE
    PARAMETERS:
        - input_dims
        - inter_1,... (Intermediate dimensions like 300,100)
        - latent_dims
    
    '''
    def __init__(self, input_dims, inter_1, inter_2, inter_3,  latent_dims):
        #------------------------ Encoder -------------------
        #Store input data dimensions
        input_dims = (input_dims, )  #Tuple

        intermediate_dims = inter_1
        intermediate_dims2 = inter_2
        intermediate_dims3 = inter_3

        # Layers 
        x = Input(shape = input_dims, name ='input')
        #Intermediate layer(h)
        h = Dense(units = intermediate_dims, activation='relu', name ='h')(x)
        #Intermediate layer1(h2)
        h2 = Dense(units = intermediate_dims2, activation='relu', name ='h2')(h)
        #Intermediate layer1(h3)
        h3 = Dense(units = intermediate_dims3, activation='relu', name ='h3')(h2)


        #Latent layer
        z_mu = Dense(units = latent_dims, name ='Mu')(h3)
        z_log_sigma = Dense(units = latent_dims, name ='Sigma')(h3)


        '''
        args -- will be layers
        '''    
        def sampling_fn(args):
            #Unpack and store in local
            mu, sigma = args

            batch_size = K.shape(mu)[0]
            latent_dims = K.shape(mu)[1]
            epsilon = K.random_normal(shape=(batch_size, latent_dims))

            return mu + K.exp(sigma / 2) * epsilon


        #Use function with Lambda layer
        z = Lambda(sampling_fn, output_shape = (latent_dims, ), name = 'z')([z_mu, z_log_sigma])
        self.encoder = Model(x, [z_mu, z_log_sigma, z], name = 'Encoder')


        #---------------------- Decoder ---------------------
        d_i  = Input(shape = (latent_dims, ), name='decoder_input')

        decoder_h3 = Dense(units = intermediate_dims3, activation='relu', name = 'h_decoder3')
        decoder_h2 = Dense(units = intermediate_dims2, activation='relu', name = 'h_decoder2')
        decoder_h = Dense(units = intermediate_dims, activation='relu', name = 'h_decoder')      
        decoder_mean = Dense(units = input_dims[0], activation='sigmoid', name = 'decoder_output')
        

        #Decoder layers
        h_decoded3 = decoder_h3(d_i)    #Intermediate reconstruction
        h_decoded2 = decoder_h2(h_decoded3)    #Intermediate reconstruction
        h_decoded = decoder_h(h_decoded2)      #Intermediate reconstruction
        x_decoded = decoder_mean(h_decoded)   #Original reconstruction

        self.decoder = Model(d_i, x_decoded, name='Decoder')


        #End-to-end model
        vae_outputs = self.decoder(self.encoder(x)[2])
        self.vae  = Model(x, vae_outputs, name='vae')


        self.encoder.summary()
        print("\n")
        self.decoder.summary()
        print("\n")
        self.vae.summary()


        '''
        DEFINE WITHIN SCOPE
        KL + Reconstruction loss 
        '''
        def vae_loss(yTrue, yPred):
            #Reconstruction loss
            x_cent_loss = binary_crossentropy(x, vae_outputs)
            #KL Divergence
            kl_loss = - 0.0005 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma),
                                                                                    axis = -1)

            return x_cent_loss + kl_loss


        #Compile the end-to-end model
        self.vae.compile(optimizer='adam', loss = vae_loss)
        print("\n -------------------------Compiled VAE------------------------- \n")
        return
    
    
    
    '''
    RETURNS:
        - (Encoder, VAE)
    '''    
    def get_encoders(self): 
        return (self.encoder, self.vae)
    
    
    '''
    Prints Encoder summary
    '''    
    def summarize_encoder(self): 
        self.encoder.summary()
        return
            
    
    '''
    Pass in NORMALIZED Vector data
    
    PARAMETERS:
        - X
        - epochs
    FITS to Data and Trains
    '''
    def fit_and_train(self, X, epochs):
        self.vae.fit(X, X,
                    shuffle = True,
                    epochs = epochs)
        print("Training Complete")
        
        #Save Encoder
        self.save_encoder()
        
        
        
    '''
    Compresses given data down to latent_dimensions
    RETURNS Compressed data
    '''
    def compress(self, X):
        X_z = self.encoder.predict(X)[2]
        return X_z
    
    
    '''
    SAVES in h5 format
    Saves Encoder to specified file directory   
    PARAMS:
        -Filepath
    '''
    def save_encoder(self, filepath = default_savedir):
        #Save in correct format
        filename = os.path.join(filepath, "encoder.h5")
        self.encoder.save(filename)
        
    '''
    Loads Encoder from specified file directory   
    PARAMS:
        -Filepath
    '''
    def load_encoder(self, filepath = default_savedir):
      
        loaded_encoder = load_model(filepath)
        
        return loaded_encoder
        
        

                

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



























