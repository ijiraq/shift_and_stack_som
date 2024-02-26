from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dense,BatchNormalization, LeakyReLU, ReLU
from keras.models import Model
from keras import backend as K
import logging


im_dim= 21
def cnn_2dae(encoder_dims, latent_dim=64):

    #input_layer = Input(shape=(im_dim,im_dim,1), name = 'input')
    im_dim = encoder_dims[0]
    input_layer = Input((im_dim*im_dim*1,), name = 'input')
    x = Reshape((im_dim,im_dim,1))(input_layer)
   
    #x = Reshape((im_dim,im_dim,1))(input_layer)
    
    #Apply Convolution and Pooling alternatively...
    x = Conv2D(32, (3, 3),activation='relu' ,padding='same')(x) 
    #x=BatchNormalization()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3),activation='relu' , padding='same')(x)
    #x=BatchNormalization()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3),strides=(1,1),activation='relu' ,  padding='same')(x)
    #x=BatchNormalization()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #----------Store dimensions before we flatten
    shape_before_flattening = K.int_shape(x)[1:]
    #Unpack
    w, h, fmaps = shape_before_flattening
    logging.debug("Shape before flattening : {}".format(shape_before_flattening))
    #Store number of neurons
    num_neurons = w*h*fmaps 
    
    #x=BatchNormalization()(x)
    x = Flatten(name = 'flat')(x)
    
    #Feed into Dense
    encoded_output = Dense(latent_dim,name='z',activation='relu' )(x)
    #encoded_output = ReLU()(encoded_output)
    #Full model
    encoder = Model(input_layer, encoded_output, name = 'encoder')
    encoder.summary()
    
    
    #-------Decoder(AE)
    #Expects 100 as input
    x = Dense(num_neurons, name='decoder_dense',activation='relu' )(encoded_output)
    #x = ReLU()(x)
    x = Reshape(shape_before_flattening, name='decoder_10')(x)
    
    #x=BatchNormalization()(x)
    x = UpSampling2D((2, 2), name = 'decoder_9')(x)
    x = Conv2D(64, (3, 3),  padding='same',activation='relu' , name = 'decoder_8')(x) 
    #x=BatchNormalization()(x)
    #x = ReLU()(x)
    
    #x=BatchNormalization()(x)
    x = UpSampling2D((2, 2), name = 'decoder_7')(x)
    x = Conv2D(32, (3, 3),  padding='same',activation='relu' , name = 'decoder_6')(x)
    #x=BatchNormalization()(x)
    #x = ReLU()(x)
    
    
    x = UpSampling2D((2, 2), name = 'decoder_5')(x) 
    #x = Conv2D(1, (3, 3), activation='relu', padding='same', name = 'decoder_2')(x) 
    
    #LAST OUTPUT LAYER in 3D
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='decoder_4')(x)
    x = Flatten(name='decoder_3')(x)  
    x = Dense(im_dim*im_dim, activation='relu', name='decoder_2')(x)
    x = Reshape((im_dim,im_dim,1),name='decoder_1')(x)
    
    decoded = Conv2D(1, (3, 3), activation='linear', padding='same', name='decoded_output_3d')(x)
    decoded = Flatten(name='decoder_0')(decoded)  
    
    autoencoder = Model(input_layer, decoded, name='autoencoder')
    autoencoder.summary()
    
    
    
    #----------------Standalone Decoder
    z = latent_dim
    #Make an Input for Decoder
    dec_input_layer = Input((z,),name='d_in')
    
    #Reshape as before
    x = autoencoder.get_layer('decoder_dense')(dec_input_layer)
    
    #Stack from 5-->1
    x = autoencoder.get_layer('decoder_10')(x)
    x = autoencoder.get_layer('decoder_9')(x)
    x = autoencoder.get_layer('decoder_8')(x)
    x = autoencoder.get_layer('decoder_7')(x)
    x = autoencoder.get_layer('decoder_6')(x)
    x = autoencoder.get_layer('decoder_5')(x)
    x = autoencoder.get_layer('decoder_4')(x)
    x = autoencoder.get_layer('decoder_3')(x)
    x = autoencoder.get_layer('decoder_2')(x)
    x = autoencoder.get_layer('decoder_1')(x)
    
    #Go back up to 3D
    x = autoencoder.get_layer('decoded_output_3d')(x)
    decoder_output = autoencoder.get_layer('decoder_0')(x)
    
    #Define Decoder
    decoder = Model(dec_input_layer, decoder_output, name = 'decoder')
    decoder.summary()


    
    return (autoencoder, encoder, decoder)

'''
autoencoder.compile(loss='mse', optimizer = 'adam')
'''
