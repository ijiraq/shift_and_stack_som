from keras.layers import Input, Dense, BatchNormalization, Reshape, Flatten
from keras.models import Model
from keras import backend as K




'''
Constructs a Fully-Connected Autoencoder
'''
def fc_ae(in_dims):

    #--------------Encoder
    x_in = Input((in_dims,), name = 'input')

    #Compression layers
    x = Dense(128, activation='relu')(x_in)
    x = Dense(64, activation='relu')(x)
    encoded_out = Dense(32, activation='relu', name='z')(x)

    #ENCODER
    encoder = Model(x_in, encoded_out, name = 'encoder')
    encoder.summary()

    #Store the latent dimension
    latent_dim = K.int_shape(encoded_out)[1]

    
    #-----------------Decoder layers
    x = Dense(64, activation='relu', name='decoder_2')(encoded_out)
    x = Dense(128, activation='relu', name='decoder_1')(x)
    #Final output
    d_out = Dense(in_dims, activation='sigmoid', name='decoder_0')(x)
    
    autoencoder = Model(x_in, d_out, name = 'autoencoder')
    autoencoder.summary()

    #-----------------Unpack Decoder
    d_in = Input((latent_dim,))

    x = autoencoder.get_layer('decoder_2')(d_in)
    x = autoencoder.get_layer('decoder_1')(x)

    d_out = autoencoder.get_layer('decoder_0')(x)

    decoder = Model(d_in, d_out, name = 'decoder')
    decoder.summary()
    
    return (autoencoder, encoder, decoder)