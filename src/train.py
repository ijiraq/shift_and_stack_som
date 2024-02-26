'''
Module for training DESOM
'''

import argparse
import logging

import numpy as np
from desom import DESOM
import matplotlib.pyplot as plt
from datasets import load_spectra_data, load_mnist, load_im
import inspect


def run():
    """
    Based on the CL Args train a SoM based on some input data set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename',
                        help="A .npy file holding a list of image_size "
                             "x image_size cutouts of sources to create a SoM for")
    parser.add_argument('--map-size', nargs='+', default=[11, 11], type=int)
    parser.add_argument('--latent-dim', nargs=1, default=32, type=int)
    parser.add_argument('--dataset', default='image',
                        choices=['mnist', 'spectra','image'])
    parser.add_argument('--train-size', default= 1000000, type=int)
    parser.add_argument('--ae-type', default='cnn2D',
                        choices=['cnn2D', 'cnn1D', 'fc_ae'])
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='coefficient of self-organizing map loss')

    parser.add_argument('--ae-weights',default="",  type=str)

    parser.add_argument('--ae-epochs', default=0, type=int)
    parser.add_argument('--iterations', default=40000, type=int)
    parser.add_argument('--som-iterations', default=40000, type=int)
    parser.add_argument('--save-epochs', default=1000, type=int)

    # Batch SIZES
    parser.add_argument('--model-batch-size', default=512, type=int)
    parser.add_argument('--ae-batch-size', default=512, type=int)
    parser.add_argument('--Tmax', default=12.0, type=float)
    parser.add_argument('--Tmin', default=0.5, type=float)
    parser.add_argument('--save-dir', default='jjk_test_run', type=str)

    parser.add_argument('--log-level', default='ERROR',
                        choices=['ERROR', 'INFO', 'DEBUG'])
    args = parser.parse_args()

    logging.basicConfig(
        format='%(module)-12s %(message)s',
        level=getattr(logging, args.log_level))

    load_function = f"load_{args.dataset}_data"
    x_train, y_train, encoder_dims = globals()[load_function](args.data_filename)

    kwargs = {}
    for parameter in inspect.signature(get_som).parameters:
        if hasattr(args, parameter):
            kwargs[parameter] = getattr(args, parameter)

    logging.debug(f"Training SOM with parameters set to: {kwargs}")
    get_som(x_train, y_train, encoder_dims, **kwargs)


def load_spectral_data(spectral_filename, norm=False):
    """Load spectral data and specify ENCODER DIMENSIONS(ie.input dimensions)"""

    if norm:
        load_spectra_data()
        x_train= np.load(spectral_filename)[0:250000]
    else:
        logging.info('---No Normalization...')
        x_train=np.load(spectral_filename)
        logging.info(f"Xsom_tr= {np.shape(x_train)}")
    encoder_dims = [x_train.shape[1]]
    return x_train, None, encoder_dims


def load_image_data(data_filename):
    # Specify path to load images FROM
    logging.info('Loading Training set from {filename}')
    x_train = np.load(data_filename)
    y_train = None
    dim = int(np.sqrt(x_train.shape[1]))
    logging.info(f"setting dimensions to {dim}x{dim}")
    encoder_dims = [dim]
    return x_train, y_train, encoder_dims


def load_mnist_data():
    (X_train, y_train), (X_val, y_val) = load_mnist()
    encoder_dims = [int(np.sqrt(X_train.shape[1]))]
    return X_train, y_train, encoder_dims


def plot_spectral_som(som, fig_filename='spectral_som.png'):
    # Setup Map Size
    map_size = som.map_size
    upper_lim = map_size[0] * map_size[1]
    prototypes = som.prototypes
    decoded_prototypes = som.decode(prototypes)

    # Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(8, 8))
    # Iterate over each and every prototype
    for k in range(upper_lim):
        x = decoded_prototypes[k]
        ax[k // map_size[1]][k % map_size[1]].plot(x)
        ax[k // map_size[1]][k % map_size[1]].axis('off')

    plt.subplots_adjust(hspace=1.05, wspace=1.05)
    plt.savefig(fig_filename)


def plot_image_som(som, fig_filename='image_som.png'):
    # Setup Map Size
    map_size = som.map_size
    upper_lim = map_size[0] * map_size[1]
    prototypes = som.prototypes
    decoded_prototypes = som.decode(prototypes)
    im_size = int(decoded_prototypes[0].shape**0.5)

    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(8,8))
    # Iterate over each and every prototype
    for k in range(upper_lim):
        x = decoded_prototypes[k].reshape(im_size, im_size)
        ax[k // map_size[1]][k % map_size[1]].imshow(x, cmap='gray')
        ax[k // map_size[1]][k % map_size[1]].axis('off')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(fig_filename)


def get_som(x_train, y_train, encoder_dims, ae_type, map_size, gamma,
            latent_dim, ae_epochs, ae_batch_size, iterations,
            som_iterations, save_epochs, model_batch_size, Tmax, Tmin,
            optimizer='adam', ae_act='relu', ae_init='glorot_uniform',
            decay='exponential',
            save_dir='results'):

    som = DESOM(encoder_dims=encoder_dims,
                ae_type=ae_type,
                map_size=map_size,
                latent_dim=latent_dim)
    # Specify Gamma and Optimizer
    # Initialize DESOM
    som.initialize(ae_act, ae_init)
    # Compile
    som.compile(gamma, optimizer)

    # Train AE
    som.pretrain(x_train,
                 optimizer=optimizer,
                 epochs=ae_epochs,
                 batch_size=ae_batch_size,
                 save_dir=save_dir)

    # som.load_ae_weights("results/tmp/ae_weights-epoch150-MP.h5")

    # Train model
    som.init_som_weights(x_train)
    som.fit(x_train, y_train,
            iterations=iterations,
            som_iterations=som_iterations,
            save_epochs=save_epochs,
            batch_size=model_batch_size,
            decay=decay,
            Tmax=Tmax,
            Tmin=Tmin,
            save_dir=save_dir
            )
    return som


if __name__ == '__main__':
    run()










