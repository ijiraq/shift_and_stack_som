"""
Dataset loading functions

Modified by JJ Kavelaars - 2024-02-16

@author Florent Forest
@version 1.0
"""

import logging
import numpy as np
import os
from sklearn import preprocessing
# import StandardScaler
from sklearn import model_selection as md


def load_spectra_data():
    
    # Specify where your file is
    spectra_datapath = os.path.join(os.getcwd(), 'spec250k.npy')
    X = np.load(spectra_datapath)
    Xt,Xv = md.train_test_split(X, test_size=.3)

    # Standardize each ROW
    print("Standardizing data... ")
    # scaler_m= preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Xt.T)
    scaler_s= preprocessing.StandardScaler().fit(Xt.T)
    Xt= scaler_s.transform(Xt.T)
    Xv= scaler_s.transform(Xv.T)


    X_train = Xt.T
    X_test = Xv.T

    print ('Savine X_train...')
    np.save('X_train',X_train)
    print ('Savine X_test...')
    np.save('X_test',X_test)

    print ('Done!')    
    #return (X_train, None), (None, None)
    print ('Done!')    
    
    #print("SPECTRA Data dimensions: {}".format(X_train.shape))


    #return X_train


def load_mnist(flatten=True, validation=False):
    # Dataset, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Divide by 255.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if flatten: # flatten to 784-dimensional vector
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    if validation: # Return train and test set
        return (x_train, y_train), (x_test, y_test)
    else: # Return only train set with all images
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        return (x, y), (None, None)


def load_fashion_mnist(flatten=True, validation=False):
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Divide by 255.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if flatten: # flatten to 784-dimensional vector
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    if validation: # Return train and test set
        return (x_train, y_train), (x_test, y_test)
    else: # Return only train set with all images
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        return (x, y), (None, None)

def load_usps(data_path='./data/usps'):
    import h5py
    with h5py.File(data_path+'/usps.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
    x = np.concatenate((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))
    print('USPS samples', x.shape)
    return (x, y), (None, None)

def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return (x, y), (None, None)

def make_reuters_data(data_dir):
    """
    NOTE: RCV1-V2 data is heavy and not included.
    The data can be downloaded from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
    Necessary files are:
        'rcv1-v2.topics.qrels'
        'lyrl2004_tokens_test_pt0.dat'
        'lyrl2004_tokens_test_pt1.dat',
        'lyrl2004_tokens_test_pt2.dat',
        'lyrl2004_tokens_test_pt3.dat',
        'lyrl2004_tokens_train.dat'
    """
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_im(data_path,
            filename,
            normalize=False,
            im_size=32,
            train_size=500000,
            val_size=80000):

    if normalize:
        n_file= 0
        cat_tr = []
        cat_va = []
        for im_file in [filename, ]:
            n_file += 1
            data0 = np.load(os.path.join(data_path, im_file))
            data = []
            for k1 in range(len(data0)):
                len_data = len(np.where(np.isnan(data0[k1])==1)[0])
                if len_data == 0:
                    data.append(data0[k1])
            data = np.array(data)
            sz= np.shape(data)
            np.random.shuffle(data)
            logging.info(f"Loading {im_file} of size {sz}")
            cat_tr.append(data[0:train_size])
            cat_va.append(data[train_size:train_size+val_size])
        cat_tr = np.concatenate(cat_tr).reshape(-1, im_size*im_size)
        cat_va = np.concatenate(cat_va).reshape(-1, im_size*im_size)
        np.random.shuffle(cat_tr)
        np.random.shuffle(cat_va)
        logging.info('Saving Training and Validation sets ...')
        np.save('X_train', cat_tr)
        np.save('X_test', cat_va)

    else:
        logging.info('Loading Training set from {filename}')
        cat_tr = np.load(os.path.join(data_path, filename))
    logging.info('Final Training size... ', np.shape(cat_tr))

    return (cat_tr, None), (None, None)


def load_data(dataset_name, train_size, val_size, validation=False):
    if dataset_name == 'mnist':
        return load_mnist(flatten=True, validation=validation)
    elif dataset_name == 'fmnist':
        return load_fashion_mnist(flatten=True, validation=validation)
    elif dataset_name == 'usps':
        if validation:
            print('Train/validation split is not available for this dataset.')
        return load_usps()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        if validation:
            print('Train/validation split is not available for this dataset.')
        return load_reuters()
    elif dataset_name == 'im':
        print ('train_size: ', train_size)
        print ('val_size: ', val_size)
        return load_im(train_size=train_size, val_size=val_size, validation=validation)
    
    elif dataset_name == 'spectra':
        return load_spectra_data()
    else:
        print('Dataset {} not available! Available datasets are im, mnist, fmnist, usps and reuters10k.'.format(dataset_name))
        exit(0)
