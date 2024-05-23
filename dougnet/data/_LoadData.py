import os
import struct
import numpy as np

def LoadMNIST():
    """
    Load MNIST data 

    Returns
    -------
    X_train, y_train, X_test, y_test (as a tuple)
    numpy arrays of MNIST data

    Author: Douglas Rubin (code modified from Sebastian Raschka)
    
    """
    current_path = os.path.dirname(os.path.abspath(__file__))

    # if data not yet downloaded, download and save
    if not os.path.exists(current_path + '/mnist.npz'):
        url1 = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        url2 = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        url3 = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        url4 = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

        # download data
        os.system('curl -o ' + current_path + '/mnist_training_features.gz ' + url1)
        os.system('curl -o  ' + current_path + '/mnist_training_labels.gz ' + url2)
        os.system('curl -o  ' + current_path + '/mnist_testing_features.gz ' + url3)
        os.system('curl -o  ' + current_path + '/mnist_testing_labels.gz ' + url4)

        # unzip data
        os.system('gzip ' + current_path + '/mnist_*.gz -d')


        # define helper function to load data into numpy arrays
        def load_mnist_helper(path, kind='train'):
            """Unpack MNIST data from byte format to numpy arrays"""
            if kind == 'train':
                images_path = current_path + '/mnist_training_features'
                labels_path = current_path + '/mnist_training_labels'
            else:
                images_path = current_path + '/mnist_testing_features'
                labels_path = current_path + '/mnist_testing_labels'

            with open(labels_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                labels = np.fromfile(lbpath, dtype=np.uint8)

            with open(images_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
                images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
                images = ((images / 255.) - .5) * 2

            return images, labels

        X_train, y_train = load_mnist_helper('', kind='train')
        X_test, y_test = load_mnist_helper('', kind='test')

        # save data in numpy format
        np.savez_compressed(current_path + '/mnist.npz', 
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test
                            )

        # delete the downloaded byte format data
        os.system('rm ' + current_path + '/mnist_t* ')

        
    # load data and return
    mnist = np.load(current_path + '/mnist.npz')
    return (mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test'])