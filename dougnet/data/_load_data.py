import os
import struct
import pickle
import tempfile
import numpy as np

HOME_DIR = os.environ['HOME']

data_docstring = """
Load {0} data.

Parameters
----------
data_dir : string, default="$HOME/dougnet_data"
    Local directory for data (directory created if it does not already exist).  If data 
    file exists in data_dir this function will read-in and return the data.  If it does 
    not, this function will download the data to data_dir, read it in and return it.

Returns
-------
X_train, y_train, X_test, y_test (as a tuple)
    numpy arrays of data    
"""


# define helper function to load downloaded mnist data into numpy arrays
def _load_mnist_helper(path, kind='train'):
    """Unpack MNIST data from byte format to numpy arrays"""
    if kind == 'train':
        images_path = path + '/mnist_training_features.gz'
        labels_path = path + '/mnist_training_labels.gz'
    else:
        images_path = path + '/mnist_testing_features.gz'
        labels_path = path + '/mnist_testing_labels.gz'

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


def _LoadMNIST_main(data_dir):
    """Download mnist data from Yann Lecun's website."""
    url1 = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url2 = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    url3 = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url4 = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    
    # if data not yet downloaded, download and save
    if not os.path.exists(data_dir + '/mnist.npz'):
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        
            # download data
            os.system('curl -o ' + tmp + '/mnist_training_features.gz ' + url1)
            os.system('curl -o  ' + tmp + '/mnist_training_labels.gz ' + url2)
            os.system('curl -o  ' + tmp + '/mnist_testing_features.gz ' + url3)
            os.system('curl -o  ' + tmp + '/mnist_testing_labels.gz ' + url4)

            # unzip data
            os.system('gzip ' + tmp + '/mnist_*.gz -d')

            # load data
            X_train, y_train = _load_mnist_helper(tmp, kind='train')
            X_test, y_test = _load_mnist_helper(tmp, kind='test')

        # save data in numpy format
        np.savez_compressed(data_dir + '/mnist.npz', 
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test
                            )
        

def _LoadMNIST_backup(data_dir):
    """Download mnist data from backup website."""
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    
    # if data not yet downloaded, download and save
    if not os.path.exists(data_dir + '/mnist.npz'):
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        
            # download data
            os.system('curl -o ' + tmp + '/mnist_data.npz ' + url)

            # load data
            mnist_file = np.load(tmp + '/mnist_data.npz')
            X_test = mnist_file["x_test"].reshape(10_000, -1)
            X_train = mnist_file["x_train"].reshape(60_000, -1)
            y_test = mnist_file["y_test"].astype(np.int64)
            y_train = mnist_file["y_train"].astype(np.int64)

        # save data in numpy format
        np.savez_compressed(data_dir + '/mnist.npz', 
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test
                            )
        
        
def LoadMNIST(data_dir=f"{HOME_DIR}/dougnet_data"):
    # make directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # download data if not already
    try:
        _LoadMNIST_main(data_dir)
    except:
        _LoadMNIST_backup(data_dir)

    # load data and return
    mnist = np.load(data_dir + '/mnist.npz')
    return (mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test'])
LoadMNIST.__doc__ = data_docstring.format("MNIST")


def LoadCIFAR10(data_dir=f"{HOME_DIR}/dougnet_data"):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    # make directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # download data if not already downloaded
    if not os.path.exists(data_dir + '/cifar10.npz'):
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
            # download data and untar
            os.system(f"curl -o {tmp}/cifar-10-python.tar.gz {url}")
            os.system(f"tar -xvzf {tmp}/cifar-10-python.tar.gz -C {tmp}")

            # read-in training data and concatenate
            data = []
            labels = []
            for i in range(1, 6):
                with open(f"{tmp}/cifar-10-batches-py/data_batch_{i}", 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                    data.append(data_dict[b'data'])
                    labels.append(data_dict[b'labels'])
            X_train = np.concatenate(data).reshape(50_000, 3, 32, 32).transpose(0, 2, 3, 1)
            y_train = np.concatenate(labels)

            # read-in testing data
            with open(f"{tmp}/cifar-10-batches-py/test_batch", 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            X_test = data_dict[b'data'].reshape(10_000, 3, 32, 32).transpose(0, 2, 3, 1)
            y_test = np.array(data_dict[b'labels'])


        # save data in numpy format
        np.savez_compressed(data_dir + '/cifar10.npz', 
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test
                            )
        
    # load data and return
    data = np.load(data_dir + '/cifar10.npz')
    return (data[f] for f in ['X_train', 'y_train', 'X_test', 'y_test'])
LoadCIFAR10.__doc__ = data_docstring.format("CIFAR10")


def LoadBB(data_dir=f"{HOME_DIR}/dougnet_data"):
    """
    Load Beavis and Butthead do America script.

    Parameters
    ----------
    data_dir : string, default="$HOME/dougnet_data"
        Local directory for data (directory created if it does not already exist).  If data 
        file exists in data_dir this function will read-in and return the data.  If it does 
        not, this function will download the data to data_dir, read it in and return it.

    Returns
    -------
    Beavis and Butthead do America script as a string   
    """
    url = 'https://www.dailyscript.com/scripts/beavis_and_butthead_do_america.html'

    # make directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # download data if not already downloaded
    if not os.path.exists(data_dir + '/beavis_and_butthead.txt'):
        
        # download
        os.system(f"curl -o {data_dir}/beavis_and_butthead.html {url}")
        
        # read-in html file
        with open(f'{data_dir}/beavis_and_butthead.html', 'r', encoding="utf8") as fp:
            text=fp.read()
        
        # trim unwanted chars at beginning and end
        start_indx = text.find("\n\t\n\t\n\tINT")
        end_indx = text.find("\n\t\n\t\t\t\t\t\tEND")
        text = text[start_indx:end_indx]
        
        # write as a txt file
        with open(f'{data_dir}/beavis_and_butthead.txt', "w") as text_file:
            text_file.write(text)
            
        # delete downloaded html file
        os.system(f"rm {data_dir}/beavis_and_butthead.html")
        
    with open(f'{data_dir}/beavis_and_butthead.txt', "r") as file:
        text = file.read()
        
    return text