from __future__ import division
from __future__ import print_function
import os
import sys
import math
import cPickle
import numpy as np
import scipy as sp
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import glob
import cv2

def perform_pca(data, dim):
    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:])))
    pca = PCA(n_components=dim)
    return pca.fit_transform(data)

def random_horizontal_flip(images):
    rands = 2*np.random.randint(2, size=len(images)) - 1
    flipped_images = np.asarray([image[:, ::rand, :] for image, rand in zip(images, rands)])
    return flipped_images

def resize_images(images, new_size, interpolation=cv2.INTER_LINEAR):
    resized_images = np.asarray([cv2.resize(image, new_size, interpolation=interpolation) for image in images])
    return resized_images

def crop_images(images, size, offsets=None):
    if offsets is None:
        # center crop
        cur_height, cur_width = images.shape[1:3]
        offset_h, offset_w = cur_height - size[0], cur_width - size[1]
        odd_h, odd_w = offset_h % 2, offset_w % 2
        offsets = (offset_h//2+odd_h, offset_w//2+odd_w)
    offsets = np.asarray(offsets)
    if len(offsets.shape) == 1:
        offsets = np.tile(offsets, (len(images), 1))

    cropped_images = np.asarray([image[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1], :] for image, offset in zip(images, offsets)])
    return cropped_images

def random_crop(images, new_size):
    offsets = np.random.randint(8, size=(len(images), 2))
    cropped_images = crop_images(images, new_size, offsets)
    return cropped_images

def random_brightness(images, maxdelta):
    def _adjust_brightness(image, delta):
        adjusted_image = image + delta
        adjusted_image = np.clip(adjusted_image, 0., 1.)
        return adjusted_image

    if maxdelta < 0.: maxdelta = -maxdelta
    if maxdelta > 1.: maxdelta = 1.
    deltas = np.random.uniform(-maxdelta, maxdelta, size=(len(images)))
    randomized_images = np.asarray([_adjust_brightness(image, delta) for image, delta in zip(images, deltas)])
    return randomized_images

def random_contrast(images, lower, upper):
    def _adjust_contrast(image, factor):
        mean_px = np.mean(image)
        adjusted_image = image + factor * (image - mean_px)
        adjusted_image = np.clip(adjusted_image, 0., 1.)
        return adjusted_image

    if lower > upper: lower, upper = upper, lower
    if lower < 0.: lower = 0.
    factors = np.random.uniform(lower, upper, size=(len(images)))
    randomized_images = np.asarray([_adjust_contrast(image, factor) for image, factor in zip(images, factors)])
    return randomized_images

def per_image_standardization(images):
    def _per_image_standardization(image):
        h, w = image.shape[0:2]
        image_mean = np.mean(image)
        image_sd = np.std(image)
        min_sd = 1. / math.sqrt(float(h*w))
        m = image_mean
        sd = max(image_sd, min_sd)
        standardized_image = (image - m) / sd
        return standardized_image

    standardized_images = np.asarray([_per_image_standardization(image) for image in images])
    return standardized_images

def distorted_images(images, new_size):
    images = random_crop(images, new_size)
    images = random_horizontal_flip(images)
    images = random_brightness(images, 0.4)
    images = random_contrast(images, 0.2, 1.8)
    #images = per_image_standardization(images)
    return images

class DatasetBase(object):
    def __init_(self):
        self.images = np.empty(1)
        self.train_images = np.empty(1)
        self.test_images = np.empty(1)
        self.train_size = 0
        self.labels = np.empty(1)
        self.train_labels = np.empty(1)
        self.test_labels = np.empty(1)
        self.codes = None
        self.train_codes = None
        self.test_codes = None
        self.read_idx = -1
        self.rand_idx = None

    def save_codes(self, pkl_path):
        self.codes.dump(pkl_path)

    def load_codes(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.codes = cPickle.load(f)

    def train_batch(self, size, with_idx=False):
        idx = self.rand_idx[self.read_idx+1 : self.read_idx+1 + size]

        if self.read_idx+1 + size*2 > self.train_size+1:
            last_batch = True
            self.read_idx = -1
            self.rand_idx = np.random.permutation(self.train_size)
        else:
            last_batch = False
            self.read_idx += size

        if with_idx:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], idx, last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    idx, last_batch
        else:
            if self.codes is not None:
                return self.train_images[idx], self.train_labels[idx],\
                    self.train_codes[idx], last_batch
            else:
                return self.train_images[idx], self.train_labels[idx],\
                    last_batch

    def test_batch(self, size, with_idx=False):
        idx = np.random.permutation(len(self.images)-self.train_size)[:size]
        if with_idx:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx], idx
            else:
                return self.test_images[idx], self.test_labels[idx], idx
        else:
            if self.codes is not None:
                return self.test_images[idx], self.test_labels[idx],\
                    self.test_codes[idx]
            else:
                return self.test_images[idx], self.test_labels[idx]

    def construct_subdataset(self):
        if len(self.labels[0].shape) > 1 or self.labels[0].shape[0] > 1:
            one_hot = True
        else:
            one_hot = False
        self.train_subdata = []
        self.test_subdata = []
        for i in xrange(self.n_labels):
            if one_hot:
                #label = np.zeros(self.n_labels, dtype=np.float32)
                #label[i] = 1.
                #subset_idxs_train = np.array([j for j, l in enumerate(self.train_labels) if all(l == label)])
                #subset_idxs_test = np.array([j for j, l in enumerate(self.test_labels) if all(l == label)])
                label = i
                subset_idxs_train = np.array([j for j, l in enumerate(self.train_labels) if np.argmax(l) == label])
                subset_idxs_test = np.array([j for j, l in enumerate(self.test_labels) if np.argmax(l) == label])
            else:
                label = i
                subset_idxs_train = np.array([j for j, l in enumerate(self.train_labels) if l == label])
                subset_idxs_test = np.array([j for j, l in enumerate(self.test_labels) if l == label])
            subset_train = self.train_images[subset_idxs_train]
            subset_test = self.test_images[subset_idxs_test]
            self.train_subdata.append(subset_train)
            self.test_subdata.append(subset_test)

    def get_data(self, label, size, train=True, random=True):
        if not hasattr(self, 'train_subdata'):
            self.construct_subdataset()

        #if not isinstance(label, int) and len(label.shape) > 1:
        if not isinstance(label, int):
            if isinstance(label, list):
                label = np.asarray(label)
            label = np.argmax(label)
        if train:
            data = self.train_subdata[label]
        else:
            data = self.test_subdata[label]

        if random:
            idx = np.random.permutation(len(data))[:size]
        else:
            idx = np.asarray(range(min(len(data), size)))
        return data[idx]

class MixtureGaussianDataset(DatasetBase):
    def __init__(self, dataset_size):
        super(MixtureGaussianDataset, self).__init__()
        self.n_labels = 8
        self.mus = [(2*math.cos(i*math.pi/4), 2*math.sin(i*math.pi/4)) for i in xrange(8)]
        subset_size = dataset_size // 8
        self.dataset_size = subset_size * 8
        self.labels = []
        for i in xrange(8):
            self.labels += [i]*subset_size
        self.images = [np.random.normal(self.mus[label], 0.02, 2) for label in self.labels]
        self.images, self.labels = np.asarray(self.images), np.asarray(self.labels)
        rand = np.random.permutation(self.dataset_size)
        self.images, self.labels = self.images[rand], self.labels[rand]
        self.train_size = int(self.dataset_size * 0.9)
        self.train_images = self.images[:self.train_size]
        self.test_images = self.images[self.train_size:]
        self.train_labels = self.labels[:self.train_size]
        self.test_labels = self.labels[self.train_size:]
        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)
        self.codes = None

    #def test_batch(self, size, with_idx=False):
    #    labels = np.random.randint(8, size=(size))
    #    samples = [np.random.normal(self.mus[label], 0.02, 2) for label in labels]
    #    if with_idx:
    #        return self.test_images[idx], self.test_labels[idx], None
    #    else:
    #        return self.test_images[idx], self.test_labels[idx]

class MnistDataset(DatasetBase):
    def __init__(self, code_dim, code_init, color=False):
        super(MnistDataset, self).__init__()
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        self.train_images = mnist.train.images
        self.train_images = np.reshape(self.train_images, [len(self.train_images), 28, 28, 1])
        self.train_labels = mnist.train.labels
        self.train_size = len(self.train_images)
        self.test_images = mnist.test.images
        self.test_images = np.reshape(self.test_images, [len(self.test_images), 28, 28, 1])
        self.test_labels = mnist.test.labels
        self.images = np.concatenate((self.train_images, self.test_images))
        self.labels = np.concatenate((self.train_labels, self.test_labels))
        self.n_labels = 10

        if code_init == None:
            self.codes = None
        elif code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        else:
            print('load codes from ' + code_init)
            self.load_codes(code_init)
            print('done.')

        if self.codes is not None:
            self.codes /= np.linalg.norm(self.codes, axis=1, keepdims=True)

        if color == True:
            self.images = np.asarray([np.tile(image(1, 1, 3)) for image in self.images])

        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

class FashionMnistDataset(DatasetBase):
    def __init__(self, code_dim, code_init, color=False):
        super(FashionMnistDataset, self).__init__()
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('/home/chiba/data/fashion-mnist/data/fashion/', one_hot=True)
        self.train_images = mnist.train.images
        self.train_images = np.reshape(self.train_images, [len(self.train_images), 28, 28, 1])
        self.train_labels = mnist.train.labels
        self.train_size = len(self.train_images)
        self.test_images = mnist.test.images
        self.test_images = np.reshape(self.test_images, [len(self.test_images), 28, 28, 1])
        self.test_labels = mnist.test.labels
        self.images = np.concatenate((self.train_images, self.test_images))
        self.labels = np.concatenate((self.train_labels, self.test_labels))
        self.n_labels = 10

        if code_init == None:
            self.codes = None
        elif code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        else:
            print('load codes from ' + code_init)
            self.load_codes(code_init)
            print('done.')

        if self.codes is not None:
            self.codes /= np.linalg.norm(self.codes, axis=1, keepdims=True)

        if color == True:
            self.images = np.asarray([np.tile(image(1, 1, 3)) for image in self.images])

        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

class Cifar10Dataset(DatasetBase):
    def __init__(self, dataset_path, code_dim, code_init):
        super(Cifar10Dataset, self).__init__()
        self.images = None
        self.labels = None
        self.names = None
        self.codes = None
        self.code_dim = code_dim
        for i in xrange(1, 6):
            print('Extracting data_batch_%s...'%i)

            subset_path = os.path.join(dataset_path, 'data_batch_%s'%i)
            if not os.path.exists(subset_path):
                raise ValueError('File %s does not exist.'%subset_path)

            data = self._unpickle(subset_path)
            images = data['data']
            images = images.astype(np.float32) / 255.
            images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

            if self.images is None:
                self.images = images
            else:
                self.images = np.concatenate((self.images, images), axis=0)

            if self.labels is None:
                self.labels = self._get_onehot_vectors(data['labels'])
            else:
                self.labels = np.concatenate((self.labels,
                    self._get_onehot_vectors(data['labels'])), axis=0)

            if self.names is None:
                self.names = data['filenames']
            else:
                self.names = np.concatenate((self.names,
                    data['filenames']), axis=0)

        print('Extracting test_batch...')
        subset_path = os.path.join(dataset_path, 'test_batch')
        if not os.path.exists(subset_path):
            raise ValueError('File %s does not exist.'%subset_path)

        data = self._unpickle(subset_path)
        images = data['data']
        images = images.astype(np.float32) / 255.
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

        self.images = np.concatenate((self.images, images), axis=0)
        self.labels = np.concatenate((self.labels,
                self._get_onehot_vectors(data['labels'])), axis=0)
        self.names = np.concatenate((self.names,
                data['filenames']), axis=0)
        self.n_labels = 10
        
        if code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        elif code_init is not None:
            print('load codes from ' + code_init)
            self.load_codes(code_init)
            print('done.')
        
        if self.codes is not None:
            self.codes /= np.linalg.norm(self.codes, axis=1, keepdims=True)

        self.train_size = 50000
        self.train_images = self.images[:self.train_size]
        self.test_images = self.images[self.train_size:]
        self.train_labels = self.labels[:self.train_size]
        self.test_labels = self.labels[self.train_size:]
        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

    def _unpickle(self, file_path):
        f = open(file_path, 'rb')
        data = cPickle.load(f)
        f.close()
        return data

    def _get_onehot_vectors(self, labels):
        x = np.array(labels).reshape(1, -1)
        x = x.transpose()
        encoder = OneHotEncoder(n_values=max(x)+1)
        x = encoder.fit_transform(x).toarray()
        return x

class CelebADataset(DatasetBase):
    def __init__(self, dataset_path, code_dim, code_init):
        super(CelebADataset, self).__init__()
        self.images = None
        self.codes = None
        self.n_labels = 1

        if self.images is None:
            print('load images...')
            files = glob.glob(os.path.join(dataset_path, '*.jpg'))
            images = [cv2.imread(file) for file in files]
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            images = [cv2.resize(image, (64, 64)) for image in images]
            images = [image.astype(np.float32) / 255. for image in images]
            self.images = np.asarray(images)
            print('done.')

        # dummy labels
        self.labels = np.zeros([len(self.images), 1])

        if code_init == 'gaussian':
            self.codes = np.random.randn(len(self.images), code_dim)
        elif code_init == "pca":
            print('processing pca...')
            self.codes = perform_pca(self.images, code_dim)
            print('done.')
        elif code_init is not None:
            print('load codes from ' + code_init)
            self.load_codes(code_init)
            print('done.')
        
        if self.codes is not None:
            self.codes = self.codes / np.linalg.norm(self.codes, axis=1, keepdims=True)

        self.train_size = int(0.9*len(self.images))
        self.train_images = self.images[:self.train_size]
        self.test_images = self.images[self.train_size:]
        self.train_labels = self.labels[:self.train_size]
        self.test_labels = self.labels[self.train_size:]
        if self.codes is not None:
            self.train_codes = self.codes[:self.train_size]
            self.test_codes = self.codes[self.train_size:]

        self.read_idx = -1
        self.rand_idx = np.random.permutation(self.train_size)

def tile_image(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d, w*d, 3), dtype=np.float32)
    for idx, img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h, idx_x*w:(idx_x+1)*w, :] = img
    return r
if __name__ == "__main__":
    import time
    #s = time.time()
    #dataset = CelebADataset('/home/develop/research/img_align_celeba/', 0, None)
    #e = time.time() - s
    #print(e)
    import cv2
    dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py', code_dim=0, code_init=None)
    image = dataset.test_images[1]
    images = np.tile(image, (64, 1, 1, 1))
    s = time.time()
    images = distorted_images(images)
    e = time.time() - s
    print(e)

    images = tile_image(images) * 255.
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    cv2.imwrite("distorted.png", images)

