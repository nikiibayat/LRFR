import torch
import os
import os.path as osp
import pickle
import lmdb
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import scipy


class VGG_Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index][0], self.samples[index][1]


def folder2lmdb(samples, name="train", write_frequency=5000, num_workers=16):
    batch_size = 128
    dataset = VGG_Dataset(samples)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    print("Number of training samples in total: {}".format(len(samples)))
    print("Number of batches: {}".format(len(data_loader)))
    lmdb_path = osp.join("./dataset", "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=10e+11, readonly=False,
                   meminit=False, map_async=True)

    ii = 0
    txn = db.begin(write=True)
    for idx, (images, labels) in enumerate(data_loader):
        for j in range(len(images)):
            img = scipy.misc.imread(images[j], mode='RGB').astype(np.float)
            img_hr = scipy.misc.imresize(img, (64, 64))
            img_lr = scipy.misc.imresize(img, (16, 16))
            # If training => do random flip
            if name == "train" and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            img_hr = np.array(img_hr) / 127.5 - 1.
            img_lr = np.array(img_lr) / 127.5 - 1.
            print("putting image {} with label {}".format(ii, labels[j].shape))
            txn.put(u'{}'.format(ii).encode('ascii'), dumps_pyarrow((img_lr, img_hr)))
            ii += 1
            if ii % write_frequency == 0:
                print("[%d/%d]" % (ii, len(data_loader)*batch_size))
                txn.commit()
                txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(ii+1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__')) - 1
            self.keys = pickle.loads(txn.get(b'__keys__'))
            self.keys = self.keys[:-1]

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        return pickle.loads(byteflow)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def load_data(train_path, test_path, mode='train'):
    # gallery_dataset = ImageFolderLMDB('./model/HR_gallery_embedding_target.lmdb')
    # gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=1)

    print("Loading dataset from %s" % train_path)
    train_dataset = ImageFolder(train_path)
    test_dataset = ImageFolder(test_path)
    print("Dataset loaded!")
    samples = []

    for img_path, label in train_dataset.samples:
        print("img path {} with label {} appended to samples.".format(img_path, label))
        samples.append((img_path, label))

    # pickle.dump(train_idx_to_class, open('./model/HR_train_idx_to_class.pkl', 'wb'))
    # pickle.dump(test_idx_to_class, open('./model/HR_test_idx_to_class.pkl', 'wb'))

    print("Number of samples: ", len(samples))
    return samples


def dumps_pyarrow(obj):
    return pickle.dumps(obj)


root = '/home/nbayat5/Desktop/VggFaces/'
train_path = root + 'train'
test_path = root + 'test'

samples = load_data(train_path, test_path, mode='train')
folder2lmdb(samples, 'VggFaces_LR_HR_Train')