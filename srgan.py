"""
Instrustion on running the script:
1. Download the dataset(vggfaces2 or celebA)
2. Save the folder 'img_align_celeba' or 'img_vgg' to 'datasets/'
4. Run the sript using command 'python3 srgan.py'
"""

from __future__ import print_function, division
import scipy
import scipy.misc
import tensorflow as tf
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import \
    InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import imageio
plt.switch_backend('agg')
import sys
import numpy as np
import os
import os.path as osp
import pickle
import lmdb
from torch.utils.data import Dataset
import random
import torch
# from data_loader import DataLoader


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


def load_data(vgg_dataset, batch_size, is_testing=False):
    imgs_hr = []
    imgs_lr = []
    batch_images = []
    length = len(vgg_dataset)
    randomList = random.sample(range(0, length-1), batch_size)
    for idx in range(len(randomList)):
        batch_images.append(vgg_dataset.__getitem__(idx)[0])

    for img in batch_images:
        img_hr = scipy.misc.imresize(img, (224, 224))
        img_lr = scipy.misc.imresize(img, (21, 15))

        # If training => do random flip
        if not is_testing and np.random.random() < 0.5:
            img_hr = np.fliplr(img_hr)
            img_lr = np.fliplr(img_lr)

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    return imgs_hr, imgs_lr


class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 21  # Low resolution height
        self.lr_width = 15  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = 224 # High resolution height
        self.hr_width = 224 # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the
        # high resolution
        # and the generated high resolution images and minimize the mse
        # between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'VggFace2'
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                               img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator - freeze
        # weights for discriminator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        # with compile we configure training process
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features
        extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at:
        # https://github.com/keras-team/keras/blob/master/keras/applications
        # /vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(
                layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u3 = deconv2d(u1) # I added
        u2 = deconv2d(u3)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same',
                        activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(
                layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        # out = open('output.txt','w')
        start_time = datetime.datetime.now()
        # vgg_dataset = ImageFolderLMDB('/home/nbayat5/scratch/vggface2/VggFaces_LR_HR_Train.lmdb')
        vgg_dataset = ImageFolderLMDB('/imaging/nbayat/VggFaceLmdb/VggFaces_LR_HR_Train.lmdb')

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            # imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            imgs_hr, imgs_lr = load_data(vgg_dataset, batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated =
            # Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            # imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            imgs_hr, imgs_lr = load_data(vgg_dataset, batch_size)

            # The generators want the discriminators to label the generated
            # images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr],
                                                  [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print(self.combined.metrics_names)
            print("%d time: %s" % (epoch, elapsed_time), "g_loss: ", g_loss)
            # out.write(str(self.combined.metrics_names))
            # out.write("%d time: %s" % (epoch, elapsed_time),"g_loss: ",
            # str(g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, vgg_dataset)
                self.generator.save('srgan_21-15-to-224-224.h5')
        # out.close()

    def sample_images(self, epoch, vgg_dataset):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        # imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        imgs_hr, imgs_lr = load_data(vgg_dataset, batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig(
                'images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()

    def save_images(self, images, folder_name):
        os.makedirs('Results/%s' % folder_name, exist_ok=True)

        images = 0.5 * images + 0.5

        # Save generated images and the high resolution originals
        cnt = 0
        for image in images:
            image = np.asarray(image)
            path = "Results/%s/%d.png" % (folder_name, cnt)
            im = imageio.imwrite(path, image)
            cnt += 1

        plt.close()




if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=30000, batch_size=1, sample_interval=50)


