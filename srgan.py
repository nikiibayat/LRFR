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
from data_loader import DataLoader
import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from skimage.transform import resize

import keras.backend as K


class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 16  # Low resolution height
        self.lr_width = 16  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4  # High resolution height
        self.hr_width = self.lr_width * 4  # High resolution width
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
        self.dataset_name = 'CelebA'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

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
        u2 = deconv2d(u1)

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

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

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
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

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
                self.sample_images(epoch)
                self.generator.save('saved_model/srgan_model.h5')
        # out.close()

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2,
                                                      is_testing=True)
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


def fid_score(images1, images2):
    # load inception v3 model
    model = InceptionV3(include_top=False, input_shape=(75, 75, 3),
                        pooling='avg')
    fid = calculate_fid(model, images1, images2)
    print('FID (different): %.3f' % fid)


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_fid(model, images1, images2):
    N = len(images1)
    images1 = np.asarray(images1)
    images2 = np.asarray(images2)
    images1 = images1.reshape((N, 64, 64, 3))
    images2 = images2.reshape((N, 64, 64, 3))
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, (75, 75, 3))
    images2 = scale_images(images2, (75, 75, 3))
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def create_datasets(gan):
    gan.generator.load_weights('saved_model/VGG_saved_model/VGG16to64.h5')
    rootdir = "/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/identities"
    print("creating datasets...")

    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            path = os.path.join(rootdir, subdir)
            parts = path.split("/")
            if len(parts) != 8:
                continue

            print("Identity name: ",parts[7]) #identity name
            split_set = "Train"
            imgs_hr, imgs_lr = gan.data_loader.load_dataforIdentities(path, split_set)
            root = "/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/"
            #path_hr = os.path.join(root, "HR{}".format(split_set), parts[7].rstrip())
            #path_lr = os.path.join(root, "LR{}".format(split_set), parts[7].rstrip())
            path_srgan = os.path.join(root, "VGG_SRGAN_{}/{}".format(split_set, parts[7].rstrip()))
            #if not os.path.exists(path_hr):
             #   os.makedirs(path_hr)
            #if not os.path.exists(path_lr):
             #   os.makedirs(path_lr)
            if not os.path.exists(path_srgan):
                os.makedirs(path_srgan)
            counter = 1
            fake_hr = gan.generator.predict(imgs_lr)
            for idx in range(len(imgs_lr)):
                fake_hr[idx] = 0.5 * fake_hr[idx] + 0.5
                fake_hr[idx] = np.asarray(fake_hr[idx])
                #path_hr_write = path_hr+"/%s_%d.jpg" % (parts[7].rstrip(), counter)
                #path_lr_write = path_lr+"/%s_%d.jpg" % (parts[7].rstrip(), counter)
                path_srgan_write = path_srgan+"/%s_%d.jpg" % (parts[7].rstrip(), counter)
                #imageio.imwrite(path_hr_write, imgs_hr[idx])
                #imageio.imwrite(path_lr_write, imgs_lr[idx])
                imageio.imwrite(path_srgan_write, fake_hr[idx])
                print("image %s_%d.png saved." % (parts[7].rstrip(), counter))
                counter += 1
            """
            """
            break


if __name__ == '__main__':
    gan = SRGAN()
    # gan.train(epochs=30000, batch_size=1, sample_interval=50)

    # to create train test validation HR - LR - SRGAN datasets uncomment method below and change names
    create_datasets(gan)

    #gan.generator.load_weights('saved_model/VGG_saved_model/VGG16to64.h5')
    #celebA_path = "/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/face_recognition_LR_test"
    #lfw_path = "/home/nbayat5/Desktop/LFW/face_recognition_without_bbx/LR_Test"



    ##########PSNR - SSIM - FID ###############
    # with tf.Session() as sess:
    #     print("CelebA Tensorflow PSNR: ", np.mean(tf.image.psnr(imgs_hr,
    #     fake_hr, max_val=1).eval()))
    #     print("CelebA Tensorflow SSIM: ", np.mean(tf.image.ssim(
    #     tf.image.convert_image_dtype(imgs_hr, tf.float32),
    #     tf.image.convert_image_dtype(fake_hr, tf.float32),max_val=1).eval()))
    #     # fid_score(imgs_hr, fake_hr)
    #     gan.save_images(fake_hr, "Generated")

    # gan2 = SRGAN()
    # gan2.generator.load_weights('saved_model/VGG_saved_model/srgan_model.h5')
    # fake_hr2 = gan2.generator.predict(imgs_lr)
    # with tf.Session() as sess:
    # print("VGG Tensorflow PSNR: ", np.std(tf.image.psnr(imgs_hr,fake_hr2,
    # max_val=1).eval()))
    # print("VGG Tensorflow SSIM: ", np.std(tf.image.ssim(tf.image.convert_image_dtype(imgs_hr, tf.float32),tf.image.convert_image_dtype(fake_hr2, tf.float32),max_val=1).eval()))
    # for k in range(0,100,2):
    #     gan2.save_images(imgs_hr[k:k+2], imgs_lr[k:k+2], fake_hr2[k:k+2], "VGG",k)
