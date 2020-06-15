# Each of the LR faces is upscaled to 224x224 using of the SR techniques.
# (a) bicubic interpolation, (b) SCN, (c) sparse representation super-resolution (ScSR), (d) LapSRN, (e) SRGAN
from srgan import SRGAN # put srgan.py in the folder
import numpy as np
import imageio
import scipy


def upscale(gan, img_LR_path, img_HR_path):
    img_lr = scipy.misc.imread(img_LR_path, mode='RGB').astype(np.float)
    img_hr = scipy.misc.imread(img_HR_path, mode='RGB').astype(np.float)
    imgs_hr = []
    imgs_lr = []
    imgs_hr.append(scipy.misc.imresize(img_hr, (224, 224)))
    imgs_lr.append(scipy.misc.imresize(img_lr, (28, 28)))
    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    fake_hr = gan.generator.predict(imgs_lr)
    fake_hr = np.asarray(fake_hr)
    fake_hr = np.asarray(0.5 * fake_hr + 0.5)
    parts = img_LR_path.split("/")
    fake_path = "/imaging/nbayat/AR/LRFR_Pairs/fake_HR/{}".format(parts[len(parts)-1])
    imageio.imwrite(fake_path, np.squeeze(fake_hr))
    parts = img_HR_path.split("/")
    hr_path = "/imaging/nbayat/AR/LRFR_Pairs/HR/{}.jpg".format(parts[len(parts)-1])
    imageio.imwrite(hr_path, imgs_hr[0])
    return np.squeeze(fake_hr), imgs_hr

def main():
    gan = SRGAN()
    gan.generator.load_weights('srgan_28-28-to-224-224.h5')
    # upscale(gan, "/imaging/nbayat/AR/LRFR_Pairs/m-015-1.jpg", "/imaging/nbayat/AR/LRFR_Pairs/m-015-14.jpg")

    img_path = "/home/nbayat5/Desktop/VggFaces/train/n009279/0285_03.jpg"
    upscale(gan, img_path, img_path) #vgg train

if __name__ == "__main__":
    main()
