# Each of the LR faces is upscaled to 224x224 using of the SR techniques.
# (a) bicubic interpolation, (b) SCN, (c) sparse representation super-resolution (ScSR), (d) LapSRN, (e) SRGAN
from srgan import SRGAN # put srgan.py in the folder
import numpy as np
import imageio
import scipy

def upscale(gan, img_path):
    img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
    imgs_hr = []
    imgs_lr = []
    imgs_hr.append(scipy.misc.imresize(img, (224, 224)))
    imgs_lr.append(scipy.misc.imresize(img, (28, 28)))
    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    fake_hr = gan.generator.predict(imgs_lr)
    fake_hr = np.asarray(fake_hr)
    fake_hr = np.asarray(0.5 * fake_hr + 0.5)
    print("path save img: ", "./SRGAN.png")
    imageio.imwrite("./SRGAN.png", np.squeeze(fake_hr))

def main():
    gan = SRGAN()
    gan.generator.load_weights('srgan_28-28-to-224-224.h5')
    upscale(gan, "GT.jpg")


if __name__ == "__main__":
    main()
