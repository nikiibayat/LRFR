# Each of the LR faces is upscaled to 224x224 using of the SR techniques.
# (a) bicubic interpolation, (b) SCN, (c) sparse representation super-resolution (ScSR), (d) LapSRN, (e) SRGAN
from srgan import SRGAN # put srgan.py in the folder
import numpy as np
import imageio
import scipy
import os


def upscale(gan,root_path, img_LR_path, img_HR_path):
    lr = 28
    hr = 224
    img_lr = scipy.misc.imread(img_LR_path, mode='RGB').astype(np.float)
    img_hr = scipy.misc.imread(img_HR_path, mode='RGB').astype(np.float)
    imgs_hr = []
    imgs_lr = []
    imgs_hr.append(scipy.misc.imresize(img_hr, (hr, hr)))
    imgs_lr.append(scipy.misc.imresize(img_lr, (lr, lr)))
    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    fake_hr = gan.generator.predict(imgs_lr)

    fake_hr = 0.5 * fake_hr + 0.5
    imgs_hr = 0.5 * imgs_hr + 0.5

    parts = img_LR_path.split("/")
    fake_path = root_path + "fake_HR_{}/{}".format(str(hr), parts[len(parts)-1])

    imageio.imwrite(fake_path, np.squeeze(fake_hr))
    parts = img_HR_path.split("/")
    hr_path = root_path + "HR_{}/{}".format(str(hr), parts[len(parts)-1])
    imageio.imwrite(hr_path, imgs_hr[0])
    return np.squeeze(fake_hr), imgs_hr

def main():
    gan = SRGAN()
    gan.generator.load_weights('srgan_28-28-to-224-224.h5') # trained on vgg train
    # gan.generator.load_weights('/home/nbayat5/Desktop/srgan/saved_model/VGG_saved_model/VGG16to64.h5')

    # root_path = "/imaging/nbayat/AR/LRFR_Pairs"
    root_path = "/home/nbayat5/Desktop/LFW/LR_HR_pairs/"
    for filename in os.listdir(root_path):
        parts = filename.split('-')
        if parts[len(parts)-1] == "1.jpg":
            img_LR_path = os.path.join(root_path, filename)
            parts[len(parts) - 1] = "14.jpg"
            HR_filename = '-'.join(parts)
            img_HR_path = os.path.join(root_path, HR_filename)
            if os.path.exists(img_HR_path):
                print(img_LR_path)
                upscale(gan,root_path, img_LR_path, img_HR_path)


if __name__ == "__main__":
    main()
# select_fifty()

def select_fifty():
    root_path = "/imaging/nbayat/AR/LRFR_Pairs/fake_HR"
    male_count = 0
    female_count = 0
    for filename in os.listdir(root_path):
        img_LR_path = os.path.join(root_path, filename)
        parts = filename.split('-')
        parts[len(parts) - 1] = "14.jpg"
        HR_filename = '-'.join(parts)
        img_HR_path = os.path.join("/imaging/nbayat/AR/LRFR_Pairs/HR", HR_filename)
        if parts[0] == "m":
            if male_count >= 50:
                print(filename+" is removed.")
                os.system("rm {}".format(img_LR_path))
                os.system("rm {}".format(img_HR_path))
            male_count += 1
        else:
            if female_count >= 50:
                print(filename + " is removed.")
                os.system("rm {}".format(img_LR_path))
                os.system("rm {}".format(img_HR_path))
            female_count += 1


