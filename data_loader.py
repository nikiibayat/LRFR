# In this file a HR-LR pair will be returned from chosen dataset.

# HR size : 224x224 -bicubic interpolation
# LR size : a)21x15 b)16x12 c)11x8  -bicubic interpolation

# AR dataset: original LR and HR face images are taken from
# face images #01 and #14 respectively of each subject. Since
# the AR dataset has 100 subjects, we obtained 100 LR–HR
# pairs of face images for this experiment.


import dlib
import imageio
import cv2
from glob import glob
import matplotlib.pyplot as plt
import random
import os

# This detector is based on histogram of oriented gradients (HOG) and linear SVM
def faceDetect(imgArray):
    detector = dlib.get_frontal_face_detector()
    dets = detector(imgArray, 1)
    if len(dets) >= 1:
        return True
    else:
        return False


def main():
    dsize = 40
    flag = False
    # path = glob('../LFW/Abdullah_Gul/*.jpg')
    path = glob('../LFW/Abid_Hamid_Mahmud_Al-Tikriti/*.jpg')
    # path = glob('../LFW/Michael_Chang/*.jpg')
    imgs = []
    while (not flag):
        print("size: ({}, {})".format(dsize, dsize))
        for idx, img_path in enumerate(path):
            img = imageio.imread(img_path, pilmode='RGB')
            img_resized = cv2.resize(img, dsize=(dsize, dsize), interpolation=cv2.INTER_AREA)
            if faceDetect(img_resized):
                img_path = img_path.split("/")
                print(img_path[len(img_path)-1])
                imgs.append((img, img_path[len(img_path)-2], img_path[len(img_path)-1]))
                flag = True
            elif idx == len(path)-1:
                if not flag:
                    dsize += 5
                else:
                    break

    gallery_face, identity, id = imgs[random.randint(0, len(imgs))][0]
    plt.imshow(gallery_face)
    plt.show()
    dir_path = os.path.join("../LFW/LR_HR_pairs",identity)
    if not os.path.exists():
        print("Directory {} created!".format(identity))
        os.makedirs(identity)



if __name__ == "__main__":
    main()

# img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
# img_hr = scipy.misc.imresize(img, (224, 224))
# img_lr = scipy.misc.imresize(img, (21, 15))
# # If training => do random flip
# if name == "train" and np.random.random() < 0.5:
#     img_hr = np.fliplr(img_hr)
#     img_lr = np.fliplr(img_lr)
#
# img_hr = np.array(img_hr) / 127.5 - 1.
# img_lr = np.array(img_lr) / 127.5 - 1.
