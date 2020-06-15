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
from PIL import Image
import random
import os


# This detector is based on histogram of oriented gradients (HOG) and linear SVM
def faceDetect(imgArray):
    detector = dlib.get_frontal_face_detector()
    dets, scores, idx = detector.run(imgArray, 1, -1)
    if len(scores) != 0:
        return max(scores)
    else:
        return -1


def create_dataset(path, count):
    scores = []
    for idx, img_path in enumerate(path):
        img = imageio.imread(img_path, pilmode='RGB')
        scores.append(faceDetect(img))

    gallery_index = scores.index(max(scores))
    gallery_face = imageio.imread(path[gallery_index], pilmode='RGB')
    path_parts = path[gallery_index].split("/")
    identity = path_parts[len(path_parts) - 2]
    gallery_id = path_parts[len(path_parts) - 1]
    # plt.title("gallery face")
    # plt.imshow(gallery_face)
    # plt.show()
    dir_path = os.path.join("../LFW/LR_HR_pairs", identity)
    if not os.path.exists(dir_path):
        print("identity number {}: {} -- Directory created!".format(count, identity))
        os.makedirs(dir_path)
    gallery_id2 = identity + "_gallery.jpg"
    imageio.imwrite(os.path.join(dir_path, gallery_id2), gallery_face)

    while True:
        random_index = random.randint(0, len(path) - 1)
        if random_index != gallery_index:
            break
    probe_path = path[random_index]
    probe_img = imageio.imread(probe_path, pilmode='RGB')
    # plt.title("probe face")
    # plt.imshow(probe_img)
    # plt.show()
    probe_path_parts = probe_path.split("/")
    probe_id = probe_path_parts[len(probe_path_parts) - 2] + "_probe.jpg"
    imageio.imwrite(os.path.join(dir_path, probe_id), probe_img)


def create_AR_dataset():
    # For each AR CD finds images #1 and #14 of 50 men and 50 women
    directory_list = ["dbf1", "dbf2", "dbf3", "dbf4", "dbf5", "dbfaces6", "dbfaces7", "dbfaces8"]
    male_count_1 = 0
    male_count_14 = 0
    female_count_1 = 0
    female_count_14 = 0
    for dir in directory_list:
        directory = "/imaging/nbayat/AR/" + dir
        for filename in os.listdir(directory):
            if filename.endswith(".raw"):
                path = os.path.join(directory, filename)
                parts = filename.split(".")[0].split("-")
                number = parts[2]
                if number == "1":  # LR
                    if parts[0] == "m":
                        if male_count_1 >= 50:
                            continue
                        male_count_1 += 1
                    else:
                        if female_count_1 >= 50:
                            continue
                        female_count_1 += 1
                    print(male_count_1, ": ", path)
                    os.system(
                        "magick convert -size 768X576 -depth 8 -interlace plane rgb:" + path + " /imaging/nbayat/AR/LRFR_Pairs/" +
                        filename.split(".")[0] + ".jpg")

                elif number == "14":  # HR
                    if parts[0] == "m":
                        if male_count_14 >= 50:
                            continue
                        male_count_14 += 1
                    else:
                        if female_count_14 >= 50:
                            continue
                        female_count_14 += 1
                    print(male_count_14, ": ", path)
                    os.system(
                        "magick convert -size 768X576 -depth 8 -interlace plane rgb:" + path + " /imaging/nbayat/AR/LRFR_Pairs/" +
                        filename.split(".")[0] + ".jpg")


def main():
    # path = glob('../LFW/lfw-deepfunneled/Abdullah_Gul/*.jpg')
    """
    root = "../LFW/lfw-deepfunneled"
    count = 1
    for subdir, dirs, files in os.walk(root):
        for dir in dirs:
            path = glob("../LFW/lfw-deepfunneled/"+dir+"/*.jpg")
            create_dataset(path, count)
            count += 1
    """
    create_AR_dataset()


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
