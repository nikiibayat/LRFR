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
    try:
        gallery_index, probe_index = random.sample(range(0, len(path) - 1), 2)
        gallery_face = imageio.imread(path[gallery_index], pilmode='RGB')
        path_parts = path[gallery_index].split("/")
        identity = path_parts[len(path_parts) - 2].rstrip()
        gallery_id = identity + '-14.jpg'
        gallery_path = os.path.join("../Desktop/celebA/LR_HR_pairs", gallery_id)
        if not os.path.exists(gallery_path):
            print("gallery image ", gallery_path)
            imageio.imwrite(gallery_path, gallery_face)
            probe_path = path[probe_index]
            probe_img = imageio.imread(probe_path, pilmode='RGB')
            probe_path_parts = probe_path.split("/")
            probe_id = probe_path_parts[len(probe_path_parts) - 2].rstrip() + '-1.jpg'
            imageio.imwrite(os.path.join("../Desktop/celebA/LR_HR_pairs", probe_id), probe_img)
    except:
        return


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
                    os.system(
                        "magick convert -size 768X576 -depth 8 -interlace plane rgb:" + path + " /imaging/nbayat/AR/LRFR_Pairs/" +
                        filename.split(".")[0] + ".jpg")

                elif number == "14":  # HR
                    os.system(
                        "magick convert -size 768X576 -depth 8 -interlace plane rgb:" + path + " /imaging/nbayat/AR/LRFR_Pairs/" +
                        filename.split(".")[0] + ".jpg")


def main():

    root = "../Desktop/celebA/identities/"
    # root = "../Desktop/LFW/lfw-deepfunneled/"
    count = 1
    for subdir, dirs, files in os.walk(root):
        for dir in dirs:
            path = glob(root+dir+"/*.jpg")
            create_dataset(path, count)
            count += 1

    # create_AR_dataset()


if __name__ == "__main__":
    main()
