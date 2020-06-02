# In this file a HR-LR pair will be returned from chosen dataset.

# HR size : 224x224 -bicubic interpolation
# LR size : a)21x15 b)16x12 c)11x8  -bicubic interpolation

# AR dataset: original LR and HR face images are taken from
# face images #01 and #14 respectively of each subject. Since
# the AR dataset has 100 subjects, we obtained 100 LR–HR
# pairs of face images for this experiment.


import dlib
import imageio


def faceDetect(imgArray):
    detector = dlib.get_frontal_face_detector()
    # Run the face detector, upsampling the image 1 time to find smaller faces.
    dets = detector(imgArray, 1)
    if len(dets) >= 1:
        return True
    else:
        return False


def main():
    num = input("please enter number of the image: ")
    while num != str(-1):
        img_path = "Abdullah_Gul/Abdullah_Gul_{}.jpg".format(num)
        img = imageio.imread(img_path, pilmode='RGB')
        print(faceDetect(img))
        num = input("please enter number of the image: ")


if __name__ == "__main__":
    main()
