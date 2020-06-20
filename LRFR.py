import os
import extract_embedding as ee
from scipy import spatial
import pickle
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from torchvision import transforms
import preprocessing
from PIL import Image
import torch


def compute_cosine(gallery_embedding, probe_embedding):
    return spatial.distance.cosine(gallery_embedding, probe_embedding)


def extract_inception_feature(aligner, facenet_preprocess, facenet, img_path):
    img = preprocessing.ExifOrientationNormalize()(Image.open(img_path).convert('RGB'))
    try:
        bbs, _ = aligner.detect(img)
    except Exception as e:
        print(e)
    if bbs is None:
        # if no face is detected
        return None

    faces = torch.stack([extract_face(img, bb) for bb in bbs])
    preprocessed_faces = facenet_preprocess(faces)
    temp = facenet(preprocessed_faces)
    embeddings = temp.detach().numpy()
    return embeddings

def compute_ranks(fake_path, hr_path, model="vggface"):
    hr_samples = pickle.load(open("{}_gallery_embeddings.pickle".format(model), "rb"))
    lr_samples = pickle.load(open("{}_probe_embeddings.pickle".format(model), "rb"))

    print("length of hr pickle file: ", len(hr_samples.keys()))
    print("length of lr pickle file: ", len(lr_samples.keys()))

    scores = []
    for rank in range(1, len(lr_samples.keys()) + 1):
        score = 0
        for filename in os.listdir(fake_path):
            cosine_distances = []
            hr_files = []
            if filename in lr_samples.keys():
                fake_embd = lr_samples[filename]
                parts = filename.split('-')
                parts[len(parts) - 1] = "14.jpg"
                GT_filename = '-'.join(parts)
                for hr_filename in os.listdir(hr_path):
                    if hr_filename in hr_samples.keys():
                        hr_files.append(hr_filename)
                        hr_embd = hr_samples[hr_filename]
                        cosine_distances.append(compute_cosine(hr_embd, fake_embd))

            indices = sorted(range(len(cosine_distances)), key=lambda i: cosine_distances[i])[:rank]
            for index in indices:
                if hr_files[index] == GT_filename:
                    score += 1
        print("Rank {} score is: {}".format(rank, score))
        scores.append(score)

    x = np.arange(1, len(lr_samples.keys()) + 1)
    plt.plot(x, scores, color="orange", label="SRGAN - {} model".format(model))
    plt.xlabel("Rank")
    plt.ylabel("Cumulative Score")
    plt.title("Cumulative Match Characteristic for AR")
    plt.legend()
    plt.savefig("Inception_ScoreRanksAR.png")


def generate_embeddings(path, type="probe", model="vggface"):
    aligner = MTCNN(prewhiten=False, keep_all=True, thresholds=[0.6, 0.7, 0.9])
    facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    samples = {}
    for idx, filename in enumerate(os.listdir(path)):
        if filename not in samples.keys():
            print(idx+1, "Filename: ", filename)
            face = os.path.join(path, filename)
            if model == "vggface":
                embd = ee.predict(face)
                samples[filename] = embd
            else:
                embd = extract_inception_feature(aligner, facenet_preprocess, facenet, face)
                if embd is None:
                    print("Could not find face on {}".format(face))
                    continue
                if embd.shape[0] > 1:
                    print("Multiple faces detected for {}, taking one with highest probability".format(face))
                    embd = embd[0, :]
                samples[filename] = embd.flatten()

    pickle.dump(samples, open("{}_{}_embeddings.pickle".format(model, type), "wb"))
    return samples


def dump_data(fake_path, hr_path, model="vggface"):
    lr_samples = generate_embeddings(fake_path, type="probe", model=model)
    hr_samples = generate_embeddings(hr_path, type="gallery", model=model)

    remove_keys = []
    for key in hr_samples.keys():
        parts = key.split('-')
        parts[len(parts) - 1] = "1.jpg"
        lr_key = '-'.join(parts)
        if lr_key not in lr_samples.keys():
            print("hr key {} will be deleted".format(key))
            remove_keys.append(key)
    for key in remove_keys:
        del hr_samples[key]

    remove_keys = []
    for key in lr_samples.keys():
        parts = key.split('-')
        parts[len(parts) - 1] = "14.jpg"
        hr_key = '-'.join(parts)
        if hr_key not in hr_samples.keys():
            print("lr key {} will be deleted".format(key))
            remove_keys.append(key)

    for key in remove_keys:
        del lr_samples[key]

    pickle.dump(lr_samples, open("{}_probe_embeddings.pickle".format(model), "wb"))
    pickle.dump(hr_samples, open("{}_gallery_embeddings.pickle".format(model), "wb"))


def main():
    model = "inception_resnet"
    if model == "vggface":
        fake_path = "/imaging/nbayat/AR/LRFR_Pairs/fake_HR_224"
        hr_path = "/imaging/nbayat/AR/LRFR_Pairs/HR_224"
    else:
        fake_path = "/imaging/nbayat/AR/LRFR_Pairs/fake_HR_224" # or fake_HR_64
        hr_path = "/imaging/nbayat/AR/LRFR_Pairs/HR_224" #or HR_64

    dump_data(fake_path, hr_path, model=model)
    compute_ranks(fake_path, hr_path, model=model)


if __name__ == "__main__":
    main()




