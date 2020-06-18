# Get HR-LR pair
# Super resolve LR face
# Extract feature vectors of both super-resolved face and original HR
# Use cosine distance to match score between feature vectors
import os
import extract_embedding as ee
from scipy import spatial
import pickle


def compute_cosine(gallery_embedding, probe_embedding):
    return spatial.distance.cosine(gallery_embedding, probe_embedding)


def main():
    fake_path = "/imaging/nbayat/AR/LRFR_Pairs/fake_HR"
    hr_path = "/imaging/nbayat/AR/LRFR_Pairs/HR"

    hr_samples = pickle.load(open("gallery_embeddings.pickle","rb"))
    lr_samples = pickle.load(open("probe_embeddings.pickle","rb"))
    """
    samples = {}
    path = fake_path
    for idx, filename in enumerate(os.listdir(path)):
        if filename not in samples.keys():
            print(idx+1, "Filename: ", filename)
            face = os.path.join(path, filename)
            embd = ee.predict(face)
            samples[filename] = embd
    pickle.dump(samples, open("probe_embeddings.pickle", "wb"))
    print("----All probe embeddings are saved in hr_samples----")
    """

    print("length of hr pickle file: ", len(hr_samples.keys()))
    print("length of lr pickle file: ", len(lr_samples.keys()))
    rank = 2
    score = 0
    for filename in os.listdir(fake_path):
        print("Probe Image: ", filename)
        cosine_distances = []
        hr_files = []
        fake_embd = lr_samples[filename]
        parts = filename.split('-')
        parts[len(parts) - 1] = "14.jpg"
        GT_filename = '-'.join(parts)
        for hr_filename in os.listdir(hr_path):
            hr_files.append(hr_filename)
            hr_embd = hr_samples[hr_filename]
            cosine_distances.append(compute_cosine(hr_embd, fake_embd))

        indices = sorted(range(len(cosine_distances)), key=lambda i: cosine_distances[i])[:rank]
        print("indices: ", indices)
        for index in indices:
            if hr_files[index] == GT_filename:
                print("probe {} matched with {}".format(filename, hr_files[index]))
                score += 1

    print("Rank {} score is: {}".format(rank, score))



if __name__ == "__main__":
    main()
