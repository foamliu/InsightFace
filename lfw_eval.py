import math
import os
import pickle
import tarfile

import numpy as np
from torchvision import transforms
from tqdm import tqdm

from config import *
from models import data_transforms
from utils import align_face, get_face_attributes


def extract(filename):
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


def process():
    subjects = [d for d in os.listdir('data/lfw_funneled') if os.path.isdir(os.path.join('data/lfw_funneled', d))]
    assert (len(subjects) == 5749), "Number of subjects is: {}!".format(len(subjects))

    file_names = []
    for i in range(len(subjects)):
        sub = subjects[i]
        folder = os.path.join('data/lfw_funneled', sub)
        files = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        for file in files:
            filename = os.path.join(folder, file)
            file_names.append({'filename': filename, 'class_id': i, 'subject': sub})

    assert (len(file_names) == 13233), "Number of files is: {}!".format(len(file_names))

    samples = []
    for item in tqdm(file_names):
        filename = item['filename']
        class_id = item['class_id']
        sub = item['subject']
        is_valid, landmarks = get_face_attributes(filename)
        if is_valid:
            samples.append(
                {'class_id': class_id, 'subject': sub, 'full_path': filename, 'landmarks': landmarks})

    with open(pickle_file, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)


def get_image(samples, transformer, file):
    filtered = [sample for sample in samples if file in sample['full_path'].replace('\\', '/')]
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), file)
    sample = filtered[0]
    full_path = sample['full_path']
    landmarks = sample['landmarks']
    img = align_face(full_path, landmarks)
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)
    return img


def evaluate():
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    filename = 'data/lfw_test_pair.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()

    transformer = data_transforms['val']

    angles = []

    with torch.no_grad():
        for line in tqdm(lines):
            tokens = line.split()
            file0 = tokens[0]
            img0 = get_image(samples, transformer, file0)
            file1 = tokens[1]
            img1 = get_image(samples, transformer, file1)
            imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
            imgs[0] = img0
            imgs[1] = img1
            output = model(imgs)
            # print('output.size(): ' + str(output.size()))
            feature0 = output[0].cpu().numpy()
            feature1 = output[1].cpu().numpy()
            x0 = feature0 / np.linalg.norm(feature0)
            x1 = feature1 / np.linalg.norm(feature1)
            cosine = np.dot(x0, x1)
            # print('cosine: ' + str(cosine))
            theta = math.acos(cosine)
            theta = theta * 180 / math.pi
            is_same = tokens[2]
            angles.append('{} {}'.format(theta, is_same))

    with open('data/angles.txt', 'w') as file:
        file.writelines(angles)


if __name__ == "__main__":
    if not os.path.isdir('data/lfw_funneled'):
        extract('data/lfw-funneled.tgz')

    pickle_file = 'data/lfw_funneled.pkl'
    if not os.path.isfile(pickle_file):
        process()

    angles_file = 'data/angles.txt'
    if not os.path.isfile(angles_file):
        evaluate()
