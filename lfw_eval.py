import math
import os
import pickle
import tarfile

import cv2 as cv
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from config import *
from models import data_transforms
from utils import align_face, get_face_all_attributes, draw_bboxes


def extract(filename):
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
        is_valid, bounding_boxes, landmarks = get_face_all_attributes(filename)

        if is_valid:
            samples.append(
                {'class_id': class_id, 'subject': sub, 'full_path': filename, 'bounding_boxes': bounding_boxes,
                 'landmarks': landmarks})

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
            angles.append('{} {}\n'.format(theta, is_same))

    with open('data/angles.txt', 'w') as file:
        file.writelines(angles)


def visualize(angles_file):
    with open(angles_file) as file:
        lines = file.readlines()

    ones = []
    zeros = []
    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
            if angle > 75:
                wrong += 1
        else:
            zeros.append(angle)
            if angle <= 75:
                wrong += 1

    import numpy
    from matplotlib import pyplot

    bins = numpy.linspace(0, 180, 181)

    pyplot.hist(zeros, bins, alpha=0.5, label='0')
    pyplot.hist(ones, bins, alpha=0.5, label='1')
    pyplot.legend(loc='upper right')
    pyplot.show()

    print('Accuracy: {}%'.format(100 - wrong / 6000 * 100))


def show_bboxes(folder):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']
    for sample in tqdm(samples):
        full_path = sample['full_path']
        bounding_boxes = sample['bounding_boxes']
        landmarks = sample['landmarks']
        img = cv.imread(full_path)
        img = draw_bboxes(img, bounding_boxes, landmarks)
        filename = os.path.basename(full_path)
        filename = os.path.join(folder, filename)
        cv.imwrite(filename, img)


if __name__ == "__main__":
    filename = 'data/lfw-funneled.tgz'
    if not os.path.isdir('data/lfw_funneled'):
        print('Extracting {}...'.format(filename))
        extract(filename)

    pickle_file = 'data/lfw_funneled.pkl'
    if not os.path.isfile(pickle_file):
        print('Processing {}...'.format(pickle_file))
        process()
    else:
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        lines = []
        samples = data['samples']
        for sample in samples:
            line = sample['full_path'] + '\n'
            line.replace('\\', '/')
            lines.append(line)

        with open('data/full_path.txt', 'w') as file:
            file.writelines(lines)

    angles_file = 'data/angles.txt'
    if not os.path.isfile(angles_file):
        print('Evaluating {}...'.format(angles_file))
        evaluate()

    print('Visualizing {}...'.format(angles_file))
    visualize(angles_file)

    folder = 'data/lfw_with_bboxes'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    print('Drawing boxes...')
    show_bboxes(folder)
