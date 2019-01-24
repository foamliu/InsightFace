import pickle
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import pickle_file, batch_size, num_workers
from models import data_transforms
from utils import align_face


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        samples = data['samples']

        num_samples = len(samples)
        num_train = num_samples
        # num_train = int(train_split * num_samples)

        if split == 'train':
            self.samples = samples[:num_train]
            self.transformer = data_transforms['train']

        # else:
        #     self.samples = samples[num_train:]
        #     self.transformer = data_transforms['val']

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['full_path']
        landmarks = sample['landmarks']
        img = align_face(full_path, landmarks)
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        class_id = sample['class_id']
        return img, class_id

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        np.random.shuffle(self.samples)


def show_align():
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = random.sample(data['samples'], 10)

    sample_inputs = []
    for i, sample in enumerate(samples):
        full_path = sample['full_path']
        subject = sample['subject']
        landmarks = sample['landmarks']
        raw = cv.imread(full_path)
        raw = cv.resize(raw, (224, 224))
        img = align_face(full_path, landmarks)
        filename = 'images/{}_raw.jpg'.format(i)
        cv.imwrite(filename, raw)
        filename = 'images/{}_img.jpg'.format(i)
        cv.imwrite(filename, img)


if __name__ == "__main__":
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    print(batch_size)
    print(len(train_dataset))
    print(len(train_loader))
