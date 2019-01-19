import os
import pickle
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import *
from mtcnn.detector import detect_faces


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data')
    zip_ref.close()


def get_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)
        width, height = img.size

        if len(bounding_boxes) == 1:
            x1, y1, x2, y2 = bounding_boxes[0][0], bounding_boxes[0][1], bounding_boxes[0][2], bounding_boxes[0][3]
            if x1 < 0 or x1 >= width or x2 < 0 or x2 >= width or y1 < 0 or y1 >= height or y2 < 0 or y2 >= height or x1 >= x2 or y1 >= y2:
                return False, None, None

            landmarks = [int(round(x)) for x in landmarks[0]]
            is_valid = (x2 - x1) > width / 10 and (y2 - y1) > height / 10

            return is_valid, (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), landmarks

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None, None


if __name__ == "__main__":
    if not os.path.isdir('data/CASIA-WebFace'):
        extract('data/CASIA-WebFace.zip')

    samples = []
    subjects = [d for d in os.listdir('data/CASIA-WebFace') if os.path.isdir(os.path.join('data/CASIA-WebFace', d))]
    for sub in tqdm(subjects):
        folder = os.path.join('data/CASIA-WebFace', sub)
        # print(folder)
        files = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        # print(files)
        for file in files:
            filename = os.path.join(folder, file)
            # print(filename)
            is_valid, face_location, landmarks = get_face_attributes(filename)
            if is_valid:
                samples.append(
                    {'subject': sub, 'full_path': filename, 'face_location': face_location, 'landmarks': landmarks})

    np.random.shuffle(samples)
    f = open(pickle_file, 'wb')
    save = {
        'samples': samples
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
