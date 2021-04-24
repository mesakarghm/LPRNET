import os
import json
import random
from tqdm import tqdm
import cv2
'''
from augmentation import SequentialTransform, RandomScalingAndRotation, RandomTranslation, ColorDistorion, \
    RandomShearing, GaussianBlur
'''

# Remove this one later
def resize_and_normalize(img):
    width = 94
    height = 24
    return cv2.resize(img, (width, height)) / 255


def resize(img):
    width = 94
    height = 24
    return cv2.resize(img, (width, height))


def normalize(img):
    return img / 255


def augmentation(img):
    # translation = RandomTranslation((-0.1, 0.1), (-0.1, 0.1))
    # rotation_and_scaling = RandomScalingAndRotation((-10, 10), (0.9, 1.1))
    # shearing = RandomShearing((-0.1, 0.1), (-0.1, 0.1))
    # color_distortion = ColorDistorion()
    # blurring = GaussianBlur(0.1)
    #
    # h, w = img.shape[:2]
    #
    # sequential = SequentialTransform([random.choice([translation, rotation_and_scaling, shearing])],
    #                                  [color_distortion, blurring],
    #                                  (w, h))
    # img_aug, _ = sequential.apply_transform(img, points=None, border_mode=cv2.BORDER_CONSTANT)
    return img


class Loader:
    def __init__(self, labelfile, img_dir, norm_func=normalize, augmen_func=augmentation,
                 preproc_func=resize, load_all=False):
        self.img_dir = img_dir
        self.preproc_func = preproc_func
        self.norm_func = norm_func
        self.augmen_func = augmen_func
        self.load_all = load_all

        with open(labelfile, "r") as f:
            obj = json.load(f)
        self.data = obj["data"]
        self.class_names = obj["class_names"]
        self.lookup = {name: i for i, name in enumerate(self.class_names)}

        if load_all:
            self.images = []
            for img_data in tqdm(self.data):
                path = os.path.join(self.img_dir, img_data["images"]["filename"])
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("Cannot open image at", path)
                img = self.preproc_func(img)
                self.images.append(img)

    def get_num_chars(self):
        return len(self.class_names)

    def parse_label(self, label):
        text = []
        for idx in label:
            text.append(self.class_names[idx])
        return "".join(text)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.data):
            self.i = 0
        img_data = self.data[self.i]

        if self.load_all:
            img = self.images[self.i]
        else:
            img = cv2.imread(os.path.join(self.img_dir, img_data["images"]["filename"]))
            img = self.preproc_func(img)

        if self.augmen_func is not None:
            img = self.augmen_func(img)
        img = self.norm_func(img)

        text = img_data["annotations"]["text"]
        label = [self.lookup[c] for c in text]
        self.i += 1
        return img, label, [len(label)]

    def __len__(self):
        return len(self.data)

    def __call__(self, *args, **kwargs):
        return self