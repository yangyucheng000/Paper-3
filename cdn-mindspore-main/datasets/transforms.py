import random

import PIL
import numpy as np
import cv2

import mindspore
from mindspore import ops, Tensor
import mindspore.dataset.vision as vision


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.T  
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 
    w = x1 - x0
    h = y1 - y0
    cxcywh = np.stack((cx, cy, w, h), axis=-1)  
    return cxcywh  


def crop(image, target, region):
    crop_op = vision.Crop(*region)
    cropped_image = crop_op(image)

    target = target.copy()
    i, j, h, w = region

    target["size"] = Tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = Tensor([w, h], dtype=mindspore.float32)
        cropped_boxes = boxes - Tensor([j, i, j, i])
        cropped_boxes = ops.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = ops.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        else:
            keep = target['masks'].flatten().any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    horizontalflip = vision.HorizontalFlip()
    flipped_image = horizontalflip(image)

    w, h = image.shape

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * Tensor([-1, 1, -1, 1]) + Tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return ops.interpolate(input, size, scale_factor, mode, align_corners)


def resize(image_set, image, target, size, max_size=None):

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h, _ = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    h, w, _ = image.shape

    size = get_size(image.shape, size, max_size)
    rescaled_image = cv2.resize(image, size, cv2.INTER_CUBIC)

    if image_set == 'val':
        return rescaled_image, target

    ratio_width, ratio_height = float(size[1])/float(w), float(size[0])/float(h)

    target = target.copy()
    if "boxes" in target:
        boxes = target['boxes']
        boxes = boxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        target['boxes'] = boxes

    nh, nw = size
    target["size"] = np.array([nh, nw])

    if "masks" in target:
        target['masks'] = cv2.resize(target['masks'][:, None].float(), size, interpolation=cv2.INTER_NEAREST)[:, 0] > 0.5

    return rescaled_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = vision.RandomCrop.get_params(img, [h, w]) # to be implemented
        return crop(img, target, region)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, image_set, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.image_set = image_set
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(self.image_set, img, target, size, self.max_size)


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return np.array(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: np.ndarray, target=None):
        image = image / 255.0
        image = (image - self.mean) / self.std
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturatio=0, hue=0):
        self.color_jitter = vision.RandomColorAdjust(brightness, contrast, saturatio, hue)

    def __call__(self, img, target):
        return self.color_jitter(img), target


class OutData(object):
    def __init__(self, max_size=1333):
        self.max_size = max_size

    def __call__(self, img, target):
        (tensor, mask), target = self.pad(img, target)
        tensor = tensor.transpose(2, 0, 1)
        return (tensor, mask), target
    
    def pad(self, img, target):
        h, w, c = img.shape
        tensor = np.zeros((self.max_size, self.max_size, c), dtype=np.float32)
        tensor[:h, :w, :] = img
        mask = np.ones((self.max_size, self.max_size), dtype=np.bool_)
        mask[:h, :w] = False
        return (tensor, mask), target
