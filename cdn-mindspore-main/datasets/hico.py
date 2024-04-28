from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import datasets.transforms as T

valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90)

valid_verb_ids = list(range(1, 118))

class HICODetection:

    def __init__(self, img_set, img_folder, anno_file, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)

        self.num_queries = num_queries
        self._valid_obj_ids = valid_obj_ids
        self._valid_verb_ids = valid_verb_ids

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        target = {}
        target['annotations'] = img_anno['annotations']
        target['orig_size'] =np.array([int(h), int(w)])
        target['size'] =np.array([int(h), int(w)])

        target['filename'] = img_anno['file_name']
        target['hoi_annotation'] = img_anno['hoi_annotation']
        target['id'] = idx

        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        counts = defaultdict(lambda: 0)
        for img_anno in annotations:
            hois = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                           self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                           self._valid_verb_ids.index(hoi['category_id']))
                counts[triplet] += 1
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


def make_hico_transforms(image_set, ratio=1.0):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(v / ratio) for v in scales]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(image_set, scales, max_size=int(1333/ratio)),
                T.Compose([
                    T.RandomResize(image_set, [400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(image_set, scales, max_size=int(1333/ratio)),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize(image_set, [int(800/ratio)], max_size=int(1333/ratio)),
            normalize,
            T.OutData()
        ])

    raise ValueError(f'unknown {image_set}')


def preprocess_fn(args, img_set, img, target):
    local_valid_verb_ids = valid_verb_ids
    local_valid_obj_ids = valid_obj_ids
    
    boxes = [obj['bbox'] for obj in target['annotations']]
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)

    if img_set == 'train':
        classes = [(i, local_valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(target['annotations'])]
    else:
        classes = [local_valid_obj_ids.index(obj['category_id']) for obj in target['annotations']]
    classes = np.array(classes, dtype=np.int64)

    transforms = make_hico_transforms(img_set, ratio=args.img_ratio)

    if img_set == 'train':
        w, h = img.shape
        np.clip(boxes[:, 0::2], 0, w, out=boxes[:, 0::2])
        np.clip(boxes[:, 1::2], 0, h, out=boxes[:, 1::2])
        keep = np.bitwise_and(boxes[:, 3] > boxes[:, 1], boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target['boxes'] = boxes
        target['labels'] = classes
        target['iscrowd'] = np.array([0 for _ in range(boxes.shape[0])])
        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        img, target = transforms(img, target)

        kept_box_indices = [label[0] for label in target['labels']]

        target['labels'] = target['labels'][:, 1]

        obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
        sub_obj_pairs = []
        for hoi in target['hoi_annotation']:
            if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                continue
            sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
            if sub_obj_pair in sub_obj_pairs:
                verb_labels[sub_obj_pairs.index(sub_obj_pair)][local_valid_verb_ids.index(hoi['category_id'])] = 1
            else:
                sub_obj_pairs.append(sub_obj_pair)
                obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                verb_label = [0 for _ in range(len(local_valid_verb_ids))]
                verb_label[local_valid_verb_ids.index(hoi['category_id'])] = 1
                sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                verb_labels.append(verb_label)
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)

        if len(sub_obj_pairs) == 0:
            target['obj_labels'] = np.zeros((0,), dtype=np.int64)
            target['verb_labels'] = np.zeros((0, len(local_valid_verb_ids)), dtype=np.float32)
            target['sub_boxes'] = np.zeros((0, 4), dtype=np.float32)
            target['obj_boxes'] = np.zeros((0, 4), dtype=np.float32)
            target['matching_labels'] = np.zeros((0,), dtype=np.int64)
        else:
            target['obj_labels'] = np.stack(obj_labels)
            target['verb_labels'] = np.array(verb_labels, dtype=np.float32)
            target['sub_boxes'] = np.stack(sub_boxes)
            target['obj_boxes'] = np.stack(obj_boxes)
            target['matching_labels'] = np.ones_like(target['obj_labels'])
    else:
        target['boxes'] = boxes
        target['labels'] = classes

        (tensor, mask), target = transforms(img, target)

        hois = []
        for hoi in target['hoi_annotation']:
            hois.append((hoi['subject_id'], hoi['object_id'], local_valid_verb_ids.index(hoi['category_id'])))
        target['hois'] = np.array(hois, dtype=np.int64)
    return tensor, mask, target


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
        'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, num_queries=args.num_queries)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
