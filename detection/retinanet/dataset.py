# clip attribute classification
import os
import json
import torch
from PIL import Image
import pandas as pd
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
import skimage
import ipdb
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from retinanet.tokenizers import Tokenizer

IMAGE_FOLDER_PATH = './data/MIMIC-CXR/2.0.0/'
BOX_ANNOTATION_CSV_PATH = './MS_CXR_Local_Alignment_with_Attibutes_woPosition.csv'
BOX_REPORT_CSV_PATH = './MS_CXR_Local_Alignment_with_Reports.csv'
BOX_ANNOTATION_JSON_PATH = './MS_CXR_Local_Alignment_v1.0.0.json'

RETINANET_PRED_BOX_PATH = './Detection_boxes.npy'
DETECTION_IMAGE_PATH = './Detection_image_path_sort.npy'
DETECTION_REPORT_PATH = ',/Detection_report_sort.npy'


class BaseDataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.image_folder = IMAGE_FOLDER_PATH
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.split = split
        tokenizer = Tokenizer(args)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = args.max_seq_length
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.Resizer = Resizer_img()
        self.args = args

        BOXES = np.load(RETINANET_PRED_BOX_PATH, allow_pickle=True)
        IMAGES = np.load(DETECTION_IMAGE_PATH, allow_pickle=True)
        REPORT = np.load(DETECTION_REPORT_PATH, allow_pickle=True)
        REPORT_annotation = pd.read_csv(args.report_ann_path)
        self.BOXES = BOXES # predetected boxes from teacher detection model (for reduce the memory)
        self.IMAGES = IMAGES
        self.report_all = REPORT
        self.report_annot = REPORT_annotation # annotation from generation model (for reduce the memory)
        self.report_annot = self.report_annot.fillna(0)

        # unlabeled data
        self.examples = self.ann[self.split]

        self.data = pd.read_csv(BOX_ANNOTATION_CSV_PATH)
        self.report = pd.read_csv(BOX_REPORT_CSV_PATH)
        with open(BOX_ANNOTATION_JSON_PATH, 'r') as obj:
            json_data = json.load(obj)
        self.images = json_data['images'][split]

        annotations = json_data['annotations'][split]
        X = [x['id'] for x in annotations]
        y = [c['category_id'] for c in annotations]
        
        self.unlabeled_idxs = list(range(len(self.IMAGES)))
        self.labeled_idxs = list(range(len(self.IMAGES), len(self.IMAGES)+len(self.images)))

        self.class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]

    def __len__(self):
        return len(self.IMAGES) + len(self.images)
        # return len(self.examples) + len(self.images)

    def num_classes(self):
        return 8

    def name_to_label(self, name):
        class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        return class_name.index(name)

    def label_to_name(self, label):
        class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        return class_name[label]


class MimiccxrSingleImageDataset(BaseDataset): # used for train_mimin_teacher_report_nms.py
    def __getitem__(self, idx):
        if idx < len(self.IMAGES):
            image_path = self.IMAGES[idx]
            image = np.array(Image.open((image_path)).convert('RGB'))

            image_path_split = image_path.split('/') 
            subject_id, study_id = image_path_split[7][1:], image_path_split[8][1:]

            annotation = np.array([np.array(x)  for x in self.BOXES[idx]]).astype(float)
            attributes = np.zeros((len(annotation), self.args.num_class)) 
            attributes_report = np.zeros((self.args.num_class)) 

            image_h, image_w = image.shape[0], image.shape[1]
            
            report = self.report_all[idx]
        else:
            idx = idx-len(self.IMAGES)
            image_info = self.images[idx]
            file_name = image_info['file_name'].split('.')[0]
            file_path = image_info['path']
            
            
            image = np.array(Image.open(os.path.join(self.image_folder, file_path)).convert('RGB'))
            data_info = self.data[self.data['dicom_id'] == file_name]

            image_path_split = file_path.split('/')
            subject_id, study_id = image_path_split[2][1:], image_path_split[3][1:]

            annotation = np.zeros((data_info.shape[0], 5))
            attributes = np.zeros((data_info.shape[0], self.args.num_class))
            attributes_report = np.zeros((self.args.num_class)).astype(int)
            for i, dat in enumerate(data_info.itertuples()):
                
                x, y, w, h = getattr(dat, 'x'), getattr(dat, 'y'), getattr(dat, 'w'), getattr(dat, 'h'), 
                height = getattr(dat, 'image_height')
                width = getattr(dat, 'image_width')
                image_h, image_w = height, width
                x_scale = 224 / width
                y_scale = 224 / height

                x1 = x 
                y1 = y 
                x2 = (x + w) 
                y2 = (y + h) 
                # print(x1,y1,x2,y2)
                if (x2-x1) < 1 or (y2-y1) < 1:
                    continue
                annotation[i, 0] = np.round(x1)
                annotation[i, 1] = np.round(y1)
                annotation[i, 2] = np.round(x2)
                annotation[i, 3] = np.round(y2)

                annotation[i, 4]  = self.name_to_label(getattr(dat, 'category_name'))
                # i+=1
                attributes[i] = eval(getattr(dat, 'attibutes'))
                attributes_report = (attributes_report|np.array(attributes[i]).astype(int))

                # get report
                if i==0:
                    report_info = self.report[self.report['dicom_id'] == file_name]
                    for rep in (report_info.itertuples()):
                        rep = rep
                    impression = getattr(rep, 'impression')
                    findings =  getattr(rep, 'findings')
                    report = findings
                    
        subject = self.report_annot[self.report_annot['subject_id'] == int(subject_id)]
        study = subject[subject['study_id'] == int(study_id)]
        struct_label = np.zeros(8)
        for i in range(0, len(self.class_name)):
            label_name = self.class_name[i]
            label_val = getattr(study, label_name)
            if (int(label_val) > 0):
                struct_label[i] = 1
        report_ids = self.tokenizer(report)[:self.max_seq_length]
        report_masks = [1] * len(report_ids)
        seq_length = len(report_ids)
        sample = {'img': image/255.0, 'annot': np.array(annotation)}
        if self.transform is not None:
            sample = self.transform(sample)

        sample['report_ids'] = report_ids
        sample['report_masks'] = report_masks
        sample['seq_length'] = seq_length
        sample['struct_label'] = struct_label
        return sample

    def image_aspect_ratio(self, idx):
        if idx < len(self.examples):
            example = self.examples[idx]
            # image_id = example['id']
            image_path = example['image_path']
            # try:
            image = (Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
        else:
            idx = idx-len(self.examples)
            image_info = self.images[idx]
            file_name = image_info['file_name'].split('.')[0]
            file_path = image_info['path']

            image = (Image.open(os.path.join(self.image_folder, file_path)).convert('RGB'))
        # except:
            # return self.__getitem__(idx+1)
        return float(image.width) / float(image.height)



class Resizer_img(object):

    def __call__(self, sample, min_side=608, max_side=1024):
        image = sample

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        # print(np.array(image).max())
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        # print("after:", np.array(image).max())
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return (new_image)


def collater(data):
    sample = data
    imgs = [s['img'] for s in sample]
    annots = [s['annot'] for s in sample]
    scale = [s['scale'] for s in sample]
    report_ids = [s['report_ids'] for s in sample]
    report_masks = [s['report_masks'] for s in sample]
    seq_length = [s['seq_length'] for s in sample]
    attributes = [s['attributes'] for s in sample]
    attributes_report = [s['attributes_report'] for s in sample]
    image_location = [s['image_location'] for s in sample]
    if 'label_text' in sample[0].keys():
        label_text = [s['label_text'] for s in sample]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 4)) * -1
        attribute_padded = torch.zeros((len(annots), max_num_annots, 19))
        image_location_padded = torch.zeros((len(annots), max_num_annots, 5))

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
            for idx, (attr, loc) in enumerate(zip(attributes, image_location)):
                # print(attr.shape) # (1,23)
                if attr.shape[0] > 0:
                    attribute_padded[idx, :attr.shape[0], :] = torch.Tensor(attr)
                    image_location_padded[idx, :loc.shape[0], :] = torch.Tensor(loc)
    else:
        annot_padded = torch.ones((len(annots), 1, 4)) * -1
        attribute_padded = torch.zeros((len(annots), 1, 19))
        image_location_padded = torch.zeros((len(annots), 1, 5))

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scale, 'report_ids': report_ids, 'report_masks': report_masks, 'seq_length': seq_length
            , 'attributes': attribute_padded, 'attributes_report': attributes_report, 'image_location': image_location_padded}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        if 'label_text' in sample.keys():
            image, annots, label_text = sample['img'], sample['annot'], sample['label_text']
        else:
            image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        # print(np.array(image).max())
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        # print("after:", np.array(image).max())
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        # print(new_image.shape)
        annots[:, :4] *= scale
        scale = [new_image.shape[0], new_image.shape[1], scale]
        if 'label_text' in sample.keys():
            return {'img':torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'label_text': label_text, 'scale': scale}

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            if 'label_text' in sample.keys():
                image, annots, label_text = sample['img'], sample['annot'], sample['label_text']
            else:
                image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            if 'label_text' in sample.keys():
                sample= {'img':image, 'annot': annots, 'label_text': label_text}
            else:
                sample= {'img':image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        # print(sample.keys())
        if 'label_text' in sample.keys():
            image, annots, label_text = sample['img'], sample['annot'], sample['label_text']
        else:
            image, annots = sample['img'], sample['annot']
        img = ((image.astype(np.float32)-self.mean)/self.std)
        # print("normalize", img.max())
        if 'label_text' in sample.keys():
            return {'img':img, 'annot': annots, 'label_text': label_text}
        return {'img':img, 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
