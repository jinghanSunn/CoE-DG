# from __future__ import annotations, print_function, division
import enum
from re import X
import sys
import os
# from tkinter import image_names
import torch
import numpy as np
import pandas as pd
import random
import csv
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
# from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
# from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
#     Normalizer
from PIL import Image

BOX_ANNOTATION_CSV_PATH = './data/MS-CXR/ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1/MS_CXR_Local_Alignment_v1.0.0.csv'
BOX_ANNOTATION_JSON_PATH = './data/MS-CXR/ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1/MS_CXR_Local_Alignment_v1.0.0.json'
IMAGE_FOLDER_PATH = './data/MIMIC-CXR/2.0.0/'

class MimicDataset(Dataset):
    """Coco dataset."""

    def __init__(self, set_name='train', transform=None):
        """
        Args:
            root_dir (string): Annotation directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if set_name == 'train':
            self.resize = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                ])
        else:
            self.resize = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                                     ])
        self.root_dir = BOX_ANNOTATION_CSV_PATH
        self.json_dir = BOX_ANNOTATION_JSON_PATH
        self.image_folder = IMAGE_FOLDER_PATH
        self.set_name = set_name
        self.transform = transform

        self.data = pd.read_csv(self.root_dir)
        with open(self.json_dir, 'r') as obj:
            json_data = json.load(obj)
        self.images = json_data['images']

        annotations = json_data['annotations']
        X = [x['id'] for x in annotations]
        y = [c['category_id'] for c in annotations]
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=6)
        if set_name == 'train':
            self.images = [img for img in self.images if img['id'] in X_train]
        else:
            self.images = [img for img in self.images if img['id'] in X_test]

        # also load the reverse (label -> name)
        self.classes = {"Cardiomegaly":0,"Lung Opacity":1,"Edema":2,"Consolidation":3,"Pneumonia":4,"Atelectasis":5,"Pneumothorax":6,"Pleural Effusion":7}
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
    # def load_classes(self):
    #     # load class names (name -> label)
    #     categories = self.coco.loadCats(self.coco.getCatIds())
    #     categories.sort(key=lambda x: x['id'])

    #     self.classes             = {}
    #     self.coco_labels         = {}
    #     self.coco_labels_inverse = {}
    #     for c in categories:
    #         self.coco_labels[len(self.classes)] = c['id']
    #         self.coco_labels_inverse[c['id']] = len(self.classes)
    #         self.classes[c['name']] = len(self.classes)

    #     # also load the reverse (label -> name)
    #     self.labels = {}
    #     for key, value in self.classes.items():
    #         self.labels[value] = key

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print("idx", idx)
        # img = self.load_image(idx)
        # annot = self.load_annotations(idx)
        # sample = {'img': img, 'annot': annot}
        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
        image_info = self.images[idx]
        file_name = image_info['file_name'].split('.')[0]
        file_path = image_info['path']

        image = np.array(Image.open(os.path.join(self.image_folder, file_path)).convert('RGB'))
        # print("open", image.max()) # 255
        # image = skimage.io.imread(os.path.join(self.image_folder, file_path))
        # print(image.shape)
        
        # print(self.data)
        data_info = self.data[self.data['dicom_id'] == file_name]

        # print(data_info.shape[0])
        annotation = np.zeros((data_info.shape[0], 5))
        # for i in range(data_info.shape[0]):
        # i = 0
        for i, dat in enumerate(data_info.itertuples()):
            # print(dat)
            # dat = data_info.loc[i,:]
            # x, y, w, h = dat['x'], dat['y'], dat['w'], dat['h']
            x, y, w, h = getattr(dat, 'x'), getattr(dat, 'y'), getattr(dat, 'w'), getattr(dat, 'h'), 
            # print(x,y,w,h)
            height = getattr(dat, 'image_height')
            width = getattr(dat, 'image_width')
            x_scale = 224 / width
            y_scale = 224 / height

            # x1 = x * x_scale
            # y1 = y * y_scale
            # x2 = (x + w) * x_scale
            # y2 = (y + h) * y_scale
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
        sample = {'img': image/255.0, 'annot': annotation}
        if self.transform is not None:
            sample = self.transform(sample)
        # image = sample['img']
        # annotation = sample['annot']
        # image = self.resize(Image.fromarray(image)) # resize image
        # sample = {'img': image, 'annot': annotation}
        # print("transform", sample['img'].max()) # 1136
        return sample

    def name_to_label(self, name):
        class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        return class_name.index(name)

    def label_to_name(self, label):
        class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        return class_name[label]

    def num_classes(self):
        return 8

    def image_aspect_ratio(self, idx):
        image_info = self.images[idx]
        file_name = image_info['file_name'].split('.')[0]
        file_path = image_info['path']
        image = Image.open(os.path.join(self.image_folder, file_path))
        return float(image.width) / float(image.height)
    
    def load_annotations(self, image_index):
        # get ground truth annotations
        image_info = self.images[image_index]
        file_name = image_info['file_name'].split('.')[0]
        data_info = self.data[self.data['dicom_id'] == file_name]

        # print(data_info.shape[0])
        annotation = np.zeros((data_info.shape[0], 5))
        # for i in range(data_info.shape[0]):
        # i = 0
        for i, dat in enumerate(data_info.itertuples()):
            # print(dat)
            # dat = data_info.loc[i,:]
            # x, y, w, h = dat['x'], dat['y'], dat['w'], dat['h']
            x, y, w, h = getattr(dat, 'x'), getattr(dat, 'y'), getattr(dat, 'w'), getattr(dat, 'h'), 
            # print(x,y,w,h)
            height = getattr(dat, 'image_height')
            width = getattr(dat, 'image_width')
            x_scale = 224 / width
            y_scale = 224 / height

            # x1 = x * x_scale
            # y1 = y * y_scale
            # x2 = (x + w) * x_scale
            # y2 = (y + h) * y_scale
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

        return annotation

# dataset_train = MimicDataset(set_name='train',transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
# dataloader_train = DataLoader(dataset_train, num_workers=3)
# for data in dataloader_train:
#     img = data['img']
#     annotation = data['annot']
#     print(annotation)