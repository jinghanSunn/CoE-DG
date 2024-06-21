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
from random import sample
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import skimage.io
import skimage.transform
import skimage.color
import skimage
from retinanet.dataloader import CocoDataset, CSVDataset 
from PIL import Image

BOX_ANNOTATION_CSV_PATH = './data/MS-CXR/ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1/MS_CXR_Local_Alignment_v1.0.0.csv'
BOX_ANNOTATION_JSON_PATH = './data/MS-CXR/ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1/MS_CXR_Local_Alignment_v1.0.0.json'
IMAGE_FOLDER_PATH = './data/MIMIC-CXR/2.0.0/'

class MimicDataset(Dataset):
    """Coco dataset."""

    def __init__(self, set_name='train', transform=None, return_name=False):
        """
        Args:
            root_dir (string): Annotation directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if set_name == 'train':
            self.resize = transforms.Compose([
                transforms.Resize(256),
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
        self.return_name = return_name

        self.data = pd.read_csv(self.root_dir)
        with open(self.json_dir, 'r') as obj:
            json_data = json.load(obj)
        self.images = json_data['images'][set_name]
        annotations = json_data['annotations'][set_name]
        X = [x['id'] for x in annotations]
        y = [c['category_id'] for c in annotations]

        self.classes = {"Cardiomegaly":1,"Lung Opacity":2,"Edema":3,"Consolidation":4,"Pneumonia":5,"Atelectasis":6,"Pneumothorax":7,"Pleural Effusion":8}

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image_info = self.images[idx]
        file_name = image_info['file_name'].split('.')[0]
        file_path = image_info['path']

        image = np.array(Image.open(os.path.join(self.image_folder, file_path)).convert('RGB'))

        data_info = self.data[self.data['dicom_id'] == file_name]

        annotation = np.zeros((data_info.shape[0], 5))
        for i, dat in enumerate(data_info.itertuples()):
            x, y, w, h = getattr(dat, 'x'), getattr(dat, 'y'), getattr(dat, 'w'), getattr(dat, 'h'), 
            height = getattr(dat, 'image_height')
            width = getattr(dat, 'image_width')
   
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
        if self.return_name:
            sample['name'] = file_name
            return sample
        return sample

    def name_to_label(self, name):
        class_name = ["Background", "Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        return class_name.index(name)

    def label_to_name(self, label):
        class_name = ["Background", "Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        return class_name[label]

    def num_classes(self):
        return 8 + 1

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

        annotation = np.zeros((data_info.shape[0], 5))
        for i, dat in enumerate(data_info.itertuples()):
            x, y, w, h = getattr(dat, 'x'), getattr(dat, 'y'), getattr(dat, 'w'), getattr(dat, 'h'), 
            # print(x,y,w,h)
            height = getattr(dat, 'image_height')
            width = getattr(dat, 'image_width')

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

