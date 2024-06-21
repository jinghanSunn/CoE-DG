import os
import json
import pickle
import torch
from PIL import Image
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
import skimage
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.args = args
        # self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        # for cl in self.class_name:
            # print(self.tokenizer(cl))
        if (args.dataset_name == 'mimic_detection_pre' or args.dataset_name == 'mimic_detection_multi') and split =='train':
            self.transform=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                                     ])
        elif (args.dataset_name == 'mimic_detection_pre' or args.dataset_name == 'mimic_detection_multi') and split =='test':
            self.transform=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                                     ])
        else:
            self.transform_img=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                                     ])
            self.transform = transforms.Compose([Normalizer(), Resizer()])


        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
        REPORT_annotation = pd.read_csv(args.structure_path)
        self.report_annot = REPORT_annotation
        self.report_annot = self.report_annot.fillna(0)
    def __len__(self):
        return len(self.examples)

class BaseTestDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.args = args
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.class_name = ["Cardiomegaly","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion"]
        if (args.dataset_name == 'mimic_detection_pre' or args.dataset_name == 'mimic_detection_multi') and split =='train':
            self.transform=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                                     ])
        else:
            self.transform_img=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                                     ])
            self.transform = transforms.Compose([Normalizer(), Resizer()])


        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
        REPORT_annotation = pd.read_csv(args.structure_path)
        self.report_annot = REPORT_annotation
        self.report_annot = self.report_annot.fillna(0)
    def __len__(self):
        return len(self.examples)



class MimiccxrDetectionImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        subject_id, study_id  = example['subject_id'], example['study_id']
        subject = self.report_annot[self.report_annot['subject_id'] == int(subject_id)]
        study = subject[subject['study_id'] == int(study_id)]
        struct_label = np.zeros(8)
        for i in range(0, len(self.class_name)):
            label_name = self.class_name[i]
            label_val = getattr(study, label_name)
            # print(label_val)
            if (int(label_val) > 0):
                struct_label[i] = 1

        image_id = example['id']
        image_path = example['image_path'][0] 
        try:
            image = np.array(Image.open(os.path.join(self.image_dir, image_path)).convert('RGB'))
        except:
            return self.__getitem__(idx+1)
        if self.transform is not None:
            image = self.transform(image/255.0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, struct_label, seq_length)
        return sample

class MimiccxrDetectionImageTestDataset(BaseTestDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        subject_id, study_id  = example['subject_id'], example['study_id']
        subject = self.report_annot[self.report_annot['subject_id'] == int(subject_id)]
        study = subject[subject['study_id'] == int(study_id)]
        struct_label = np.zeros(8)
        for i in range(0, len(self.class_name)):
            label_name = self.class_name[i]
            label_val = getattr(study, label_name)
            # print(label_val)
            if (int(label_val) > 0):
                struct_label[i] = 1

        image_id = example['id']
        image_path_split = example['image_path'][0].split('/') 
        folder = '/'.join(image_path_split[:3]) 
        try:
            file_paths = os.listdir(os.path.join(self.image_dir, folder))
            pkl_path = os.path.join(self.image_dir, example['image_path'][0].replace('.jpg', '.pkl'))
            box_paths = [file for file in file_paths if 'box7' in file]
            # print(box_paths)
        
            if len(box_paths) == 0:
                image = Image.open(os.path.join(self.image_dir, example['image_path'][0])).convert('RGB')
                h = image.height
                w = image.width
                if self.transform is not None:
                    image = self.transform(image)
                boxes = image
                label_name = self.tokenizer('normal')
                if self.args.loc_mode == 'embedding':
                    coord_embed = coordinate_embeddings(np.array([[0,0,w,h]]), 256, w, h)
                elif self.args.loc_mode == 'convert':
                    coord_embed = coordinate_convert(np.array([[0,0,w,h]]), w, h)[0].unsqueeze(0)
            else:
                image = Image.open(os.path.join(self.image_dir, example['image_path'][0])).convert('RGB') 
                h = image.height
                w = image.width
                if self.transform is not None:
                    image = self.transform(image)
                boxes = image
                info = pickle.load(open(pkl_path, 'rb'))
                transformed_anchors = info['box']
                scores = info['score']
                w = info['height'] 
                h = info['width']
                if len(info['class'])!=0:
                    label_name = self.tokenizer((info['class'][0])) 
                else:
                    label_name = self.tokenizer('normal')
                idxs_7 = np.where(scores>0.7)

                if len(idxs_7[0]) == 0:
                    transformed_anchors = np.array([[0,0,w,h]])
                else:
                    transformed_anchors = transformed_anchors[idxs_7[0]]
                if self.args.loc_mode == 'embedding':
                    coord_embed = coordinate_embeddings(transformed_anchors, 256, w, h)
                elif self.args.loc_mode == 'convert':
                    coord_embed = coordinate_convert(transformed_anchors, w, h)

            if len(label_name) != 5:
                label_name_pad = np.zeros(5)
                label_name_pad[0:4] = label_name
                label_name = label_name_pad
        except:
            return self.__getitem__(idx+1)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, boxes, report_ids, report_masks, struct_label, seq_length, coord_embed, torch.Tensor(label_name))
        return sample

class MimiccxrPreDetectionImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        subject_id, study_id  = example['subject_id'], example['study_id']
        subject = self.report_annot[self.report_annot['subject_id'] == int(subject_id)]
        study = subject[subject['study_id'] == int(study_id)]
        struct_label = np.zeros(8)
        for i in range(0, len(self.class_name)):
            label_name = self.class_name[i]
            label_val = getattr(study, label_name)
            if (int(label_val) > 0):
                struct_label[i] = 1

        image_id = example['id']
        image_path_split = example['image_path'][0].split('/')
        folder = '/'.join(image_path_split[:3]) 
        try:
            file_paths = os.listdir(os.path.join(self.image_dir, folder))
            box_paths = [file for file in file_paths if 'box7' in file]
        
            if len(box_paths) == 0: 
                image = Image.open(os.path.join(self.image_dir, example['image_path'][0])).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                boxes = image
            else:
                boxes = torch.zeros((len(box_paths), 3, 224, 224))
                for file in box_paths:
                    box_path = os.path.join(self.image_dir, folder, file)
                    box = Image.open(box_path).convert('RGB')
                    if self.transform is not None:
                        box = self.transform(box)
                    boxes = torch.cat((boxes, box.unsqueeze(0)), dim=0)
        except:
            return self.__getitem__(idx+1)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, boxes, report_ids, report_masks, struct_label, seq_length)
        return sample

class MimiccxrPreDetectionMultiDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        subject_id, study_id  = example['subject_id'], example['study_id']
        subject = self.report_annot[self.report_annot['subject_id'] == int(subject_id)]
        study = subject[subject['study_id'] == int(study_id)]
        struct_label = np.zeros(8)
        for i in range(0, len(self.class_name)):
            label_name = self.class_name[i]
            label_val = getattr(study, label_name)
            # print(label_val)
            if (int(label_val) > 0):
                struct_label[i] = 1

        image_id = example['id']
        image_path_split = example['image_path'][0].split('/') 
        folder = '/'.join(image_path_split[:3]) 
        try:
            file_paths = os.listdir(os.path.join(self.image_dir, folder))
            pkl_path = os.path.join(self.image_dir, example['image_path'][0].replace('.jpg', '.pkl'))
            box_paths = [file for file in file_paths if 'box7' in file]
            # print(box_paths)
        
            if len(box_paths) == 0: # if no abnormality
                image = Image.open(os.path.join(self.image_dir, example['image_path'][0])).convert('RGB')
                h = image.height
                w = image.width
                if self.transform is not None:
                    image = self.transform(image)
                boxes = image
                label_name = self.tokenizer('normal')
                if self.args.loc_mode == 'embedding':
                    coord_embed = coordinate_embeddings(np.array([[0,0,w,h]]), 256, w, h)
                elif self.args.loc_mode == 'convert':
                    coord_embed = coordinate_convert(np.array([[0,0,w,h]]), w, h)[0].unsqueeze(0)
            else:
                image = Image.open(os.path.join(self.image_dir, example['image_path'][0])).convert('RGB') 
                h = image.height
                w = image.width
                if self.transform is not None:
                    image = self.transform(image)
                boxes = image
                info = pickle.load(open(pkl_path, 'rb'))
                transformed_anchors = info['box'] # [x1, y1, x2, y2]
                scores = info['score']
                w = info['height'] 
                h = info['width']
            
                if len(info['class'])!=0:
                    label_name = self.tokenizer((info['class'][0])) 
                else:
                    label_name = self.tokenizer('normal')
    
                idxs_7 = np.where(scores>0.7)

                if len(idxs_7[0]) == 0:
                    transformed_anchors = np.array([[0,0,w,h]])
                else:
                    transformed_anchors = transformed_anchors[idxs_7[0]]
                if self.args.loc_mode == 'embedding':
                    coord_embed = coordinate_embeddings(transformed_anchors, 256, w, h)
                elif self.args.loc_mode == 'convert':
                    coord_embed = coordinate_convert(transformed_anchors, w, h)
            if len(label_name) != 5:
                label_name_pad = np.zeros(5)
                label_name_pad[0:4] = label_name
                label_name = label_name_pad
        except:
            return self.__getitem__(idx+1)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, boxes, report_ids, report_masks, struct_label, seq_length, coord_embed, torch.Tensor(label_name))
        return sample

class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image):
        img = ((image.astype(np.float32)-self.mean)/self.std)

        return img
    
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=512, max_side=512):
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

        return torch.from_numpy(new_image)

def coordinate_embeddings(boxes, dim, w, h):
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [K, 6] ([x1, y1, x2, y2, w_image, h_image])
    :param dim: sin/cos embedding dimension
    :return: [K, 4, 2 * dim]
    """
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [K, 4] ([x1, y1, x2, y2])
    :param dim: sin/cos embedding dimension
    :return: [K, 4, 2 * dim]
    """
    boxes = torch.Tensor(boxes)
    num_boxes = boxes.shape[0]


    # transform to (x_c, y_c, w, h) format
    boxes_ = boxes.new_zeros((num_boxes, 4))
    boxes_[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    boxes_[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    boxes_[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes = boxes_

    # position
    pos = boxes.new_zeros((num_boxes, 4))
    pos[:, 0] = boxes[:, 0] / w * 100
    pos[:, 1] = boxes[:, 1] / h * 100
    pos[:, 2] = boxes[:, 2] / w * 100
    pos[:, 3] = boxes[:, 3] / h * 100

    # sin/cos embedding
    dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / dim)
    sin_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).sin()
    cos_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).cos()

    return torch.cat((sin_embedding, cos_embedding), dim=-1).cpu()

def coordinate_convert(boxes, w, h):
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [K, 6] ([x1, y1, x2, y2, w_image, h_image])
    :param dim: sin/cos embedding dimension
    :return: [K, 4, 2 * dim]
    """
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [K, 4] ([x1, y1, x2, y2])
    :param dim: sin/cos embedding dimension
    :return: [K, 4, 2 * dim]
    """
    boxes = torch.Tensor(boxes)
    num_boxes = boxes.shape[0]

    # position
    pos = boxes.new_zeros((num_boxes, 4))
    pos[:, 0] = (boxes[:, 0] / w * 100).int()
    pos[:, 1] = (boxes[:, 1] / h * 100).int()
    pos[:, 2] = (boxes[:, 2] / w * 100).int()
    pos[:, 3] = (boxes[:, 3] / h * 100).int()


    return pos.cpu()