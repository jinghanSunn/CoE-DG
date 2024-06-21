import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
from .dataset_detection import MimiccxrDetectionImageDataset, MimiccxrPreDetectionImageDataset, MimiccxrPreDetectionMultiDataset, MimiccxrDetectionImageTestDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])


        if self.dataset_name == 'mimic_detection_multi':
            self.dataset = MimiccxrPreDetectionMultiDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'test':
            self.dataset = MimiccxrDetectionImageTestDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        if self.dataset_name == 'mimic_detection_multi':
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'collate_fn': self.collate_fn_multi,
                'num_workers': self.num_workers
            }
        elif self.dataset_name == 'mimic_cxr':
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'collate_fn': self.collate_fn_ori,
                'num_workers': self.num_workers
            }
        else:
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers
            }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn_ori(data):
        images_id, images, reports_ids, reports_masks, struct_label, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
       
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(struct_label)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, struct_label, seq_lengths = zip(*data)
        widths = [int(s.shape[0]) for s in images]
        heights = [int(s.shape[1]) for s in images]
        batch_size = len(images)

        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

        for i in range(batch_size):
            img = images[i]
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
        padded_imgs = padded_imgs.permute(0, 3, 1, 2)
        images = padded_imgs
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(struct_label)
    
    @staticmethod
    def collate_fn_multi(data):
        images_id, images, reports_ids, reports_masks, struct_label, seq_lengths, coord_embed, label_name = zip(*data)
        
        coord_embed_sta = torch.stack(coord_embed, 0)
        label_name = torch.stack(label_name, 0)
        batch_size = len(images)
        max_num_annots = max(annot.shape[0] for annot in images)
        max_width = 224
        max_height = 224


        padded_imgs = torch.zeros((batch_size, max_num_annots, 3, max_width, max_height)) 

        for idx, annot in enumerate(images):
            #print(annot.shape)
            if annot.shape[0] > 0:
                padded_imgs[idx, :annot.shape[0], :, :, :] = annot


        images = padded_imgs
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(struct_label), coord_embed_sta, label_name

