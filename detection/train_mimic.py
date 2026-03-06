"""
Training script for RetinaNet on MIMIC-CXR dataset.

This script trains a RetinaNet model for chest X-ray abnormality detection
using the MIMIC-CXR dataset with bounding box annotations.
"""
import argparse
import collections
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from retinanet import coco_eval, csv_eval, model
from retinanet.dataloader import (
    Augmenter, CocoDataset, CSVDataset, Normalizer, Resizer, collater
)
from retinanet.mimic_dataloader import MimicDataset, MimicNormalDataset

assert torch.__version__.split('.')[0] == '1', "PyTorch version 1.x is required"

print(f'CUDA available: {torch.cuda.is_available()}')


def main(args=None):
    """
    Main training function for RetinaNet on MIMIC-CXR dataset.

    Args:
        args: Command line arguments (optional)
    """
    parser = argparse.ArgumentParser(
        description='Training script for RetinaNet network on MIMIC-CXR dataset.'
    )

    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument(
        '--depth',
        type=int,
        default=50,
        help='ResNet depth, must be one of 18, 34, 50, 101, 152'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--mode', type=int, default=100, help='Training mode')
    parser.add_argument('--save_folder', type=str, default='', help='Path to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to pretrained model')

    parser = parser.parse_args(args)

    # Create save directory if it doesn't exist
    if not os.path.exists(parser.save_folder):
        os.makedirs(parser.save_folder)

    # Create the data loaders
    if parser.dataset == 'mimic':
        dataset_train = MimicDataset(set_name='train',transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = MimicDataset(set_name='val',transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=16, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 50, 101')

    if parser.load_model is not None:
        retinanet.load_state_dict(torch.load(parser.load_model).module.state_dict(), strict=True)
        print(f"load model from {parser.load_model}")

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    total_params = sum(p.numel() for p in retinanet.parameters())
    trainable_params = sum(p.numel() for p in retinanet.parameters() if p.requires_grad)
    print(total_params, trainable_params)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(tqdm(dataloader_train)):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                if iter_num %50==0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss

            except Exception as e:
                print(e)
                continue
            
        map = csv_eval.evaluate(dataset_val, retinanet, iou_threshold=0.5)
        print("IOU=0.5")
        print("MAP:", map)
        maps = np.zeros((0,))
        for key, item in map.items():
            map, _ = item
            maps = np.append(maps, map)
        print("Mean AP:",maps.mean())

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
