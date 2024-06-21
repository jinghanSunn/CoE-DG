import argparse
import collections
import copy

import numpy as np
import os

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.ops import nms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater_report, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from retinanet.mimic_dataloader import MimicDataset
from retinanet.dataset import MimiccxrSingleImageDataset
from retinanet.sample import TwoStreamBatchSampler
from retinanet import losses
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = alpha
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 200.0)


def patch_report(reports_ids, reports_masks, seq_lengths):
        max_seq_length = 77

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        # ipdb.set_trace()
        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks
        # ipdb.set_trace()
        return torch.LongTensor(targets), torch.FloatTensor(targets_masks)


def get_detection(regressBoxes, clipBoxes, anchors, regression, classification, img_batch):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, img_batch)

    finalResult = [[], [], []]

    finalScores = torch.Tensor([])
    finalAnchorBoxesIndexes = torch.Tensor([]).long()
    finalAnchorBoxesCoordinates = torch.Tensor([])

    if torch.cuda.is_available():
        finalScores = finalScores.cuda()
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

    for i in range(classification.shape[2]):
        scores = torch.squeeze(classification[:, :, i])
        scores_over_thresh = (scores > 0.05)
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just continue
            continue

        scores = scores[scores_over_thresh]
        anchorBoxes = torch.squeeze(transformed_anchors)
        anchorBoxes = anchorBoxes[scores_over_thresh]
        anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

        finalResult[0].extend(scores[anchors_nms_idx])
        finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
        finalResult[2].extend(anchorBoxes[anchors_nms_idx])
        # print(finalResult)

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0]) # label
        if torch.cuda.is_available():
            finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

    return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--image_dir', type=str, default='./MIMIC-CXR/2.0.0/files/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='./MIMIC-CXR/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--report_ann_path', type=str, default='./data/MS-CXR/generation_info.csv')
    parser.add_argument('--dataset')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    
    parser.add_argument('--mode', type=int, default=100)

    parser.add_argument('--save_folder', help='Path to save model', type=str, default='')
    parser.add_argument('--load_model', help='Path of pretrained model', type=str, default=None)
    parser.add_argument('--num_class', type=int, default=19, help='the number of attribute.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')
    parser.add_argument('--labeled_bs', type=int, default=4, help='the number of samples for a batch')
    parser.add_argument('--unlabeled_bs', type=int, default=2, help='the number of samples for a batch')

    parser.add_argument('--lr', type=float, default=1e-5, help='the number of samples for a batch')
    parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
    parser.add_argument('--thresh_score', type=float, default=0.9)
    parser.add_argument('--thresh_score_curr', type=float, default=0.9)
    parser.add_argument('--max_seq_length', type=int, default=77, help='the maximum sequence length of the reports.')

    parser = parser.parse_args(args)

    if not os.path.exists(parser.save_folder):
        os.makedirs(parser.save_folder)


    dataset_val = MimicDataset(set_name='val',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_train = MimiccxrSingleImageDataset(parser, split='train', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    batch_sampler = TwoStreamBatchSampler(
        dataset_train.labeled_idxs, dataset_train.unlabeled_idxs, int(parser.batch_size), int(parser.batch_size-parser.labeled_bs))

    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater_report, batch_sampler=batch_sampler, pin_memory=True,)

    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater_report, batch_size=1)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    ema_retinanet = copy.deepcopy(retinanet)

    if parser.load_model is not None:
        ema_retinanet.load_state_dict(torch.load(parser.load_model).module.state_dict(), strict=True)
        for param in ema_retinanet.parameters():
            param.detach_()
        

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
            ema_retinanet = ema_retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        ema_retinanet = torch.nn.DataParallel(ema_retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    focalLoss = losses.FocalLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    ema_retinanet.eval()
    retinanet.module.freeze_bn()

    it_num = 0
    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        all_pseudo_box_num = 0
        filtered_box_num = 0
        curr_filter_box = 0
        all_images = 0
        all_filter_images = 0

        
        for iter_num, data in enumerate(dataloader_train):
                flag = 0
                optimizer.zero_grad()
                all_images += data['img'][parser.labeled_bs:].shape[0]
                
                # labeled
                if torch.cuda.is_available():
                    (classification_loss, regression_loss), classification, regression, anchors, annotations  = retinanet([data['img'].cuda().float(), data['annot'].cuda()], return_result=True)
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                labeled_loss = classification_loss + regression_loss

                # unlabeled
                unlabeled_data = data['img'][parser.labeled_bs:]
                unlabel_struct_label = data['struct_label'][parser.labeled_bs:]

                max_box_len = 0
                all_boxes = torch.ones((unlabeled_data.shape[0], 10, 5)) * -1
                detect = 0
                for bs in range(unlabeled_data.shape[0]):
                    
                    unlabel_tmp = unlabeled_data[bs].unsqueeze(0)
                    scores_curr, classification_curr, transformed_anchors_curr = get_detection(retinanet.module.regressBoxes, retinanet.module.clipBoxes, anchors, regression[parser.labeled_bs+bs].unsqueeze(0), classification[parser.labeled_bs+bs].unsqueeze(0), unlabel_tmp)
                    idxs_curr = np.where(scores_curr.cpu()>parser.thresh_score_curr)
                    scores_curr_filter = scores_curr[idxs_curr[0]]
                    classification_curr_filter = classification_curr[idxs_curr[0]]
                    transformed_anchors_curr_filter = transformed_anchors_curr[idxs_curr[0]]
                    scores, classification_un, transformed_anchors = ema_retinanet(unlabel_tmp.cuda().float())
                    idxs = np.where(scores.cpu()>parser.thresh_score)

                    # Self-Adaptive NMS
                    if classification_curr_filter.shape[0] != 0:
                        scores_filter = torch.cat([scores[idxs[0]], scores_curr_filter], dim=0)
                        classification_filter = torch.cat([classification_un[idxs[0]], classification_curr_filter], dim=0)
                        transformed_anchors_filter = torch.cat([transformed_anchors[idxs[0]],transformed_anchors_curr_filter], dim=0)
                        curr_start_idx = transformed_anchors[idxs[0]].shape[0]
                        nms_idx = nms(transformed_anchors_filter, scores_filter, 0.5)
                        flag = 1
                        idxs = nms_idx.unsqueeze(0).cpu().detach()
                        transformed_anchors_filter = transformed_anchors_filter.cpu().detach()
                    else:
                        scores_filter = scores
                        classification_filter = classification_un
                        transformed_anchors_filter = transformed_anchors
                    if idxs[0].shape[0]>max_box_len:
                        max_box_len = idxs[0].shape[0]
                    if len(idxs[0])!=0:
                        detect = 1
                    else:
                        continue
                    all_filter_images += 1
                    all_pseudo_box_num += idxs[0].shape[0]
                    box_num = idxs[0].shape[0]
                    for j in range(box_num):
                        bbox = torch.zeros(5)
                        bbox[:4] = transformed_anchors_filter[idxs[0][j], :]
                        bbox[4] = int(classification_filter[idxs[0][j]])

                        # GIP: generator-guided information propagation
                        struct_annot = np.argwhere(unlabel_struct_label[bs]>0) 
                        if bbox[4].numpy() in struct_annot: 
                            filtered_box_num += 1
                            if flag == 1:
                                if idxs[0][j] >= curr_start_idx:
                                    curr_filter_box += 1
            
                            all_boxes[bs,j,:] = bbox

                unlabeled_classification_loss,  unlabeled_regression_loss = focalLoss(classification[parser.labeled_bs:], regression[parser.labeled_bs:], anchors, all_boxes.cuda())

                unlabeled_loss = unlabeled_classification_loss + unlabeled_regression_loss  
                loss = labeled_loss + unlabeled_loss     
                it_num = it_num + 1

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                if iter_num %50==0:  
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Unlabel loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), float(unlabeled_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss

        print('Evaluating dataset')
        map = csv_eval.evaluate(dataset_val, retinanet, iou_threshold=0.5)
        print("IOU=0.5")
        print("MAP:", map)
        maps = np.zeros((0,))
        for key, item in map.items():
            map, _ = item
            maps = np.append(maps, map)
        print("Mean AP:",maps.mean())
        scheduler.step(np.mean(epoch_loss))


if __name__ == '__main__':
    main()
