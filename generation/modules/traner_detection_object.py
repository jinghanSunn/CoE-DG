import json
import os
from abc import abstractmethod
from modules.loss import AsymmetricLoss, FocalLoss
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torchvision import transforms

import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import sklearn
import numpy as np


class BaseTrainer(object):
    def __init__(self, model, retinanet, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.retinanet = retinanet.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            self.retinanet = torch.nn.DataParallel(retinanet, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.transform = transforms.Resize((224, 224))

        criterion = torch.nn.CrossEntropyLoss()

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {
            'val': {self.mnt_metric: self.mnt_best},
                            }

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        record_json = {}
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, val_gts, val_res  = self._train_epoch(epoch)

            # save outputs each epoch
            save_outputs = {'gts': val_gts, 'res': val_res}
            with open(os.path.join('records', str(epoch)+'_token_results.json'), 'w') as f:
                json.dump(save_outputs, f)

            # save logged informations into log dict
            if (epoch-1) % 1 == 0:
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)
                record_json[epoch] = log

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False


            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        self._save_file(record_json)
    
    def _save_file(self, log):
        if not os.path.exists(self.args.record_dir):
            os.mkdir(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.json')
        with open(record_path, 'w') as f:
            json.dump(log, f)

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, f'current_checkpoint_{epoch}.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)


    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, retinanet, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, 
                 val_dataloader):
        super(Trainer, self).__init__(model, retinanet, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _train_epoch(self, epoch):
        self.retinanet.eval()
        train_loss = 0
        print_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, label) in enumerate(tqdm(self.train_dataloader)):
            images, reports_ids, reports_masks, label = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device)
            # detection
            max_box_len = 0
            all_boxes = torch.ones((images.shape[0], 10, 5)) * -1

            cropped_images_all_bs = []
            for bs in range(images.shape[0]):
                detect = 0
                image = images[bs].unsqueeze(0) # [1, 3, 224, 224]
                scores, classification_un, transformed_anchors = self.retinanet(image.cuda().float())
                idxs = np.where(scores.cpu()>self.args.thresh_score)
                if idxs[0].shape[0]>max_box_len:
                    max_box_len = idxs[0].shape[0]
                if len(idxs[0])!=0:
                    detect = 1

                if detect == 1:
                    for j in range(idxs[0].shape[0]):
                        bbox = transformed_anchors[idxs[0][j], :]
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])
                        try:
                            tmp_crop_img = self.transform(image[:,:,y1:y2, x1:x2])
                            Flag = 0
                        except:
                            tmp_crop_img = image
                            Flag = 1
                        if j==0:
                            crop_img = tmp_crop_img 
                        else:
                            crop_img = torch.cat([crop_img, tmp_crop_img], dim=0) 
                if idxs[0].shape[0] > 0 and Flag == 0 and detect==1:
                    cropped_images_all_bs.append(crop_img)
                else:
                    cropped_images_all_bs.append(self.transform(image)) 
                
            if detect == 0:
                padded_images = self.transform(images).unsqueeze(1)
            else:
                padded_images = torch.zeros((images.shape[0], max_box_len, 3, 224, 224)).cuda()
                for i in range(len(cropped_images_all_bs)):
                    img = cropped_images_all_bs[i]
                    padded_images[i, :int(img.shape[0]), :, :, :] = img

            pred, output = self.model((padded_images), reports_ids, mode='train')
            cls_loss = self.cls_criterion(pred, label)
            loss = cls_loss + self.criterion(output, reports_ids, reports_masks)
            print_loss += loss.item()
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if batch_idx %5000 == 0:
                print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss/5))
                print_loss = 0
            # break
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        val_gts, val_res = [], []
        if (epoch-1) % 1 == 0:
            self.model.eval()
            print("-----------val----------------")
            pred = []
            label = []
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks, label_cls) in enumerate(tqdm(self.val_dataloader)):
                    images, reports_ids, reports_masks, label_cls = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), label_cls.to(self.device)
                    # detection
                    max_box_len = 0
                    all_boxes = torch.ones((images.shape[0], 10, 5)) * -1
                    
                    # print("images.shape", images.shape) # [16, 3, 224, 224]
                    cropped_images_all_bs = []
                    for bs in range(images.shape[0]):
                        detect = 0
                        image = images[bs].unsqueeze(0) 
                        scores, classification_un, transformed_anchors = self.retinanet(image.cuda().float())
                        # print("scores", scores)
                        idxs = np.where(scores.cpu()>self.args.thresh_score)
                        # print(idxs)
                        if idxs[0].shape[0]>max_box_len:
                            max_box_len = idxs[0].shape[0]
                        if len(idxs[0])!=0:
                            detect = 1
                        # else:
                        #     continue
                        if detect == 1:
                            for j in range(idxs[0].shape[0]):
                                bbox = transformed_anchors[idxs[0][j], :]
                                x1 = int(bbox[0])
                                y1 = int(bbox[1])
                                x2 = int(bbox[2])
                                y2 = int(bbox[3])
                                try:
                                    tmp_crop_img = self.transform(image[:,:,y1:y2, x1:x2])
                                    Flag = 0
                                except:
                                    tmp_crop_img = image
                                    Flag = 1
                                if j==0:
                                    crop_img = tmp_crop_img 
                                else:
                                    crop_img = torch.cat([crop_img, tmp_crop_img], dim=0) 
                   
                        if idxs[0].shape[0] > 0 and Flag == 0 and detect==1:
                            cropped_images_all_bs.append(crop_img)
                        else:
                            cropped_images_all_bs.append(self.transform(image)) 
               
                    if detect == 0:
                        padded_images = self.transform(images).unsqueeze(1)
                    else:
                        padded_images = torch.zeros((images.shape[0], max_box_len, 3, 224, 224)).cuda()
                        for i in range(len(cropped_images_all_bs)):
                            img = cropped_images_all_bs[i]
                            padded_images[i, :int(img.shape[0]), :, :, :] = img

             
                    output, _, pred_cls = self.model((padded_images), mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    pred.extend(list(pred_cls.cpu().numpy().copy()))
                    label.extend(list(label_cls.cpu().numpy().copy()))
                    
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                            {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                # print(val_met)


                auc = roc_auc_score(label, pred, average=None)

                print("auc", auc)
                print("mean auc", np.mean(auc))
                # print("acc:", correct_all/len(dataloader_val))
                print('macro auc:', roc_auc_score(label, pred, average='macro'))
                print('weighted auc:',roc_auc_score(label, pred, average='weighted'))
                print('micro auc:', roc_auc_score(label, pred, average='micro'))
                


        self.lr_scheduler.step()
        return log, val_gts, val_res


class TrainerPre(BaseTrainer):
    def __init__(self, model, retinanet, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, 
                 val_dataloader):
        super(TrainerPre, self).__init__(model, retinanet, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _train_epoch(self, epoch):
        self.retinanet.eval()
        train_loss = 0
        print_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, label) in enumerate(tqdm(self.train_dataloader)):
            images, reports_ids, reports_masks, label = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device)
            pred, output = self.model((images), reports_ids, mode='train')
            cls_loss = self.cls_criterion(pred, label)
            loss = cls_loss + self.criterion(output, reports_ids, reports_masks)
            print_loss += loss.item()
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if batch_idx %5000 == 0:
                print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss/5))
                print_loss = 0
            # break
        log = {'train_loss': train_loss / len(self.train_dataloader)}


        val_gts, val_res = [], []
        if (epoch-1) % 1 == 0:
            self.model.eval()
            print("-----------val----------------")
            pred = []
            label = []
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks, label_cls) in enumerate(tqdm(self.val_dataloader)):
                    images, reports_ids, reports_masks, label_cls = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), label_cls.to(self.device)
                    output, _, pred_cls = self.model((images), mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    pred.extend(list(pred_cls.cpu().numpy().copy()))
                    label.extend(list(label_cls.cpu().numpy().copy()))
                    
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                            {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                # print(val_met)


                auc = roc_auc_score(label, pred, average=None)

                print("auc", auc)
                print("mean auc", np.mean(auc))
                


        self.lr_scheduler.step()
        return log, val_gts, val_res

class TrainerMulti(BaseTrainer):
    def __init__(self, model, retinanet, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, 
                 val_dataloader):
        super(TrainerMulti, self).__init__(model, retinanet, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _train_epoch(self, epoch):
        self.retinanet.eval()
        train_loss = 0
        print_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, label, coord_embed, label_name) in enumerate(tqdm(self.train_dataloader)):
            images, reports_ids, reports_masks, label, coord_embed, label_name = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device), coord_embed.to(self.device), label_name.to(self.device)
            if self.args.loc_mode == 'embedding':
                pred, output = self.model((images), reports_ids, mode='train', location_rep=torch.reshape(coord_embed, (self.args.batch_size, -1)))
            elif self.args.loc_mode == 'convert':
                pred, output = self.model((images), reports_ids, mode='train', location_rep=coord_embed.long())
            cls_loss = self.cls_criterion(pred, label)
            # print(cls_loss)
            loss = cls_loss + self.criterion(output, reports_ids, reports_masks)
            print_loss += loss.item()
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if batch_idx %5000 == 0:
                print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss/5))
                print_loss = 0
            # break
        log = {'train_loss': train_loss / len(self.train_dataloader)}


        val_gts, val_res = [], []
        if (epoch-1) % 1 == 0:
            self.model.eval()
            print("-----------val----------------")
            pred = []
            label = []
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks, label_cls, coord_embed, label_name) in enumerate(tqdm(self.val_dataloader)):
                    images, reports_ids, reports_masks, label_cls = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), label_cls.to(self.device)
                    if self.args.loc_mode == 'embedding':
                        output, _, pred_cls = self.model((images), mode='sample', location_rep=torch.reshape(coord_embed, (self.args.batch_size, -1)))
                    elif self.args.loc_mode == 'convert':
                        if self.args.label_name:
                            output, _, pred_cls = self.model((images), mode='sample', location_rep=coord_embed.long(), label_name=label_name.long())
                        else:
                            output, _, pred_cls = self.model((images), mode='sample', location_rep=coord_embed.long())
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    pred.extend(list(pred_cls.cpu().numpy().copy()))
                    label.extend(list(label_cls.cpu().numpy().copy()))
                    
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                            {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})


                auc = roc_auc_score(label, pred, average=None)

                print("auc", auc)
                print("mean auc", np.mean(auc))
                

        self.lr_scheduler.step()
        return log, val_gts, val_res