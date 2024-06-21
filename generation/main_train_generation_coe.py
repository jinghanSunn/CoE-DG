import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.traner_detection_object import Trainer, TrainerPre, TrainerMulti
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from models.retinanet import model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_agrs():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='./MIMIC-CXR/2.0.0/files/', help='the path to the directory containing the data.')
    parser.add_argument('--structure_path', type=str, default='structure.json', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='ann_path.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_detection', choices=['mimic_cxr', 'mimic_detection', 'mimic_detection_pre', 'mimic_detection_multi'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--num_classes', type=int, default=8, help='the number of layers of Transformer.')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=4, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=20, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/mimic', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--resume_visual', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--resume_detection', type=str, help='whether to resume the training from existing checkpoints.')

    # detection
    parser.add_argument('--thresh_score', type=float, default=0.7)
    parser.add_argument('--loc_mode', default='embedding', type=str, help='')
    parser.add_argument('--label_name', default=False, help='', action='store_true')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    if args.dataset_name == 'mimic_detection_pre' or args.dataset_name == 'mimic_detection_multi':
        G_model = R2GenModel(args, tokenizer, visual='pre_object') # detect abnoramlity using pretrained detection model before training
    else:
        G_model = R2GenModel(args, tokenizer, visual='object')
    Retinanet = model.resnet101(num_classes=9, pretrained=True)

    Retinanet.load_state_dict(torch.load(args.resume_detection).module.state_dict(), strict=True)
    for param in Retinanet.parameters():
        param.detach_()

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, G_model)
    lr_scheduler = build_lr_scheduler(args, optimizer)


    # build trainer and start to train
    if args.dataset_name == 'mimic_detection_pre': # use predetected abnormality
        trainer = TrainerPre(G_model, Retinanet, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader,  test_dataloader)
    elif args.dataset_name == 'mimic_detection_multi':
        trainer = TrainerMulti(G_model, Retinanet, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader,  test_dataloader)
    else: # online detect
        trainer = Trainer(G_model, Retinanet, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader,  test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
