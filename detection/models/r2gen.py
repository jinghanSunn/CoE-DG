import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor, VisualExtractorObject, VisualExtractorPreObject
from modules.encoder_decoder_wToken import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, visual=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        if visual=='object':
            self.visual_extractor = VisualExtractorObject(args)
        if visual=='pre_object':
            self.visual_extractor = VisualExtractorPreObject(args)
        else:
            self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train', location_rep=None, label_name=None):
        # print("images.shape", images.shape)
        # att_feats, fc_feats = self.visual_extractor(images[0])
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            if location_rep is not None:
                if label_name is not None:
                    output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', location_rep=location_rep.unsqueeze(1), label_name=label_name)
                else:
                    output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', location_rep=location_rep.unsqueeze(1))
            else:
                output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            if location_rep is not None:
                output = self.encoder_decoder(fc_feats, att_feats, mode='sample', location_rep=location_rep.unsqueeze(1)) # _sample_beam返回的结果
            else:
                output = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

