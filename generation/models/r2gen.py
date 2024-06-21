import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor, VisualExtractorObject, VisualExtractorPreObject
from modules.encoder_decoder_wToken import EncoderDecoder
# from modules.encoder_decoder import EncoderDecoder



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
        self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


    def forward_mimic_cxr(self, images, targets=None, mode='train', location_rep=None, label_name=None):
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
                output = self.encoder_decoder(fc_feats, att_feats, mode='sample', location_rep=location_rep.unsqueeze(1)) 
            else:
                output = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

