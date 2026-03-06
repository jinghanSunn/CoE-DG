"""
R2Gen: Radiology Report Generation model.

This module implements the R2Gen model for automatic medical report generation
from chest X-ray images with optional object detection guidance.
"""
import numpy as np
import torch
import torch.nn as nn

from modules.encoder_decoder_wToken import EncoderDecoder
from modules.visual_extractor import (
    VisualExtractor, VisualExtractorObject, VisualExtractorPreObject
)


class R2GenModel(nn.Module):
    """
    R2Gen model for radiology report generation.

    This model combines visual feature extraction with an encoder-decoder architecture
    to generate medical reports from chest X-ray images.

    Args:
        args: Model configuration arguments
        tokenizer: Tokenizer for text processing
        visual: Visual extractor type ('object', 'pre_object', or None for default)
    """

    def __init__(self, args, tokenizer, visual=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # Select visual extractor based on configuration
        if visual == 'object':
            self.visual_extractor = VisualExtractorObject(args)
        elif visual == 'pre_object':
            self.visual_extractor = VisualExtractorPreObject(args)
        else:
            self.visual_extractor = VisualExtractor(args)

        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.forward = self.forward_mimic_cxr

    def __str__(self):
        """Return string representation with trainable parameter count."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

    def forward_mimic_cxr(self, images, targets=None, mode='train',
                          location_rep=None, label_name=None):
        """
        Forward pass for MIMIC-CXR dataset.

        Args:
            images: Input chest X-ray images
            targets: Target report token IDs (for training)
            mode: 'train' for training, 'sample' for inference
            location_rep: Optional location representations from detection
            label_name: Optional label names from detection

        Returns:
            Generated report tokens or logits depending on mode

        Raises:
            ValueError: If mode is not 'train' or 'sample'
        """
        att_feats, fc_feats = self.visual_extractor(images)

        if mode == 'train':
            if location_rep is not None:
                if label_name is not None:
                    output = self.encoder_decoder(
                        fc_feats, att_feats, targets, mode='forward',
                        location_rep=location_rep.unsqueeze(1),
                        label_name=label_name
                    )
                else:
                    output = self.encoder_decoder(
                        fc_feats, att_feats, targets, mode='forward',
                        location_rep=location_rep.unsqueeze(1)
                    )
            else:
                output = self.encoder_decoder(
                    fc_feats, att_feats, targets, mode='forward'
                )
        elif mode == 'sample':
            if location_rep is not None:
                output = self.encoder_decoder(
                    fc_feats, att_feats, mode='sample',
                    location_rep=location_rep.unsqueeze(1)
                )
            else:
                output = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'sample'")

        return output

