import torch
import torch.nn as nn
import torchvision.models as models
from models import resnetFromRetinaNet

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        model = resnetFromRetinaNet.resnet101(100)
        self.model = model
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

class VisualExtractorObject(nn.Module):
    def __init__(self, args):
        super(VisualExtractorObject, self).__init__()
        # self.device, device_ids = self._prepare_device(args.n_gpu)
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        # model = getattr(models, self.visual_extractor)(pretrained=False)
        model = resnetFromRetinaNet.resnet101(100)
        # print(model)
        if args.resume_visual is not None:
            model.load_state_dict(torch.load(args.resume_visual).module.state_dict(), strict=False)
        else:
            model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth', map_location=torch.device('cpu')), strict=False) 
        self.model = model
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward_batch(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
    
    def forward(self, images):
        for bs in range(images.shape[0]):
            images_crop = images[bs] 
            patch_feat, avg_feat = self.forward_batch(images_crop)
            if bs == 0:
                patch_feats, avg_feats = patch_feat.max(dim=0)[0].unsqueeze(0), avg_feat.max(dim=0)[0].unsqueeze(0)
            else:
                patch_feats, avg_feats = torch.cat((patch_feats, patch_feat.max(dim=0)[0].unsqueeze(0)), dim=0), torch.cat((avg_feats, avg_feat.max(dim=0)[0].unsqueeze(0)), dim=0)
        return patch_feats, avg_feats


class VisualExtractorPreObject(nn.Module):
    def __init__(self, args):
        super(VisualExtractorPreObject, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = resnetFromRetinaNet.resnet101(100)
        # print(model)
        if args.resume_visual is not None:
            model.load_state_dict(torch.load(args.resume_visual).module.state_dict(), strict=False)
        else:
            model.load_state_dict(torch.load('./data/resnet101-5d3b4d8f.pth', map_location=torch.device('cpu')), strict=False) # resnet101-5d3b4d8f.
        self.model = model
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward_batch(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
    
    def forward(self, images):
        for bs in range(images.shape[0]):
            images_crop = images[bs] 
            mask = images_crop.sum(dim=(1,2,3))
            images_crop = images_crop[mask!=0]
            patch_feat, avg_feat = self.forward_batch(images_crop)
            if bs == 0:
                patch_feats, avg_feats = patch_feat.max(dim=0)[0].unsqueeze(0), avg_feat.max(dim=0)[0].unsqueeze(0)
            else:
                patch_feats, avg_feats = torch.cat((patch_feats, patch_feat.max(dim=0)[0].unsqueeze(0)), dim=0), torch.cat((avg_feats, avg_feat.max(dim=0)[0].unsqueeze(0)), dim=0)
        return patch_feats, avg_feats
