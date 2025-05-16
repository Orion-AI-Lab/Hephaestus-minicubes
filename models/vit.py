import timm
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, num_classes, configs):
        super(ViT, self).__init__()

        in_chans = (len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length']

        self.model = timm.models.create_model(configs["architecture"].lower(), pretrained=True, num_classes=num_classes, in_chans=in_chans, dynamic_img_size=True)


    def forward(self, x):
        return self.model(x)