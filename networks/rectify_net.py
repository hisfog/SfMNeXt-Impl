import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import *

class RectifyNet(nn.Module):

    def __init__(self, num_layers=18, pretrained=True):
        super(RectifyNet, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = RotDecoder(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], 1)

        b, c, h, w = x.size()
        x = F.interpolate(x, [h//2, w//2], mode='bilinear', align_corners=True) 

        features = self.encoder(x)
        rot = self.decoder([features])
        return rot

class RotDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(RotDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.conv_squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)

        self.convs_pose = []
        self.convs_pose.append(
            nn.Conv2d(num_input_features * 256, 256, 3, stride, 1))
        self.convs_pose.append(nn.Conv2d(256, 256, 3, stride, 1))
        self.convs_pose.append(
            nn.Conv2d(256, 3 * num_frames_to_predict_for, 1))

        self.relu = nn.ReLU()

        self.convs_pose = nn.ModuleList(list(self.convs_pose))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.conv_squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs_pose[i](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        # out = 0.01 * out.mean(3).mean(2)
        rot = out.view(-1, 3)
        rot[:, 2] = 0 # for mc, pitch is often fixed when helding a camera

        return rot


