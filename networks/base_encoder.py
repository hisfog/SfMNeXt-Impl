from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import hub

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2) # +176
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4) # +64
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8) # +40
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16) # +24

        # self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16) 
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        # x_d5 = self.up5(x_d4, features[0]) 
        # out = self.conv3(x_d5) 
        out = self.conv3(x_d4) 
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features



class BaseEncoder(nn.Module):
    def __init__(self, backend, model_dim=32, num_features=2048):
        super(BaseEncoder, self).__init__()
        self.encoder = Encoder(backend)

        self.decoder = DecoderBN(num_features=num_features, num_classes=model_dim)
        # self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
        #                               nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, **kwargs)

    @classmethod
    def build(cls, model_dim, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='') 
        basemodel = hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=False)

        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, model_dim=model_dim, **kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    model = BaseEncoder.build(100)
    x = torch.rand(2, 3, 352, 704)
    # bins, pred = model(x)
    x = model(x)
    print(x.shape)
    # print(bins.shape, pred.shape)

