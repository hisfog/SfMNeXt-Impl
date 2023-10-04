import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
# from backbones_unet import __available_models__


class Unet(nn.Module):
    """
    This class utilizes a pre-trained model as a backbone network and 
    rearranges it as an encoder for fully convolutional neural network UNet.
    The UNet model consists of an encoder path and a decoder path. 
    The encoder path is made up of multiple downsampling blocks, 
    which reduce the spatial resolution of the feature maps and 
    increase the number of filters. The decoder path is made up 
    of multiple upsampling blocks, which increase the spatial 
    resolution of the feature maps and decrease the number of filters. 
    The decoder path also concatenates feature maps from the corresponding 
    encoder layer through skip connections, allowing the model to make use 
    of both low-level and high-level features.
    Reference: Ronneberger, O., Fischer, P., & Brox, T. (2015). 
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    In International Conference on Medical image computing and 
    computer-assisted intervention (pp. 234-241). Springer, Cham.
    NOTE: This is based off an old version of Unet in 
    https://github.com/qubvel/segmentation_models.pytorch
    Parameters
    ----------
    backbone: string, default='resnet50'
        Name of classification model (without last dense layers) used 
        as feature extractor to build segmentation model. The backbone 
        is created using the timm wrapper (pre-training on ImageNet1K).
    encoder_freeze: boolean, default=False
        If ``True`` set all layers of encoder (backbone model) as non-trainable.
    pretrained: boolean, default=True
        If true, it download the weights of the pre-trained backbone network 
        in the ImageNet1K dataset.
    preprocessing: boolean, default=False
        If true, the preprocessing step is applied according to the mean 
        and standard deviation of the encoder model.
    non_trainable_layers: tuple, default=(0, 1, 2, 3, 4)
        Specifies which layers are non-trainable. 
        For example, if it is 3, the layer of index 3 in the given backbone network 
        are set as non-trainable. If the encoder_freeze parameter is set to True, 
        set this parameter.
    backbone_kwargs:
        pretrained (bool): load pretrained ImageNet-1k weights if true
        scriptable (bool): set layer config so that model is jit scriptable 
                           (not working for all models yet)
        exportable (bool): set layer config so that model is traceable 
                           ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit 
                       scripted layers (so far activations only)

        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific for more information;
        https://github.com/rwightman/pytorch-image-models
    backbone_indices: list, default=None
        Out indices, selects which indices to output.
    decoder_use_batchnorm: boolean, default=True
        If ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` 
        layers is used.    
    decoder_channels: tuple, default=(256, 128, 64, 32, 16)
        Tuple of numbers of ``Conv2D`` layer filters in decoder blocks.  
    in_channels: int, default=1
        Specifies the channel number of the image. By default it is 1. 
        If there is an RGB image input, please set it to 3.
    num_classes: int, default=5
        A number of classes for output (output shape - ``(batch, classes, h, w)``).
    center: boolean, default=False
        If ``True`` add ``Conv2dReLU`` block on encoder head.
    norm_layer: torch.nn.Module, default=nn.BatchNorm2d
        Is a layer that normalizes the activations of the previous layer.
    activation: torch.nn.Module, default=nn.ReLU
        An activation function to apply after the final convolution layer.
        Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, 
        **"tanh"**, **"identity"**, **relu**.
        Default is **relu**
    """
    def __init__(
            self,
            backbone='convnext_large',
            encoder_freeze=False,
            pretrained=True,
            preprocessing=False,
            non_trainable_layers=(0, 1, 2, 3, 4),
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            # decoder_channels=(256, 128, 64, 32, 16),
            decoder_channels=(1024, 512, 256, 128),
            # decoder_channels=(1536, 768, 384, 192),
            in_channels=1,
            num_classes=5,
            center=False,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices 
        # specified based on the alignment of features
        # and some models won't have a full enough range 
        # of feature strides to work properly.
        if backbone not in __available_models__:
            raise RuntimeError(
                f"Unknown backbone {backbone}!\n"
                f"Existing models should be selected " 
                f"from pretrained_backbones_unet import __available_models__"
            )

        encoder = create_model(
            backbone, features_only=True, out_indices=backbone_indices, 
            in_chans=in_channels, pretrained=pretrained, **backbone_kwargs
        )
        encoder_channels = [info["num_chs"] for info in encoder.feature_info][::-1]
        self.encoder = encoder
        if encoder_freeze: self._freeze_encoder(non_trainable_layers)
        if preprocessing:
            self.mean = self.encoder.default_cfg["mean"] 
            self.std = self.encoder.default_cfg["std"]
        else:
            self.mean = None
            self.std = None

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            activation=activation
        )

    def forward(self, x: torch.Tensor):
        if self.mean and self.std:
            x = self._preprocess_input(x)
        x = self.encoder(x)
        # for ft in x:
        #     print(ft.shape, " ==")
        x.reverse()  
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        """
        Inference method. Switch model to `eval` mode, 
        call `.forward(x)` with `torch.no_grad()`
        Parameters
        ----------
        x: torch.Tensor
            4D torch tensor with shape (batch_size, channels, height, width)
        Returns
        ------- 
        prediction: torch.Tensor
            4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training: self.eval()
        x = self.forward(x)
        return x

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Set selected layers non trainable, excluding BatchNormalization layers.
        Parameters
        ----------
        non_trainable_layer_idxs: tuple
            Specifies which layers are non-trainable for 
            list of the non trainable layer names.
        """
        non_trainable_layers = [
            self.encoder.feature_info[layer_idx]["module"].replace(".", "_")\
             for layer_idx in non_trainable_layer_idxs
        ]
        for layer in non_trainable_layers:
            for child in self.encoder[layer].children():
                for param in child.parameters():
                    param.requires_grad = False
        return

    def _preprocess_input(self, x, input_range=[0, 1], inplace=False):
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        if x.ndim < 3:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {x.size()}"
            )

        if not inplace:
            x = x.clone()

        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)

# UNet Blocks

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=False
        )
        self.bn = norm_layer(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, scale_factor=2.0, 
        activation=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, activation=activation)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            # x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
            if skip!=None:
                x = F.interpolate(x, size=[skip.size(2), skip.size(3)], mode='bilinear', align_corners=True)
            else:
                x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=True,
            activation=nn.ReLU
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0, 
                activation=activation, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        # list(decoder_channels[:-1][:len(encoder_channels)])
        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]

        out_channels = decoder_channels

        if len(in_channels) != len(out_channels):
            in_channels.append(in_channels[-1]//2)

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x



__available_models__ = [
    'convnext_atto', 'convnext_atto_ols', 'convnext_base', 'convnext_base_384_in22ft1k', 'convnext_base_in22ft1k', 
    'convnext_base_in22k', 'convnext_femto', 'convnext_femto_ols', 'convnext_large', 'convnext_large_384_in22ft1k',
    'convnext_large_in22ft1k', 'convnext_large_in22k', 'convnext_nano', 'convnext_nano_ols', 'convnext_pico', 'convnext_pico_ols',
    'convnext_small', 'convnext_small_384_in22ft1k', 'convnext_small_in22ft1k', 'convnext_small_in22k', 'convnext_tiny',
    'convnext_tiny_384_in22ft1k', 'convnext_tiny_hnf', 'convnext_tiny_in22ft1k', 'convnext_tiny_in22k', 'convnext_xlarge_384_in22ft1k',
    'convnext_xlarge_in22ft1k', 'convnext_xlarge_in22k', 'cs3darknet_focus_l', 'cs3darknet_focus_m', 'cs3darknet_l',
    'cs3darknet_m', 'cs3darknet_x', 'cs3edgenet_x', 'cs3se_edgenet_x', 'cs3sedarknet_l', 'cs3sedarknet_x',
    'cspdarknet53', 'cspresnet50', 'cspresnext50', 'darknet53', 'darknetaa53', 'densenet121', 'densenet161', 'densenet169',
    'densenet201', 'densenetblur121d', 'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4',
    'dm_nfnet_f5', 'dm_nfnet_f6', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'eca_nfnet_l0', 'eca_nfnet_l1', 'eca_nfnet_l2',
    'eca_resnet33ts', 'eca_resnext26ts', 'ecaresnet26t', 'ecaresnet50d', 'ecaresnet50t', 'ecaresnet101d', 'ecaresnet269d',
    'ecaresnetlight', 'edgenext_base', 'edgenext_small', 'edgenext_small_rw', 'edgenext_x_small', 'edgenext_xx_small', 'efficientnet_b0', 
    'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_el', 'efficientnet_el_pruned',
    'efficientnet_em', 'efficientnet_es', 'efficientnet_es_pruned', 'efficientnet_lite0', 'efficientnetv2_rw_m',
    'efficientnetv2_rw_s', 'efficientnetv2_rw_t', 'ese_vovnet19b_dw', 'ese_vovnet39b', 'fbnetc_100',
    'fbnetv3_b', 'fbnetv3_d', 'fbnetv3_g', 'gc_efficientnetv2_rw_t', 'gcresnet33ts', 'gcresnet50t', 'gcresnext26ts',
    'gcresnext50ts', 'gernet_l', 'gernet_m', 'gernet_s', 'ghostnet_100', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b',
    'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 'gluon_resnet101_v1b',
    'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 
    'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d', 
    'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'gluon_xception65', 
    'hardcorenas_a', 'hardcorenas_b', 'hardcorenas_c', 'hardcorenas_d', 'hardcorenas_e', 'hardcorenas_f', 'hrnet_w18', 
    'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 
    'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'lambda_resnet26t', 
    'lambda_resnet50ts', 'lcnet_050', 'lcnet_075', 'lcnet_100', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 
    'legacy_seresnet50', 'legacy_seresnet101', 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 
    'legacy_seresnext101_32x4d', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 'mnasnet_100', 'mnasnet_small', 'mobilenetv2_050', 
    'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140', 'mobilenetv3_large_100', 'mobilenetv3_large_100_miil', 
    'mobilenetv3_large_100_miil_in21k', 'mobilenetv3_rw', 'mobilenetv3_small_050', 'mobilenetv3_small_075', 'mobilenetv3_small_100', 
    'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs', 'mobilevitv2_050', 'mobilevitv2_075', 'mobilevitv2_100', 'mobilevitv2_125', 
    'mobilevitv2_150', 'mobilevitv2_150_384_in22ft1k', 'mobilevitv2_150_in22ft1k', 'mobilevitv2_175', 'mobilevitv2_175_384_in22ft1k', 
    'mobilevitv2_175_in22ft1k', 'mobilevitv2_200', 'mobilevitv2_200_384_in22ft1k', 'mobilevitv2_200_in22ft1k', 'nf_regnet_b1', 
    'nf_resnet50', 'nfnet_l0', 'regnetv_040', 'regnetv_064', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 
    'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 'regnety_002', 'regnety_004', 
    'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080',
    'regnety_120', 'regnety_160', 'regnety_320', 'regnetz_040', 'regnetz_040h', 'regnetz_b16', 'regnetz_c16',
    'regnetz_c16_evos', 'regnetz_d8', 'regnetz_d8_evos', 'regnetz_d32', 'regnetz_e8', 'repvgg_a2', 'repvgg_b0',
    'repvgg_b1', 'repvgg_b1g4', 'repvgg_b2', 'repvgg_b2g4', 'repvgg_b3', 'repvgg_b3g4', 'res2net50_14w_8s',
    'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s', 'res2next50', 
    'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 'resnest101e',
    'resnest200e', 'resnest269e', 'resnet10t', 'resnet14t', 'resnet18', 'resnet18d', 'resnet26',
    'resnet26d', 'resnet26t', 'resnet32ts', 'resnet33ts', 'resnet34', 'resnet34d', 'resnet50', 'resnet50_gn', 
    'resnet50d', 'resnet51q', 'resnet61q', 'resnet101', 'resnet101d', 'resnet152', 'resnet152d', 'resnet200d', 'resnetaa50', 
    'resnetblur50', 'resnetrs50', 'resnetrs101', 'resnetrs152', 'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420', 
    'resnetv2_50', 'resnetv2_50d_evos', 'resnetv2_50d_gn', 'resnetv2_50x1_bit_distilled', 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k', 
    'resnetv2_50x3_bitm', 'resnetv2_50x3_bitm_in21k', 'resnetv2_101', 'resnetv2_101x1_bitm', 'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm', 
    'resnetv2_101x3_bitm_in21k', 'resnetv2_152x2_bit_teacher', 'resnetv2_152x2_bit_teacher_384', 'resnetv2_152x2_bitm', 'resnetv2_152x2_bitm_in21k', 
    'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k', 'resnext26ts', 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 
    'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 'semnasnet_075', 'semnasnet_100', 'seresnet33ts', 'seresnet50', 'seresnet152d', 
    'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26ts', 'seresnext50_32x4d', 'seresnext101_32x8d', 'seresnext101d_32x8d', 
    'seresnextaa101d_32x8d', 'skresnet18', 'skresnet34', 'skresnext50_32x4d', 'spnasnet_100', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 
    'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 
    'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d', 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 
    'tf_efficientnet_b1', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns', 
    'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 
    'tf_efficientnet_b5', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b6_ns', 
    'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 
    'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es', 'tf_efficientnet_l2_ns', 
    'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 
    'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'tf_efficientnetv2_l', 'tf_efficientnetv2_l_in21ft1k', 
    'tf_efficientnetv2_l_in21k', 'tf_efficientnetv2_m', 'tf_efficientnetv2_m_in21ft1k', 'tf_efficientnetv2_m_in21k', 'tf_efficientnetv2_s', 
    'tf_efficientnetv2_s_in21ft1k', 'tf_efficientnetv2_s_in21k', 'tf_efficientnetv2_xl_in21ft1k', 'tf_efficientnetv2_xl_in21k', 'tf_mixnet_l', 
    'tf_mixnet_m', 'tf_mixnet_s', 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 
    'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 'tinynet_a', 'tinynet_b', 'tinynet_c', 'tinynet_d', 'tinynet_e', 'tv_densenet121', 
    'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 
    'vgg19', 'vgg19_bn', 'wide_resnet50_2', 'wide_resnet101_2', 'xception41', 'xception41p', 'xception65', 'xception65p', 'xception71'
]


