import torch
import torch.nn as nn
from torchvision.models import resnet18
# from pthflops import count_ops
from ptflops import get_model_complexity_info

import sys
import networks
import argparse
from options import MonodepthOptions


class SQLdepth(nn.Module):
    def __init__(self, opt):
        super(SQLdepth, self).__init__()
        # self.encoder = networks.LiteResnetEncoderDecoder(model_dim=opt.model_dim) # for resnet18
        self.encoder = networks.ResnetEncoderDecoder(num_layers=opt.num_layers, num_features=opt.num_features, model_dim=opt.model_dim) # for resnet50
        # self.encoder = networks.Unet(pretrained=False, backbone=opt.backbone, in_channels=3, num_classes=opt.model_dim, decoder_channels=opt.dec_channels)

        # self.depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
        #                                                         query_nums=opt.query_nums, num_heads=4,
        #                                                         min_val=opt.min_depth, max_val=opt.max_depth)
        self.depth_decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                                query_nums=opt.query_nums, num_heads=4,
                                                                min_val=opt.min_depth, max_val=opt.max_depth)
    def forward(self, x):
        x = self.encoder(x)
        return self.depth_decoder(x)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opt = options.parser.parse_args()

    device = 'cuda:0'
    model = SQLdepth(opt)
    print("loaded model.")
    # inp = torch.rand(1,32,160,512).to(device)

    print("rand done.")
    # Count the number of FLOPs
    # count_ops(model, inp)
    # macs, params = get_model_complexity_info(model, (64, 192, 640), as_strings=True,
    macs, params = get_model_complexity_info(model, (3, opt.height, opt.width), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
