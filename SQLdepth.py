import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import hub
import argparse
import os
import networks

class SQLdepth(nn.Module):
    def __init__(self, opt):
        super(SQLdepth, self).__init__()
        self.opt = opt
        if opt.model_type == "cvnxt_L":
            self.encoder = networks.Unet(pretrained=(not opt.load_pretrained_model), backbone='convnext_large', in_channels=3, num_classes=opt.model_dim, decoder_channels=opt.dec_channels)
        elif opt.backbone in ["resnet", "resnet_lite"]:
            self.encoder = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        elif opt.model_type in ["nyu_pth_model", "eff_b5"]:
            self.encoder = BaseEncoder.build(num_features=opt.num_features, model_dim=opt.model_dim)
        else:
            self.encoder = networks.Unet(pretrained=(not opt.load_pretrained_model), backbone=opt.backbone, in_channels=3, num_classes=opt.model_dim, decoder_channels=opt.dec_channels)

        if self.opt.backbone.endswith("_lite"):
            self.depth_decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                             query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        else:
            self.depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                        query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)

        if self.opt.load_pretrained_model:
            self.load_pretrained_model()

    def load_pretrained_model(self):
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        print("-> Loading pretrained encoder from ", self.opt.load_pt_folder)
        encoder_path = os.path.join(self.opt.load_pt_folder, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        print("-> Loading pretrained depth decoder from ", self.opt.load_pt_folder)
        depth_decoder_path = os.path.join(self.opt.load_pt_folder, "depth.pth")
        loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
        # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_decoder.state_dict()}
        # self.depth_decoder.load_state_dict(filtered_dict_enc)
        self.depth_decoder.load_state_dict(loaded_dict_enc)


    def forward(self, x):
        x = self.encoder(x)
        return self.depth_decoder(x)["disp", 0]

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

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

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
    def __init__(self, backend, model_dim=128, num_features=2048):
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
        basemodel = hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)

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


class Depth_Decoder_QueryTr(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=16, num_heads=4, query_nums=100, dim_out=256, norm='linear',
                 min_val=0.001, max_val=10) -> None:
        super(Depth_Decoder_QueryTr, self).__init__()
        self.norm = norm
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)
        encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        # encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=512) # for resnet18
        self.transformer_encoder = nn.modules.transformer.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

        self.full_query_layer = FullQueryLayer()
        self.bins_regressor = nn.Sequential(nn.Linear(embedding_dim*query_nums, 16*query_nums),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*query_nums, 16*16),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*16, dim_out))

        self.convert_to_prob = nn.Sequential(nn.Conv2d(query_nums, dim_out, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.query_nums = query_nums

        self.min_val = min_val
        self.max_val = max_val

    # @get_local('x0')
    def forward(self, x0):
        embeddings_0 = self.embedding_convPxP(x0.clone())
        embeddings_0 = embeddings_0.flatten(2)
        embeddings_0 = embeddings_0 + self.positional_encodings[:embeddings_0.shape[2], :].T.unsqueeze(0)
        embeddings_0 = embeddings_0.permute(2, 0, 1)
        total_queries = self.transformer_encoder(embeddings_0)

        x0 = self.conv3x3(x0)
        queries = total_queries[:self.query_nums, ...]
        queries = queries.permute(1, 0, 2)

        energy_maps, summarys = self.full_query_layer(x0, queries)
        bs, Q, E = summarys.shape
        y = self.bins_regressor(summarys.view(bs, Q*E))

        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), energy_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        
        out = self.convert_to_prob(energy_maps)
        bin_widths = (self.max_val - self.min_val) * y
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        outputs = {}
        outputs["disp", 0] = pred
        # outputs["attn", 0] = out
        # outputs["bins", 0] = bin_edges
        return outputs

class FullQueryLayer(nn.Module):
    def __init__(self) -> None:
        super(FullQueryLayer, self).__init__()
    def forward(self, x, K):
        """
        given feature map of size [bs, E, H, W], and queries of size [bs, Q, E]
        return Q energy maps corresponding to Q queries of shape [bs, Q, H, W]
        and add feature noise to x of the same shape as input [bs, E, H, W]
        and summary_embedding of shape [bs, Q, E]
        """
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))
        y_norm = torch.softmax(y, dim=1)
        summary_embedding = torch.matmul(y_norm.permute(0, 2, 1), x.view(n, c, h*w).permute(0, 2, 1))
        y = y.permute(0, 2, 1).view(n, cout, h, w)
        return y, summary_embedding

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

class MonodepthOptions:
    def __init__(self):
        # self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options", fromfile_prefix_chars='@')
        self.parser.convert_arg_line_to_args = convert_arg_line_to_args

        # PATHS
        self.parser.add_argument("--eval_data_path",
                                 type=str,
                                 help="path to the evaluation data",
                                 default='data/CS_RAW/')
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="/home/Process3/KITTI_depth")#os.path.join(file_dir, ".."))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "cityscapes_preprocessed"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_features",
                                 type=int,
                                 help="resnet feature dim",
                                 default=512)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "cityscapes_preprocessed"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true",
                                 default='.png')
        self.parser.add_argument("--dim_out",
                                 type=int,
                                 help="number of bins",
                                 default=128)
        self.parser.add_argument("--query_nums",
                                 type=int,
                                 help="number of queries, should be less than h*w/p^2",
                                 default=128)
        self.parser.add_argument("--patch_size",
                                 type=int,
                                 help="patch size before ViT",
                                 default=20)
        self.parser.add_argument("--model_dim",
                                 type=int,
                                 help="model dim",
                                 default=32)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=320)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=1024)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
                                 # default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.001)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80)
        self.parser.add_argument("--use_rectify_net",
                         help="if set, uses RectifyNey for training",
                         action="store_true")
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--load_pretrained_model",
                                 help="if set, uses pretrained encoder and depth decoder for training",
                                 action="store_true")
        self.parser.add_argument("--zoe_dev_pt_path",
                                 type=str,
                                 help="path to pretrained zoe_dev model")
        self.parser.add_argument("--load_pt_folder",
                                 type=str,
                                 help="path to pretrained model")
        self.parser.add_argument("--pose_net_path",
                                 help="path to pretrained pose net",
                                 type=str,
                                 default="/home/Process3/tmp/mdp/models_22_6_27/models/weights_19/",)
        self.parser.add_argument("--pretrained_pose",
                                 help="if set, uses pretrained posenet for training",
                                 action="store_true")
        self.parser.add_argument("--log_attn",
                                 help="if set, log attn maps in evaluation",
                                 action="store_true")
        self.parser.add_argument("--diff_lr",
                                 help="if set, uses different lr for training",
                                 action="store_true")
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="posecnn",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=10)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "cityscapes"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--eval_dataset",
                                 help="dataset to eval",
                                 type=str)
        self.parser.add_argument("--backbone",
                                 help="Unet backbone type",
                                 default="tf_efficientnet_b5_ap",
                                 type=str)
        self.parser.add_argument("--model_type",
                                 help="model type",
                                 default="eff_b5",
                                 type=str)
        self.parser.add_argument("--dec_channels",
                                 nargs="+",
                                 type=int,
                                 help="decoder channels in Unet",
                                 default=[1536, 768, 384, 192, 96])
        self.parser.add_argument('--image_path', type=str,
                                help='path to a test image or folder of images')
        self.parser.add_argument('--ext', type=str,
                                help='image extension to search for in folder', default="png")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)



