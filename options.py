# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        # self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options", fromfile_prefix_chars='@')

        # PATHS
        self.parser.add_argument("--intrinsics_file_path",
                                 type=str,
                                 help="path to the camera intrinsics file",
                                 default='./splits/mc_dataset/KV_intrinsics.txt')
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
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", 
                                          "cityscapes_preprocessed", "mc_dataset", "mc_mini_dataset", "nyu_raw"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_features",
                                 type=int,
                                 help="resnet or efficient-net feature dim",
                                 default=512)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dec_channels",
                                 nargs="+",
                                 type=int,
                                 help="decoder channels in Unet",
                                 default=[1024, 512, 256, 128])
        self.parser.add_argument("--backbone",
                                 type=str,
                                 help="backbone in the Unet",
                                 default="convnext_large")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", 
                                          "cityscapes_preprocessed", "mc_dataset", "mc_mini_dataset", "nyu_raw"])
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
        self.parser.add_argument("--reg_wt",
                                 type=float,
                                 help="regularization term weight",
                                 default=0.01)
        self.parser.add_argument("--feat_wt",
                                 type=float,
                                 help="feature metric loss weight",
                                 default=0.01)
        self.parser.add_argument("--l1_weight",
                                 type=float,
                                 help="L1 loss weight",
                                 default=0.15)
        self.parser.add_argument("--ssim_weight",
                                 type=float,
                                 help="SSIM loss weight",
                                 default=0.85)
        self.parser.add_argument("--use_mini_reprojection_loss",
                                 help="if set, uses min_reproj loss in monodepth2 for training",
                                 action="store_true")
        self.parser.add_argument("--use_improved_mini_reproj_loss",
                                 help="if set, uses photometric loss with occ mask for training",
                                 action="store_true")
        self.parser.add_argument("--use_photo_geo_loss",
                                 help="if set, uses photo and geo loss for training",
                                 action="store_true")
        self.parser.add_argument("--use_flow_pose",
                                 help="if set, uses PoseFlow for training",
                                 action="store_true")
        self.parser.add_argument("--loss_geo_weight",
                                 type=float,
                                 help="geometry loss weight",
                                 default=1.0)
        self.parser.add_argument("--loss_photo_weight",
                                 type=float,
                                 help="photo loss weight",
                                 default=1.0)
        self.parser.add_argument("--loss_rt_weight",
                                 type=float,
                                 help="RT loss weight",
                                 default=1.0)
        self.parser.add_argument("--loss_rc_weight",
                                 type=float,
                                 help="RC loss weight",
                                 default=1.0)
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
                                 default=80.0)
        self.parser.add_argument("--use_optical_flow",
                                 help="if set, uses optical flow for training",
                                 action="store_true")
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
                                 # default=[0, 1])
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--pretrained_flow",
                                 help="if set, uses pretrained flow net for training",
                                 action="store_true")
        self.parser.add_argument("--pretrained_rectify",
                                 help="if set, uses pretrained rectify net for training",
                                 action="store_true")
        self.parser.add_argument("--load_adam",
                                 help="if set, uses load adam state for training",
                                 action="store_true")
        self.parser.add_argument("--load_pretrained_model",
                                 help="if set, uses pretrained encoder and depth decoder for training",
                                 action="store_true")
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
        self.parser.add_argument("--multi_gpu",
                                 help="if set, uses torch.DDP for training",
                                 action="store_true")
        self.parser.add_argument("--diff_lr",
                                 help="if set, uses different lr for training",
                                 action="store_true")
        self.parser.add_argument("--accumulation_steps",
                                 type=int,
                                 help="accumulation steps",
                                 default=1)
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
                              #    default="separate_resnet",
                                 default="posecnn",
                                 choices=["posecnn", "pose_flow", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--pred_metric_depth",
                                help='if set, predicts metric depth instead of disparity. (This only '
                                     'makes sense for stereo-trained KITTI models).',
                                action='store_true')
        self.parser.add_argument('--ext', type=str,
                                help='image extension to search for in folder', default="png")
        self.parser.add_argument('--image_path', type=str,
                                help='path to a test image or folder of images')
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

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
