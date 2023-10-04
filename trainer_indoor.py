# pyright: reportGeneralTypeIssues=warning
from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
# import wandb
from datetime import datetime as dt
import uuid
from collections import OrderedDict
from networks.Unet import Unet
PROJECT = "MCSQLdepth"
experiment_name="Mono"
class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        # assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        # assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.pose_params = []
        self.rectify_params = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales) # default=[0]
        self.num_input_frames = len(self.opt.frame_ids) # default=[0, -1, 1]
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames 
        # default=2 

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        # default=True

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # self.models["encoder"] = networks.Resnet50EncoderDecoder(model_dim=self.opt.model_dim)
        if self.opt.backbone in ["resnet", "resnet_lite"]:
            self.models["encoder"] = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        elif self.opt.backbone == "resnet18_lite":
            self.models["encoder"] = networks.LiteResnetEncoderDecoder(model_dim=self.opt.model_dim)
        elif self.opt.backbone == "eff_b5":
            self.models["encoder"] = networks.BaseEncoder.build(num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        else: 
            self.models["encoder"] = networks.Unet(pretrained=(not self.opt.load_pretrained_model), backbone=self.opt.backbone, in_channels=3, num_classes=self.opt.model_dim, decoder_channels=self.opt.dec_channels)

        if self.opt.load_pretrained_model:
            print("-> Loading pretrained encoder from ", self.opt.load_pt_folder)
            encoder_path = os.path.join(self.opt.load_pt_folder, "encoder.pth")
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["encoder"].state_dict()}
            self.models["encoder"].load_state_dict(filtered_dict_enc)

        self.models["encoder"] = self.models["encoder"].cuda()
        self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"]) 
        # self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.backbone.endswith("_lite"):
            self.models["depth"] = networks.Lite_Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                                    query_nums=self.opt.query_nums, num_heads=4,
                                                                    min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        else:
            self.models["depth"] = networks.Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                                    query_nums=self.opt.query_nums, num_heads=4,
                                                                    min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        if self.opt.load_pretrained_model:
            print("-> Loading pretrained depth decoder from ", self.opt.load_pt_folder)
            depth_decoder_path = os.path.join(self.opt.load_pt_folder, "depth.pth")
            loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)

        self.models["depth"] = self.models["depth"].cuda()
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        # self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["rectify"] = networks.RectifyNet()
        if self.opt.pretrained_rectify:
            print("-> Loading pretrained rectify model from ", self.opt.pose_net_path)
            rectify_path = os.path.join(self.opt.pose_net_path, "rectify.pth")
            loaded_dict_enc = torch.load(rectify_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["rectify"].state_dict()}
            self.models["rectify"].load_state_dict(filtered_dict_enc)

        self.models["rectify"] = self.models["rectify"].cuda()
        if self.opt.diff_lr :
            print("using diff lr for depth-net and rectify-net")
            self.rectify_params += list(self.models["rectify"].parameters())
        else :
            self.parameters_to_train += list(self.models["rectify"].parameters())


        self.models["pose"] = networks.PoseCNN(
            self.num_input_frames if self.opt.pose_model_input == "all" else 2)
            # default=2
        if self.opt.pretrained_pose :
            print("-> Loading pretrained pose-net from ", self.opt.pose_net_path)
            pose_weight_path = os.path.join(self.opt.pose_net_path, "pose.pth")
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in torch.load(pose_weight_path).items()])
            self.models["pose"].load_state_dict(state_dict)

        # self.models["pose"].to(self.device)
        self.models["pose"] = self.models["pose"].cuda()
        # self.models["pose"] = torch.nn.DataParallel(self.models["pose"])
        if self.opt.diff_lr :
            print("using diff lr for depth-net and pose-net")
            self.pose_params += list(self.models["pose"].parameters())
        else :
            self.parameters_to_train += list(self.models["pose"].parameters())

        # if self.opt.predictive_mask:
        #     assert self.opt.disable_automasking, \
        #         "When using predictive_mask, please disable automasking with --disable_automasking"

        #     # Our implementation of the predictive masking baseline has the the same architecture
        #     # as our depth decoder. We predict a separate mask for each source frame.
        #     self.models["predictive_mask"] = networks.DepthDecoder(
        #         self.models["encoder"].num_ch_enc, self.opt.scales,
        #         num_output_channels=(len(self.opt.frame_ids) - 1))
        #     self.models["predictive_mask"].to(self.device)
        #     self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        if self.opt.diff_lr :
            df_params = [{"params": self.pose_params, "lr": 0.1*self.opt.learning_rate}, {"params": self.rectify_params, "lr": 0.1*self.opt.learning_rate},
                      {"params": self.parameters_to_train, "lr": self.opt.learning_rate}]
            # self.model_optimizer = optim.AdamW(df_params, weight_decay=0.001, lr=self.opt.learning_rate)
            self.model_optimizer = optim.Adam(df_params, lr=self.opt.learning_rate)
        else : 
            # self.model_optimizer = optim.AdamW(self.parameters_to_train, weight_decay=0.001, lr=self.opt.learning_rate)
            self.model_optimizer = optim.Adam(self.parameters_to_train, lr=self.opt.learning_rate)
        # default=1e-4

        if self.opt.load_adam:
            optimizer_load_path = os.path.join(self.opt.load_pt_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                # optimizer_dict = torch.load(optimizer_load_path)
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                filtered_dict = {k: v for k, v in optimizer_dict.items() if k in self.model_optimizer.state_dict()}
                self.model_optimizer.load_state_dict(filtered_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
        # if self.opt.load_weights_folder is not None:
        #     self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nyu_raw": datasets.NYUrawDataset,
                         "mc_dataset": datasets.MCDataset,
                         "mc_mini_dataset": datasets.MCDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        # default="kitti"

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.dataset == "mc_dataset":
            train_dataset = self.dataset(
                self.opt.intrinsics_file_path, self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, num_scales=1, is_train=True, img_ext=img_ext)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        elif self.opt.dataset == "nyu_raw":
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, num_scales=1, is_train=True, img_ext=img_ext)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, num_scales=1, is_train=True, img_ext=img_ext)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 1, is_train=False, img_ext=img_ext)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        # self.model_lr_scheduler = optim.lr_scheduler.OneCycleLR(self.model_optimizer, self.opt.learning_rate, 
        #                                   epochs=self.opt.num_epochs, steps_per_epoch=len(self.train_loader),
        #                                   cycle_momentum=True,
        #                                   base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
        #                                   div_factor=10, final_div_factor=100)


        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {} # convert depth map to cam_points
        self.project_3d = {} # project cam_points to pix_coords
        self.project_depth = {} # project cam_points to depth map
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
            self.project_depth[scale] = ProjectDepth(self.opt.batch_size, h, w)
            self.project_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        # run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{self.opt.batch_size}-tep{self.epoch}-lr{self.opt.learning_rate}--{uuid.uuid4()}"
        # name = f"{experiment_name}_{run_id}"
        # wandb.init(project=PROJECT, name=name, config=self.opt, dir='.')
        self.save_model()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        accumulation_steps = self.opt.accumulation_steps

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            losses["loss"] /= accumulation_steps
            losses["loss"].backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()

            duration = time.time() - before_op_time

            should_log = True
            # if should_log and self.step % 5 == 0:
                # wandb.log({f"Train/reprojection_loss": losses["loss"].item()}, step=self.step)
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 100 == 0
            time_to_save = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                # self.val() # for kitti only
            if time_to_save:
                self.save_model()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        if self.opt.pose_model_type == "shared": # default is no
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)
            if self.opt.use_rectify_net: # rectify images
                frame_id_prev = self.opt.frame_ids[1] # ensure that frame_ids[0] == 0
                frame_id_next = self.opt.frame_ids[2]
                ref_imgs = [inputs["color", frame_id_prev, 0], inputs["color", frame_id_next, 0]]  
                ref_imgs_warped, loss_rc, loss_rt, rot_before, rot_after = self.rectify_imgs(inputs["color", 0, 0], ref_imgs, inputs["K3x3", 0])
                inputs["color_warped", frame_id_prev, 0] = ref_imgs_warped[0]
                inputs["color_warped", frame_id_next, 0] = ref_imgs_warped[1]
            else:
                frame_id_prev = self.opt.frame_ids[1] # ensure that frame_ids[0] == 0
                frame_id_next = self.opt.frame_ids[2]
                ref_imgs_warped = [inputs["color", frame_id_prev, 0], inputs["color", frame_id_next, 0]]  

        if self.opt.predictive_mask:
            # default no
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
        if self.use_pose_net:
            # default true
            outputs.update(self.predict_poses(inputs, features))

        # if not self.opt.use_photo_geo_loss:
        if self.opt.use_photo_geo_loss or self.opt.use_improved_mini_reproj_loss:
            for f_i in self.opt.frame_ids[1:]:
                if self.opt.use_rectify_net:
                    depth_of_ref = (self.models["depth"]( self.models["encoder"](inputs["color_warped", f_i, 0]) ))["disp", 0] 
                else:
                    depth_of_ref = (self.models["depth"]( self.models["encoder"](inputs["color", f_i, 0]) ))["disp", 0] 
                outputs[("depth_ref", f_i, 0)] = F.interpolate(depth_of_ref, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)                    
        if True:
            self.generate_images_pred(inputs, outputs) # warp ref images (either raw color or warped) into curr view
        if self.opt.use_photo_geo_loss: 
            total_loss = 0
            losses = {}
            disp_HW = outputs[("depth", 0, 0)]
            if self.opt.use_mini_reprojection_loss: # the min-reprojection loss in monodepth2
                total_loss, losses = self.compute_losses(inputs, outputs)
            elif self.opt.use_photo_geo_loss:
                ref_depths = []
                for f_i in self.opt.frame_ids[1:]:
                    ref_depths.append(outputs[("depth_ref", f_i, 0)])

                losses["loss_photo"], losses["loss_geo"] = self.photo_and_geometry_loss_imp(outputs, tgt_img=inputs["color", 0, 0], 
                                                                                            ref_imgs=ref_imgs_warped, tgt_depth=disp_HW)
                total_loss += self.opt.loss_photo_weight * losses["loss_photo"]
                total_loss += self.opt.loss_geo_weight * losses["loss_geo"]
                # get smoothness loss 
                losses["smoothness"] = compute_smooth_loss(disp_HW, inputs["color", 0, 0])
                total_loss += self.opt.disparity_smoothness * losses["smoothness"]
            if self.opt.use_rectify_net:
                # get rot_consistency loss and rotate triplet loss
                losses["loss_rc"] = loss_rc
                losses["loss_rt"] = loss_rt
                total_loss += self.opt.loss_rc_weight * losses["loss_rc"]
                total_loss += self.opt.loss_rt_weight * losses["loss_rt"]
            total_loss /= self.num_scales
            losses["loss"] = total_loss
        elif self.opt.use_improved_mini_reproj_loss:
            total_loss, losses = self.compute_losses_with_occ(inputs, outputs)
            if self.opt.use_rectify_net:
                losses["loss_rc"] = loss_rc
                losses["loss_rt"] = loss_rt
                total_loss += self.opt.loss_rc_weight * losses["loss_rc"]
                total_loss += self.opt.loss_rt_weight * losses["loss_rt"]
            total_loss /= self.num_scales
            losses["loss"] = total_loss
        else:
            losses = self.compute_losses(inputs, outputs)
        
        # if self.opt.use_geometry_loss:
        #     losses["loss_photo"] = loss_photo
        #     losses["loss_geo"] = loss_geo


        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # if use 2 frames for pose estimation, that is, [prev, curr]
        if self.num_pose_frames == 2:
            # default is True
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.
            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                # default go this branch
                if self.opt.use_rectify_net:
                    pose_feats = {f_i: inputs["color_warped", f_i, 0] for f_i in self.opt.frame_ids[1:]} 
                    pose_feats[0] = inputs["color", 0, 0] 
                else:
                    pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        # default go this branch
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next() # for old pytorch
            inputs = next(self.val_iter) # for new pytorch
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            depth = disp 
            # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            # outputs[("depth", 0, scale)] = disp

            for i, frame_id in enumerate(self.opt.frame_ids[1:]): #

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn" and frame_id != "s" and not self.opt.use_stereo: # default true, 

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                    T_inv = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id > 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]) 
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T) 
                # cam_points_of_ref = self.backproject_depth[source_scale](
                #     outputs["depth_ref", frame_id, source_scale], inputs[("inv_K", source_scale)]) 
                # pix_coords_in_curr = self.project_3d[source_scale](
                #     cam_points_of_ref, inputs[("K", source_scale)], T_inv) 
                # pix_coords: [bs, h, w, 2]

                # outputs[("sample", frame_id, scale)] = pix_coords

                if self.opt.use_rectify_net:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color_warped", frame_id, source_scale)],
                        pix_coords,
                        padding_mode="border",
                        align_corners=True) 
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        pix_coords,
                        padding_mode="border",
                        align_corners=True) 

                if self.opt.use_photo_geo_loss or self.opt.use_improved_mini_reproj_loss:
                    # outputs[("pred_dep", frame_id, scale)] = F.grid_sample(
                    #     outputs[("depth", 0, 0)],
                    #     pix_coords_in_curr, 
                    #     padding_mode="border",
                    #     align_corners=True) # dep_of_warped: depth of ref_color(may be color_warped)
                    outputs[("pred_dep", frame_id, scale)] = F.grid_sample(
                        outputs[("depth_ref", frame_id, source_scale)],
                        pix_coords,
                        padding_mode="border",
                        align_corners=True) # dep_of_warped: depth of ref_color(may be color_warped)
                    # ref_cam_depths = self.backproject_depth[source_scale](
                    #     outputs["depth_ref", frame_id, source_scale], inputs[("inv_K", source_scale)]) 
                    # outputs["com_depth", frame_id, source_scale] = self.project_depth[source_scale](
                    #     cam_points_of_ref, inputs[("K", source_scale)], T_inv)
                    # outputs["com_depth", frame_id, source_scale] = self.project_depth[source_scale](
                    #     cam_points, inputs[("K", source_scale)], T)
                else:
                    pass

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = self.opt.ssim_weight * ssim_loss + self.opt.l1_weight * l1_loss

        return reprojection_loss

    def compute_losses_with_occ(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            L1_reg_term = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)] 
                valid_mask = (pred.abs().mean(dim=1, keepdim=True) > 1e-3).float() 
                reprojection_err = self.compute_reprojection_loss(pred, target)
                # ref_depth = outputs["depth_ref", frame_id, 0] 
                projected_depth = outputs["pred_dep", frame_id, 0] 
                # computed_depth = outputs["com_depth", frame_id, 0] 
                computed_depth = outputs["depth", 0, 0] 
                diff_depth = (computed_depth-projected_depth).abs() / \
                    (computed_depth+projected_depth) 
                L1_reg_term.append(diff_depth * valid_mask) # L1 regularization
                # outputs["diff_depth", frame_id, 0] = diff_depth
                # weight_mask = (1-diff_depth).detach() 
                weight_mask = (1-torch.sqrt(1-(diff_depth-1)**2))
                weight_mask = weight_mask.detach()
                # outputs["diff_depth", frame_id, 0] = weight_mask
                reprojection_err = reprojection_err * weight_mask * valid_mask 
                reprojection_losses.append(reprojection_err) # min-reprojection

            reprojection_losses = torch.cat(reprojection_losses, 1)
            L1_reg_losses = torch.cat(L1_reg_term, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            L1_reg_loss = L1_reg_losses.mean(1, keepdim=True)
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
                # L1_reg_loss = L1_reg_losses
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                # L1_reg_loss, _ = torch.min(L1_reg_losses, dim=1)

            # if not self.opt.disable_automasking:
            #     outputs["identity_selection/{}".format(scale)] = (
            #         idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            loss += self.opt.reg_wt * L1_reg_loss.mean()
            # disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # if GPU memory is not enough, downsample color instead
            if norm_disp.shape[-2:] != [self.opt.height, self.opt.width]:
                bs, c, h, w = norm_disp.shape
                color = F.interpolate(color, [h, w], mode="bilinear", align_corners=False)
            smooth_loss = 0
            smooth_loss = get_smooth_loss(norm_disp, color)
            # smooth_loss
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        if not self.opt.use_rectify_net:
            total_loss /= self.num_scales
            losses["loss"] = total_loss
            return total_loss, losses
        else:
            return total_loss, losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                        if self.opt.use_rectify_net:
                            writer.add_image(
                                "warped_{}_{}/{}".format(frame_id, s, j),
                                inputs[("color_warped", frame_id, s)][j].data, self.step)
                        # writer.add_image(
                        #     "depth_ref_{}_{}/{}".format(frame_id, s, j),
                        #     normalize_image(inputs[("depth_ref", frame_id, s)][j]), self.step)
                        # writer.add_image(
                        #     "pred_dep_{}/{}".format(s, j),
                        #     normalize_image(outputs[("pred_dep", frame_id, s)][j]), self.step)
                        # if frame_id != "s":
                            # writer.add_image(
                            #     "com_depth_{}_{}/{}".format(frame_id, s, j),
                            #     normalize_image(outputs[("com_depth", frame_id, s)][j]), self.step) 
                            # writer.add_image(
                            #     "diff_depth_{}_{}/{}".format(frame_id, s, j),
                            #     outputs[("diff_depth", frame_id, s)][j], self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                # if self.opt.predictive_mask:
                #     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                #         writer.add_image(
                #             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                #             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                #             self.step)

                # elif not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk, default /home/Process3/tmp/mdp/models/
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            # for nn.DataParallel models, you must use model.module.state_dict() instead of model.state_dict()
            if model_name == 'pose' or model_name == 'rectify': # pose and rectify is not DDP model
               to_save = model.state_dict()
            else:
                to_save = model.module.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def rectify_imgs(self, tgt_img, ref_imgs, intrinsics):

        # use arn to pre-warp ref images
        rot1_list = []
        rot2_list = []
        rot3_list = []
        rot3_gt_list = []
        ref_imgs_warped = []

        for ref_img in ref_imgs:
            rot1 = self.models["rectify"](tgt_img, ref_img)
            rot_warped_img = inverse_rotation_warp(ref_img, rot1, intrinsics)
            rot1_list += [rot1]

            rot2 = self.models["rectify"](tgt_img, rot_warped_img)
            rot2_list += [rot2]

            # When rot1 is not too large or small, we use rot1 to supervise
            # rot3. Otherwise, we generate a random rotation for supervising
            # rot3. The random rotation ranges from -0.05 to +0.05.
            # if rot1.abs().mean() > 0.02 and rot1.abs().mean() < 0.1: # hard code?
            if True:
                rot3_gt = rot1.clone().detach()
                warp_ref_img = rot_warped_img.clone().detach()
            else:
                rot3_gt = (torch.rand_like(rot1).type_as(rot1) - 0.5) * 0.1
                warp_ref_img = inverse_rotation_warp(
                    ref_img, rot3_gt, intrinsics)
            rot3_gt_list.append(rot3_gt)

            rot3 = self.models["rectify"](warp_ref_img, ref_img)
            rot3_list += [rot3]

            ref_imgs_warped.append(rot_warped_img)

        #
        rot1 = torch.stack(rot1_list)
        rot2 = torch.stack(rot2_list)
        rot3 = torch.stack(rot3_list)
        rot3_gt = torch.stack(rot3_gt_list)

        # rot_consistency, rot1 or random rot as gt for rot3
        loss_rot_consistency = (rot3 - rot3_gt).abs().mean()

        # triplet loss, abs(rot2) should smaller than abs(rot1)
        loss_rot_triplet = (rot2.abs() - rot1.abs() + 0.05).clamp(min=0).mean()

        return ref_imgs_warped, loss_rot_consistency, loss_rot_triplet, rot1.abs().mean(), rot2.abs().mean()

    def photo_and_geometry_loss_imp(self, outputs, tgt_img, ref_imgs, tgt_depth):

        diff_img_list = []
        diff_color_list = []
        diff_depth_list = []
        valid_mask_list = []
        geo_diff_list = []

        for ref_img, f_i in zip(ref_imgs, self.opt.frame_ids[1:]):
            diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1, geo_diff_tmp1 = self.compute_pairwise_loss_imp(f_i, outputs,
                tgt_img, ref_img, tgt_depth,
            )
            # diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2 = self.compute_pairwise_loss_imp(f_i, inputs, outputs,
            #     ref_img, tgt_img, ref_depth,
            #     pose_inv, intrinsics,
            # )
            #     hparams
            # diff_img_list += [diff_img_tmp1, diff_img_tmp2]
            # diff_color_list += [diff_color_tmp1, diff_color_tmp2]
            # diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
            # valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]
            diff_img_list.append(diff_img_tmp1)
            diff_color_list.append(diff_color_tmp1)
            diff_depth_list.append(diff_depth_tmp1)
            valid_mask_list.append(valid_mask_tmp1)
            geo_diff_list.append(geo_diff_tmp1)

        diff_img = torch.cat(diff_img_list, dim=1) # photo_loss
        diff_color = torch.cat(diff_color_list, dim=1) 
        diff_depth = torch.cat(diff_depth_list, dim=1) 
        valid_mask = torch.cat(valid_mask_list, dim=1)
        geo_diff = torch.cat(geo_diff_list, dim=1) # geometry_loss

        # using photo loss to select best match in multiple views
        # if not hparams.no_min_optimize:
        if True:
            indices = torch.argmin(diff_color, dim=1, keepdim=True)

            diff_img = torch.gather(diff_img, 1, indices)
            diff_depth = torch.gather(diff_depth, 1, indices)
            valid_mask = torch.gather(valid_mask, 1, indices)
            geo_diff = torch.gather(geo_diff, 1, indices)

        photo_loss = mean_on_mask(diff_img, valid_mask)
        # geometry_loss = mean_on_mask(diff_depth, valid_mask)
        geometry_loss = mean_on_mask(geo_diff, valid_mask)

        return photo_loss, geometry_loss

    def compute_pairwise_loss_imp(self, f_i, outputs, tgt_img, ref_img, tgt_depth):

        ref_img_warped = outputs["color", f_i, 0] 
        ref_depth = outputs["depth_ref", f_i, 0] 
        projected_depth = outputs["pred_dep", f_i, 0] 
        computed_depth = outputs["com_depth", f_i, 0] 
        # outputs["computed_depth", f_i, 0] = computed_depth

        diff_depth = (computed_depth-projected_depth).abs() / \
            (computed_depth+projected_depth) 
        outputs["diff_depth", f_i, 0] = diff_depth
        geo_diff = (computed_depth - ref_depth).abs() #.mean(1, True)
        geo_diff = torch.mean(geo_diff, dim=1, keepdim=True)

        # masking zero values
        valid_mask_ref = (ref_img_warped.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask = valid_mask_tgt * valid_mask_ref

        diff_color = (tgt_img-ref_img_warped).abs().mean(dim=1, keepdim=True) 

        identity_warp_err = (tgt_img-ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

        diff_img = (tgt_img-ref_img_warped).abs().clamp(0, 1)

        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        diff_img = torch.mean(diff_img, dim=1, keepdim=True)

        weight_mask = (1-diff_depth).detach() 
        # weight_mask = (1-diff_depth) 
        diff_img = diff_img * weight_mask
        geo_diff = geo_diff * weight_mask

        return diff_img, diff_color, diff_depth, valid_mask, geo_diff


