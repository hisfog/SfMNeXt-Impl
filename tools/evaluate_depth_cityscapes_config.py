import os
import sys
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import skimage
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import readlines
from options import MonodepthOptions
from layers import disp_to_depth
import datasets
import networks
# from .dynamicdepth import datasets, networks
# from layers import transformation_from_parameters, disp_to_depth, grad_computation_tools
import tqdm
# import matplotlib as mpl
# import matplotlib.cm as cm
import matplotlib.pyplot as plt


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def compute_matching_mask(monodepth, lowest_cost):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        matching_depth = 1 / lowest_cost.unsqueeze(1).cuda()

        # mask where they differ by a large amount
        mask = ((matching_depth - monodepth) / monodepth) < 1.0
        mask *= ((monodepth - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    # if opt.use_future_frame == 'true':
    #     frames_to_load.append(1)
    # for idx in range(-1, -1 - opt.num_matching_frames, -1):
    #     if idx not in frames_to_load:
    #         frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
        
        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.eval_data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False)
        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False)
        dataloader = DataLoader(dataset, 4, shuffle=False, num_workers=8,
                                pin_memory=True, drop_last=False)


        encoder = networks.BaseEncoder.build(model_dim=opt.model_dim)
        depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                                query_nums=opt.query_nums, num_heads=4,
                                                                min_val=opt.min_depth, max_val=opt.max_depth)
                                                                # min_val=0.001, max_val=10.0)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder = torch.nn.DataParallel(encoder)
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder = torch.nn.DataParallel(depth_decoder)
        depth_decoder.eval()

        # if torch.cuda.is_available():
        #     encoder.cuda()
        #     depth_decoder.cuda()

        pred_disps = []
        src_imgs = []
        error_maps = []
        # mono_disps = []

        if opt.eval_split == 'cityscapes':
            print('loading cityscapes gt depths individually due to their combined size!')
            gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
        else:
            gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                    # print(input_color.shape, "==") # [16, 3, 192, 512]

                
                if opt.post_process:
                    # print("post_process *********")
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                # pred_disp = output[("disp", 0)]
                pred_disp = pred_disp.cpu()[:, 0].numpy()


                if opt.post_process:
                    n = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:n], pred_disp[n:, :, ::-1])

                pred_disps.append(pred_disp)
                # src_imgs.append(data[("color", 0, 0)])

                # monodisp, _ = disp_to_depth(teacher_output[("disp", 0)], opt.min_depth, opt.max_depth)
                # monodisp = monodisp.cpu()[:, 0].numpy()
                # mono_disps.append(monodisp)
                    

        pred_disps = np.concatenate(pred_disps)
        # src_imgs = np.concatenate(src_imgs)

        print('finished predicting!')
        if opt.save_pred_disps:
            output_path = os.path.join(
                opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

            # src_imgs_path = os.path.join(
            #     opt.load_weights_folder, "src_{}_split.npy".format(opt.eval_split))
            # print("-> Saving src imgs to ", src_imgs_path)
            # np.save(src_imgs_path, src_imgs)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]


    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if opt.eval_split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # print(gt_height, "print(gt_height)") 1024
            # print(gt_width, "print(gt_width)") 2048
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp
        # pred_depth = 1 / pred_disp

        # mono_disp = np.squeeze(mono_disps[i][0])
        # mono_disp = cv2.resize(mono_disp, (gt_width, gt_height))
        # mono_depth = 1 / mono_disp

        if opt.eval_split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]
            # 768, 2048
            # mono_depth = mono_depth[256:, 192:1856]

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        elif opt.eval_split == 'cityscapes':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            mask = gt_depth > 0
        
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            ratios.append(ratio)
            pred_depth *= ratio

            # mono_depth *= np.median(gt_depth[mask]) / np.median(mono_depth[mask])
        
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        # mono_depth[mono_depth < MIN_DEPTH] = MIN_DEPTH
        # mono_depth[mono_depth > MAX_DEPTH] = MAX_DEPTH
        
        error_map = np.abs(gt_depth - pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        error_map = np.multiply(error_map, mask)
        # error_maps.append(error_map)

        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.save_pred_disps:
        print("saving errors")
        # if opt.zero_cost_volume:
        if True:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_errors.npy".format(tag, opt.eval_split))
        np.save(output_path, np.array(errors))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    if opt.save_pred_disps:
        # error_maps = np.concatenate(error_maps) # should not concatenate
        error_map_path = os.path.join(
            opt.load_weights_folder, "error_{}_split".format(opt.eval_split))
        print("-> Saving error maps to ", error_map_path)
        # np.save(error_map_path, error_maps)
        # np.savez_compressed(error_map_path, data=np.array(error_maps, dtype="object"))
    mean_errors = np.array(errors).mean(0)
    print(mean_errors)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


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
    evaluate(opt)
    # options = MonodepthOptions()
    # evaluate(options.parse())

