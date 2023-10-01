from __future__ import absolute_import, division, print_function

import os
# import skimage.transform
import numpy as np
import PIL.Image as pil

# from kitti_utils import generate_depth_map
from .mono_dataset_nyu import MonoDatasetSingleCam


class NYUDataset(MonoDatasetSingleCam):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        #         % RGB Intrinsic Parameters
        # fx_rgb = 5.1885790117450188e+02;
        # fy_rgb = 5.1946961112127485e+02;
        # cx_rgb = 3.2558244941119034e+02;
        # cy_rgb = 2.5373616633400465e+02;
        self.K = np.array([[0.8107, 0, 0.5087, 0],
                           [0, 1.0822, 0.5286, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 480)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    # def check_depth(self):
    #     line = self.filenames[0].split()
    #     scene_name = line[0]
    #     frame_index = int(line[1])

    #     velo_filename = os.path.join(
    #         self.data_path,
    #         scene_name,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    #     return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color


class NYUrawDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYUrawDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext) # TODO
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path
    def check_depth(self):
        return False

    # def get_depth(self, folder, frame_index, side, do_flip):
    #     calib_path = os.path.join(self.data_path, folder.split("/")[0])

    #     velo_filename = os.path.join(
    #         self.data_path,
    #         folder,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    #     depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
    #     depth_gt = skimage.transform.resize(
    #         depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

    #     if do_flip: # TODO
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt



