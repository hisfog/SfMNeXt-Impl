from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

# from kitti_utils import generate_depth_map
from .mono_dataset_mc import MonoDatasetMultiCam

def read_file(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    return lines

class MCDataset(MonoDatasetMultiCam):
    """Superclass for different types of MC dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(MCDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.full_res_shape = (640, 360)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_intrinsics(self, folder):
        return self.KV_intrinsics_dict[folder]

    def get_intrinsics_map(self, file_name):
        lines = read_file(file_name)
        KV_intrinsics_dict = {}
        for line in lines:
            line = line.strip('\n')
            line = line.split()
            # print(line)
            folder_key = str(line[0])
            fx = float(line[1])
            px = float(line[3])
            fy = float(line[2])
            py = float(line[4])
            KV_intrinsics_dict[folder_key] = np.array([[fx, 0, px, 0], 
                                                       [0, fy, py, 0], 
                                                       [0, 0, 1, 0]], dtype=np.float32)
        return KV_intrinsics_dict
