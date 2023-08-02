# from __future__ import absolute_import, division, print_function

# from trainer_cityscapes import Trainer
# from trainer_debug_city import Trainer # for cityscapes
from trainer_res50_kitti import Trainer # for kitti and res50
# from trainer_lite import Trainer # for lite version
from options import MonodepthOptions
import sys
import argparse

options = MonodepthOptions()
# opts = options.parse()

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == "__main__":
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opts = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opts = options.parser.parse_args()
    trainer = Trainer(opts)
    trainer.train()
