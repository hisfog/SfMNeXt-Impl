from __future__ import absolute_import, division, print_function
from visualizer import get_local
get_local.activate()

import torch
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import networks
import os
import sys
from options import MonodepthOptions

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    """
    grid_size=14是因为patch_size=16, 所以有224/16=14个patch
    grid_index代表的是第几个grid,这个grid的attn_map是14x14大小的，代表的是对于每个patch的相似度，
    然后resize成224x224大小即可得到对原图每个位置的相似度
    然后作为透明度的掩码即可叠加到原图上。
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    _, H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], (8, 25))
    # grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    fig.tight_layout()
    
    # ax[0].imshow(grid_image)
    # ax[0].axis('off')
    
    ax.imshow(grid_image)
    ax.imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax.axis('off')
    plt.savefig('attn_vis/attn_{}.png'.format(grid_index), bbox_inches='tight', pad_inches=0)
    # plt.show()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =4)
    return image

def attn_vis(opt):
    image = Image.open('./assets/con1.png')
    transforms = T.Compose([
                           T.Resize((320, 1024)),
                           T.ToTensor(),
                                   ])

    input_tensor = transforms(image).unsqueeze(0)
    input_tensor = input_tensor.cuda()
    print(input_tensor.shape)

    get_local.clear()
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    encoder = networks.BaseEncoder.build(model_dim=opt.model_dim)
    # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
    depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                            query_nums=opt.query_nums, num_heads=4,
                                                            min_val=0.001, max_val=10.0)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    # encoder = torch.nn.DataParallel(encoder)
    encoder.eval()
    depth_decoder.cuda()
    # depth_decoder = torch.nn.DataParallel(depth_decoder)
    depth_decoder.eval()
    with torch.no_grad():
        output = depth_decoder(encoder(input_tensor))
    cache = get_local.cache
    print(list(cache.keys()))
    attn_maps = cache['Depth_Decoder_QueryTr.forward']
    # print(len(attn_maps))
    print(attn_maps[0][0].shape)
    for index in range(128):
        visualize_grid_to_grid(attn_maps[0][0], index, image, grid_size=(160, 512))
    # res = attn_maps[0][0]
    # ax = plt.gca()
    # im = ax.imshow(res)
    # cbar = ax.figure.colorbar(im, ax=ax)
    # plt.savefig('2.png')
    # visualize_grid_to_grid(attn_maps[3][0,0,1:,1:], 100, image)


    
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
    attn_vis(opt)
