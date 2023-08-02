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
import cv2
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



def hsic(x, y):
    # Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    # Kx = np.exp(- Kx**2) # 计算核矩阵
    #  
    # Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    # Ky = np.exp(- Ky**2) # 计算核矩阵

    Kx = np.outer(x, x)
    Ky = np.outer(y, y)
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2

# 计算两个一维向量之前的CKA相似度
def CKA_impl(vec1, vec2):
    return hsic(vec1, vec2) / np.sqrt(hsic(vec1, vec1) * hsic(vec2, vec2))

# 计算两个二维向量每个位置之间的相似度，并返回为一个二维的相似度矩阵（heatmap)
def CKA_vis(vec1, vec2):
  # 如果两个数组的形状不同，抛出异常
  if vec1.shape != vec2.shape:
    raise ValueError("Cannot add arrays with different shapes.")

  # 创建一个数组 c，用来存储结果
  c = np.zeros((vec1.shape[0], vec1.shape[0]))

  # 遍历每个位置，并计算相似度
  for i, v1 in enumerate(vec1):
      for j, v2 in enumerate(vec2):
        c[i][j] = CKA_impl(v1, v2)

  # 返回计算结果
  return c

def CKA_main(opt):
    # image = Image.open('./assets/noise.jpg')
    # image = Image.open('./assets/RGB/0000000182.png')
    # image = Image.open('./assets/con1.png')
    image = Image.open('./assets/nwpu/1668829037004.jpg')
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
    raw_map = attn_maps[0][0]
    # print(type(attn_maps[0][0]))
    raw_map = raw_map.transpose(1, 2, 0) # visualize all features within resolution of 20x64
    new_shape = (20, 64)
    resized_map = cv2.resize(raw_map, new_shape)
    # new_shape = (32, 10, 32) # visualize left-top local features
    # resized_map = np.resize(raw_map, new_shape)
    H, W, C = resized_map.shape
    print(resized_map.shape)
    feat_vec = resized_map.reshape(H*W, C)
    # feat_vec = feat_vec.transpose(1, 0)
    print(feat_vec.shape)
    CKA_heatmap = CKA_vis(feat_vec, feat_vec)
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    fig.tight_layout()
    ax.imshow(CKA_heatmap)
    name = "1668829037004_x0"
    # plt.colorbar()
    plt.savefig('CKA_vis/CKA_{}_{}.png'.format(name ,new_shape), bbox_inches='tight', pad_inches=0)



    
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
    CKA_main(opt)

