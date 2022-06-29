import os
import sys

code_path = os.getcwd()
sys.path.append(code_path)

import argparse
from mmseg.apis import visualize_attn
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmseg.models import build_segmentor, build_backbone, build_head
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args

def sra_flops(h, w, r, dim, num_heads, n2=None):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r if n2 is None else n2

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads
    return f1 + f2

def swa_flops(h, w, r, dim, num_heads, n2=None):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = r * r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads
    return f1 + f2

def get_swin_tr_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    net = net.backbone

    stage1 = swa_flops(H // 4, W // 4,
                       net.window_size,
                       int(net.embed_dim * 2 ** 0),
                       net.num_heads[0]) * net.depths[0]
    stage2 = swa_flops(H // 8, W // 8,
                       net.window_size,
                       int(net.embed_dim * 2 ** 1),
                       net.num_heads[1]) * net.depths[1]
    stage3 = swa_flops(H // 16, W // 16,
                       net.window_size,
                       int(net.embed_dim * 2 ** 2),
                       net.num_heads[2]) * net.depths[2]
    stage4 = swa_flops(H // 32, W // 32,
                       net.window_size,
                       int(net.embed_dim * 2 ** 3),
                       net.num_heads[3]) * net.depths[3]

    flops += stage1 + stage2 + stage3 + stage4

    if 1 in net.hila_attn:
            topdown12 = sra_flops(H // 8, W // 8,
                                  1,
                                  int(net.embed_dim * 2 ** 0),
                                  1,
                                  n2=16)*(net.depths[1]-1)
            bottomup12 = sra_flops(H // 8, W // 8,
                                   1,
                                   int(net.embed_dim * 2 ** 0),
                                   1,
                                   n2=16)*(net.depths[1])
            bottomsa12 = sra_flops(H // 4, W // 4,
                                   net.window_size,
                                   int(net.embed_dim * 2 ** 0),
                                   net.num_heads[0]) * (net.depths[1]-1)
            flops += topdown12 + bottomup12 + bottomsa12

    if 2 in net.hila_attn:
        topdown23 = sra_flops(H // 16, W // 16,
                              1,
                              int(net.embed_dim * 2 ** 1),
                              1,
                              n2=16)*(net.depths[2]-1)
        bottomup23 = sra_flops(H // 16, W // 16,
                               1,
                               int(net.embed_dim * 2 ** 1),
                               1,
                               n2=16)*(net.depths[2])
        bottomsa23 = swa_flops(H // 8, W // 8,
                               net.window_size,
                               int(net.embed_dim * 2 ** 1),
                               net.num_heads[1]) * (net.depths[2]-1)
        flops += topdown23 + bottomup23 + bottomsa23

    if 3 in net.hila_attn:
            topdown34 = sra_flops(H // 32, W // 32,
                                  1,
                                  int(net.embed_dim * 2 ** 2),
                                  1,
                                  n2=16)*(net.depths[3]-1)
            bottomup34 = sra_flops(H // 32, W // 32,
                                   1,
                                   int(net.embed_dim * 2 ** 2),
                                   1,
                                   n2=16)*(net.depths[3])
            bottomsa34 = sra_flops(H // 16, W // 16,
                                   net.window_size,
                                   int(net.embed_dim * 2 ** 2),
                                   net.num_heads[2]) * (net.depths[3]-1)
            flops += topdown34 + bottomup34 + bottomsa34

    return flops_to_string(flops), params_to_string(params)

def get_tr_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    net = net.backbone

    try:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.sr_ratio,
                           net.block1[0].attn.dim,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.sr_ratio,
                           net.block2[0].attn.dim,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.sr_ratio,
                           net.block3[0].attn.dim,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.sr_ratio,
                           net.block4[0].attn.dim,
                           net.block4[0].attn.num_heads) * len(net.block4)
    except:
        stage1 = sra_flops(H // 4, W // 4,
                           32,
                           net.block1[0].blocks[0].dim,
                           net.block1[0].blocks[0].num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           16,
                           net.block2[0].blocks[0].dim,
                           net.block2[0].blocks[0].num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           8,
                           net.block3[0].blocks[0].dim,
                           net.block3[0].blocks[0].num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           4,
                           net.block4[0].blocks[0].dim,
                           net.block4[0].blocks[0].num_heads) * len(net.block4)
    flops += stage1 + stage2 + stage3 + stage4

    patch_size = 4
    num_patches = patch_size * patch_size

    if hasattr(net, 'topdown'):
        if '12' in net.topdown:
            topdown12 = sra_flops(H // 8, W // 8,
                                  1,
                                  net.block1[0].attn.dim,
                                  1,
                                  n2=num_patches)*(len(net.block2)-1)
            bottomup12 = sra_flops(H // 8, W // 8,
                                   1,
                                   net.block1[0].attn.dim,
                                   1,
                                   n2=num_patches)*(len(net.block2))
            bottomsa12 = sra_flops(H // 4, W // 4,
                                   net.block1[0].attn.sr_ratio,
                                   net.block1[0].attn.dim,
                                   net.block1[0].attn.num_heads) * (len(net.block2)-1)
            flops += topdown12 + bottomup12 + bottomsa12

        if '23' in net.topdown:
            topdown23 = sra_flops(H // 16, W // 16,
                                  1,
                                  net.block2[0].attn.dim,
                                  1,
                                  n2=num_patches)*(len(net.block3)-1)
            bottomup23 = sra_flops(H // 16, W // 16,
                                   1,
                                   net.block2[0].attn.dim,
                                   1,
                                   n2=num_patches)*(len(net.block3))
            bottomsa23 = sra_flops(H // 8, W // 8,
                                   net.block2[0].attn.sr_ratio,
                                   net.block2[0].attn.dim,
                                   net.block2[0].attn.num_heads) * (len(net.block3)-1)
            flops += topdown23 + bottomup23 + bottomsa23

        if '34' in net.topdown:
            topdown34 = sra_flops(H // 32, W // 32,
                                  1,
                                  net.block3[0].attn.dim,
                                  1,
                                  n2=num_patches)*(len(net.block4)-1)
            bottomup34 = sra_flops(H // 32, W // 32,
                                   1,
                                   net.block3[0].attn.dim,
                                   1,
                                   n2=num_patches)*(len(net.block4))
            bottomsa34 = sra_flops(H // 16, W // 16,
                                   net.block3[0].attn.sr_ratio,
                                   net.block3[0].attn.dim,
                                   net.block3[0].attn.num_heads) * (len(net.block4)-1)
            flops += topdown34 + bottomup34 + bottomsa34

    return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    # from IPython import embed; embed()
    if hasattr(model.backbone, 'block1'):
        print('#### get Segformer transformer flops ####')
        with torch.no_grad():
            flops, params = get_tr_flops(model, input_shape)
    elif hasattr(model.backbone, 'layers'):
        print('#### get Swin transformer flops ####')
        with torch.no_grad():
            flops, params = get_swin_tr_flops(model, input_shape)
    else:
        print('#### get CNN flops ####')
        flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
