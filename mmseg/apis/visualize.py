import os.path as osp
import tempfile
import math

import mmcv
import os
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.image import tensor2imgs
from mmseg.utils import FlopCountAnalysis
from mmseg.models import build_segmentor, build_backbone, build_head
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def visualize_attn(model, data_loader, args, cfg):

    model.eval()
    out_dir = args.show_dir
    dataset = data_loader.dataset

    for j, data in enumerate(data_loader):

        if j < args.data_range[0]:
            continue
        if j > args.data_range[1]:
            break

        cidx = j + 1
        with torch.no_grad():
            result = model(return_loss=False, **data)

        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            # Save segmentation results
            if isinstance(result, list):
                for i in range(len(result)):
                    out = [result[i]]
                    out_file = osp.join(out_dir, 'segmentation', img_meta['ori_filename'][:-4] + "i" +
                                        str(i) + str(cidx) + '.png')
                    model.module.show_result(
                        img_show,
                        out,
                        palette=dataset.PALETTE,
                        out_file=out_file)
            else:
                out_file = osp.join(out_dir, 'segmentation', img_meta['ori_filename'][:-4] + "i" + str(cidx) + '.png')
                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    out_file=out_file)

            # Save ground truth
            if args.save_gt_seg:
                train_suf = 'leftImg8bit' if cfg.dataset_type == 'CityscapesDataset' else '.jpg'
                gt_suf = 'gtFine_labelTrainIds' if cfg.dataset_type == 'CityscapesDataset' else '.png'
                gt_file = os.path.join(cfg.data_root, cfg.data.test.ann_dir, img_meta['ori_filename'])\
                    .replace(train_suf, gt_suf)

                gt_seg_map = mmcv.imread(gt_file, flag='unchanged', backend='pillow')
                if cfg.dataset_type != 'CityscapesDataset':
                    gt_seg_map = np.clip(gt_seg_map - 1, 0, 300)
                out_file = osp.join(out_dir, 'gt_seg', img_meta['ori_filename'][:-4] + "i" + str(cidx) + '.png')

                model.module.show_result(
                    img_show,
                    [gt_seg_map],
                    palette=dataset.PALETTE,
                    out_file=out_file)

            # Save original
            if args.save_ori_img:
                out_file = osp.join(out_dir, "ori_img", img_meta['ori_filename'][:-4] + "i" + str(cidx) + ".png")
                mmcv.imwrite(img, out_file)

            # Save hierarchical attention
            try:
                topdown_attn12 = model.module.backbone.attn['12']['top']
                topdown_attn23 = model.module.backbone.attn['23']['top']
                topdown_attn34 = model.module.backbone.attn['34']['top']
            except:
                print("No Attention Found")
                continue

            out_file_root = osp.join(out_dir, "topdown_attn", img_meta['ori_filename'][:-4] + "i" + str(cidx))
            show_hierarchal_attn(cfg, img, topdown_attn12, topdown_attn23, topdown_attn34, out_file_root=out_file_root,
                                 x4=args.attn_coors[0], y4=args.attn_coors[1])
            print("Finished Visualizing Image:", cidx)

def show_hierarchal_attn(cfg, img, topdown_attn12, topdown_attn23, topdown_attn34, out_file_root, x4=13, y4=12):
    def get_attn(img, attns):
        attn = torch.mean(attns, dim=1, keepdim=True)
        if attn.shape[-1] == attn.shape[-2]:
            B, _, K, K = attn.shape
        else:
            B, _, K2, _ = attn.shape
            K = int(math.sqrt(K2))

        H, W, C = img.shape
        downsamp_res = int((H * W / attn.shape[0]) ** 0.5)
        num_patch_x, num_patch_y = int(H / downsamp_res), int(W / downsamp_res)

        attn = attn.reshape(B, 1, K, K)
        return attn.reshape(num_patch_x, num_patch_y, K, K), K, downsamp_res

    def get_attn_map(img, attns, x_coors, y_coors):

        attn, K, downsamp_res = get_attn(img, attns)

        K_px = K * downsamp_res // 2
        pad = K_px // 2
        attn = F.interpolate(attn, size=(K_px, K_px), mode='nearest')

        seg = attn.unsqueeze(-1).expand(-1, -1, -1, -1, 3).detach().cpu().numpy()
        color_seg = np.zeros((img.shape[0] + pad, img.shape[1] + pad, 3), dtype=np.float32)
        for x in x_coors:
            for y in y_coors:
                mask = seg[x, y, :, :]
                x_px, y_px = x * downsamp_res, y * downsamp_res
                #TODO: fix problem where formula differs for patch of 2 or 6
                # x_px, y_px = (x-1) * downsamp_res, (y-1) * downsamp_res
                color_seg[x_px:x_px + K_px, y_px:y_px + K_px] += mask
        color_seg = color_seg[..., ::-1]
        color_seg = color_seg[int(math.ceil(pad / 2)):-int(math.floor(pad / 2)),
                              int(math.ceil(pad / 2)):-int(math.floor(pad / 2))]
        return color_seg

    def normalize_and_color_and_output(color_seg, K, downsamp_res, output_suf):
        color_seg = color_seg / np.max(color_seg)

        import matplotlib
        cmap = matplotlib.cm.get_cmap('jet')
        rgb = cmap(1 - color_seg[:, :, 0])
        color_seg = rgb[:, :, 0:3] * 255

        K_px = K * downsamp_res // 2
        x_px, y_px = x4 * downsamp_res - K_px // 4, y4 * downsamp_res - K_px // 4
        color_seg[x_px:x_px + 2, y_px:y_px + K_px] = (0, 0, 255)
        color_seg[x_px + K_px:x_px + K_px + 2, y_px:y_px + K_px] = (0, 0, 255)
        color_seg[x_px:x_px + K_px + 2, y_px:y_px + 2] = (0, 0, 255)
        color_seg[x_px:x_px + K_px + 2, y_px + K_px:y_px + K_px + 2] = (0, 0, 255)

        out_img = img + color_seg * 0.5
        out_img = np.clip(out_img, 0, 255)
        out_img = out_img.astype(np.uint8)
        out_file = out_file_root[:-5] + output_suf
        mmcv.imwrite(out_img, out_file)

    # Read attention from the last inference - for sliding inference, this is the last sliding window
    base_img = mmcv.imread(img)
    if cfg.model.test_cfg.mode == 'slide':
        slide_window = cfg.model.test_cfg.crop_size[0]
        img = img[:, -1*slide_window:]
        base_img = base_img[:, -1*slide_window:]

    raw_attn12, _, _ = get_attn(img, topdown_attn12[-1])
    raw_attn23, _, _ = get_attn(img, topdown_attn23[-1])
    raw_attn34, K, downsamp_res = get_attn(img, topdown_attn34[-1])

    # Show attention at Stage 3 -> 4
    color_seg = get_attn_map(img, topdown_attn34[-1], [x4], [y4])
    normalize_and_color_and_output(color_seg, K, downsamp_res, '_hierarchical_attn_stage3to4.png')

    # Show attention at Stage 2 -> 4 and 1 -> 4
    x3, y3 = x4 * 2 - 1, y4 * 2 - 1
    attn23 = raw_attn23[x3:x3 + K, y3:y3 + K]
    attn34 = raw_attn34[x4, y4].unsqueeze(2).unsqueeze(3)
    raw_weighted_attn23 = attn23 * attn34

    for stage_1_to_4, output_suf in zip([False, True], ['_hierarchical_attn_stage2to4.png',
                                                        '_hierarchical_attn_stage1to4.png']):
        final_color_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        for _x3 in [x4 * 2 - 1 + i for i in range(K)]:
            for _y3 in [y4 * 2 - 1 + i for i in range(K)]:

                x2, y2 = _x3 * 2 - 1, _y3 * 2 - 1
                attn12 = raw_attn12[x2:x2 + K, y2:y2 + K]
                if not stage_1_to_4:
                    attn12 = torch.ones(attn12.shape, device=attn12.device) # Identity
                weighted_attn23 = raw_weighted_attn23[_x3 - x3, _y3 - y3].unsqueeze(2).unsqueeze(3)

                weighted_patch2 = attn12 * weighted_attn23
                raw_attn12_tmp = raw_attn12.clone()
                raw_attn12_tmp[x2:x2 + K, y2:y2 + K] = weighted_patch2
                raw_attn12_tmp = raw_attn12_tmp.reshape(-1, 1, raw_attn12_tmp.shape[2], raw_attn12_tmp.shape[3])
                color_seg = get_attn_map(img, raw_attn12_tmp,
                                         x_coors=[_x3 * 2 - 1 + i for i in range(4)],
                                         y_coors=[_y3 * 2 - 1 + i for i in range(4)])
                final_color_seg += color_seg
        normalize_and_color_and_output(final_color_seg, K, downsamp_res, output_suf)
