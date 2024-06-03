#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork, SigLipNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import matplotlib.pyplot as plt

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        
        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
        
        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num


def grayscale_to_plasma(image):
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray) and len(image.shape) == 2:
        image = np.array(image)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = np.array(image)
        colormap = plt.get_cmap('plasma')
        
        return colormap(image)[:,:,:3]
    elif isinstance(image, list):
        min_, max_ = min([i.min() for i in image]), max([i.max() for i in image])
        images = [(i - min_) / (max_ - min_ + 1e-6) for i in image]
        colormap = plt.get_cmap('plasma')

        return [colormap(i)[:,:,:3] for i in images]
    else:
        raise ValueError("image should be either a list of np.array or a np.array")



def compute_sim(clip_model, features, segmentation, prompt, normalize=False):
    prompt_embed = clip_model.encode_text([prompt]).detach().cpu().numpy()[0].T
    
    canonical_queries = ["object", "things", "stuff"]
    normalization_embeddings = np.concatenate([clip_model.encode_text([q]).detach().cpu().numpy().T for q in canonical_queries], axis=1)
    # print(f'"{prompt}": {json.dumps(list([list(p) for p in prompt_embed.astype(float)]))}')

    sim = features @ prompt_embed
    if normalize:
        canonical_sim = np.max(features @ normalization_embeddings, axis=-1, keepdims=False)
        sim = np.exp(sim) / (np.exp(sim) + np.exp(canonical_sim))

    sim = sim.flatten()
    sim_img = np.zeros_like(segmentation, dtype=float)[0]
    for i in range(len(features)):
        for j in range(len(segmentation)):
            sim_img[(segmentation[j] == i) & (sim_img < sim[i])] = sim[i]
    # save 
    if (sim_img == 0).sum() > 0 and (sim_img != 0).sum() > 0:
        sim_img[sim_img == 0] = np.min(sim_img[sim_img != 0])
    return sim_img

def evaluate(path, prompt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate autoencoder and openclip

    feature_type = path.split("/")[-2]

    features = np.load(path + "_f.npy")
    segmentation = np.load(path + "_s.npy")

    if features.shape[-1] == 512:
        clip_model = OpenCLIPNetwork(device)
    else:
        clip_model = SigLipNetwork(device)
    
    sim_img = compute_sim(clip_model, features, segmentation, prompt)
    sim_img = grayscale_to_plasma(sim_img)
    new_p = Image.fromarray((sim_img *255).astype(np.uint8))
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    os.makedirs(f"../eval_result/evaluate_text_query/{feature_type}/", exist_ok=True)
    new_p.save(f"../eval_result/evaluate_text_query/{feature_type}/query_result_{prompt}.png")


def evaluate_comparatively(img_dir, prompt, normalize=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate autoencoder and openclip
    import glob
    feature_type = img_dir.split("/")[-1]
    images = sorted(glob.glob(img_dir.replace(feature_type, "images") + "/*.*"))
    features = [np.load(f) for f in sorted(glob.glob(img_dir + "/*_f.npy"))]
    segmentation = [np.load(f) for f in sorted(glob.glob(img_dir + "/*_s.npy"))]
    if features[0].shape[-1] == 512:
        clip_model = OpenCLIPNetwork(device)
    else:
        clip_model = SigLipNetwork(device)
    
    sim_img = [compute_sim(clip_model, features[i], segmentation[i], prompt, normalize=normalize) for i in range(len(features))]
    sim_img = grayscale_to_plasma(sim_img)

    width = len(sim_img) // 2
    fig, axs = plt.subplots(2, width, figsize=(5*width, 5*2))
    for i in range(width):
        for j in range(2):
            axs[j,i].set_title(f"{images[i*2 + j].split('/')[-1].split('.')[0]}")
            im = axs[j,i].imshow(sim_img[i*2 + j])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.subplots_adjust(right=0.8)
    if normalize:
        plt.title(f"Exp-Normalized similarity to prompt: \n '{prompt}'")
    else:
        plt.title(f"Dot product with: \n '{prompt}'")
    os.makedirs(f"../eval_result/evaluate_text_query/{feature_type}/", exist_ok=True)
    plt.savefig(f"../eval_result/evaluate_text_query/{feature_type}/query_comparison_{prompt}.png")
    #new_p = Image.fromarray((sim_img *255).astype(np.uint8))
    
    # save raw images and sim too under /raw_images /raw_sim
    os.makedirs(f"../eval_result/evaluate_text_query/{feature_type}/raw_images/", exist_ok=True)
    os.makedirs(f"../eval_result/evaluate_text_query/{feature_type}/raw_sim_to_{prompt}/", exist_ok=True)
    for i in range(len(images)):
        img = Image.open(images[i])
        img.save(f"../eval_result/evaluate_text_query/{feature_type}/raw_images/{images[i].split('/')[-1]}")
        new_p = Image.fromarray((sim_img[i] *255).astype(np.uint8))
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(f"../eval_result/evaluate_text_query/{feature_type}/raw_sim_to_{prompt}/{images[i].split('/')[-1]}")

if __name__ == "__main__":
    # path = "/home/bieriv/LangSplat/LangSplat/data/brooklyn-bridge-colmap/language_features/246"
    feature_type = "language_features"
    # path = f"/home/bieriv/LangSplat/LangSplat/data/buenos-aires-samples/{feature_type}/bad_neigh"
    
    # prompt = "building"
    # evaluate(path, prompt)
    # prompt = "skyscraper"
    path = f"/home/bieriv/LangSplat/LangSplat/data/rotterdam-samples/{feature_type}"
    prompts = ["building", "trees or vegetation", "canal, lake or the sea", "skyscraper", "bridge", "road" ] #, "densely populated area", "expensive neighborhood", "touristic neighborhood", "dangerous neighborhood", "industrial area", "old town"]
    for prompt in prompts:
        evaluate_comparatively(path, prompt)
    # if "highlight" in feature_type:
    #     prompts = [f"an urban scene (highlighted: a {h})" for h in prompts]
    #     for prompt in prompts:
    #         evaluate_comparatively(path, prompt, normalize=True)
    # seed_num = 42
    # seed_everything(seed_num)
    
    # parser = ArgumentParser(description="prompt any label")
    # parser.add_argument("--dataset_name", type=str, default=None)
    # parser.add_argument('--feat_dir', type=str, default=None)
    # parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    # parser.add_argument("--output_dir", type=str, default=None)
    # parser.add_argument("--json_folder", type=str, default=None)
    # parser.add_argument("--mask_thresh", type=float, default=0.4)
    # parser.add_argument('--encoder_dims',
    #                     nargs = '+',
    #                     type=int,
    #                     default=[256, 128, 64, 32, 3],
    #                     )
    # parser.add_argument('--decoder_dims',
    #                     nargs = '+',
    #                     type=int,
    #                     default=[16, 32, 64, 128, 256, 256, 512],
    #                     )
    # args = parser.parse_args()

    # # NOTE config setting
    # dataset_name = args.dataset_name
    # mask_thresh = args.mask_thresh
    # feat_dir = ["/home/bieriv/LangSplat/LangSplat/data/lerf_ovs/teatime/language_features"] #[os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    # output_path = os.path.join(args.output_dir, dataset_name)
    # ae_ckpt_path = "/home/bieriv/LangSplat/LangSplat/autoencoder/ckpt/ae_ckpt/best_ckpt.pth"
    # # ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "ae_ckpt/best_ckpt.pth")
    # json_folder = os.path.join(args.json_folder, dataset_name)

    # # NOTE logger
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # os.makedirs(output_path, exist_ok=True)
    # log_file = os.path.join(output_path, f'{timestamp}.log')
    # logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    # evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, args.encoder_dims, args.decoder_dims, logger)