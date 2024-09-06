import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from PIL import Image

import torch
import torchvision
from torch import nn
from preprocess import seed_everything, SigLipNetwork, OpenCLIPNetwork, OpenCLIPNetworkConfig

import pickle


try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"





if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model', type=str, default="clip")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    model_name = args.model
    print("[ INFO ] Using model:", model_name)   

    img_folder = os.path.join(dataset_path, 'images')
    if not os.path.isdir(img_folder):
        img_folder = os.path.join(dataset_path, 'color')
    data_list = [f for f in sorted(os.listdir(img_folder))]
    data_list.sort()

    if model_name == "siglip":
        model = SigLipNetwork(device="cuda")
    elif model_name == "clip":
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        model.eval()
    else:
        assert False, "Unknown model"

    img_list = []
    for data_path in tqdm(data_list, desc="Loading images"):
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            os.remove(image_path)
            os.remove(image_path.replace("color", "depth").replace(".jpg", ".npy"))
            os.remove(image_path.replace("color", "pose").replace(".jpg", ".txt"))
            continue
        if model == "clip":
            image = cv2.resize(image, (224, 224))
        img_list.append(image)
       


    embeddings = []
    bsize=4
    for batch_i in tqdm(range(0, len(img_list), bsize), desc="Generating embeddigs"):
        try:
            batch = np.stack(img_list[batch_i:batch_i+bsize], axis=0)
            batch = torch.from_numpy(batch.astype("float16")).permute(0,3,1,2) / 255.0
            batch = batch.cuda()
            with torch.no_grad():
                embeddings.append(model.encode_image(batch).detach().cpu().numpy())
        except Exception as e:
            print(f"Error in batch {batch_i}:", e)
            for i in range(bsize):
                print(img_list[batch_i+i])
                print(img_list[batch_i+i].shape)
            #     os.remove(os.path.join(img_folder, data_list[batch_i+i]))
            #     os.remove(os.path.join(img_folder, data_list[batch_i+i].replace("color", "depth").replace("jpg", "npy")))
            #     os.remove(os.path.join(img_folder, data_list[batch_i+i].replace("color", "pose").replace("jpg", "txt")))
            # embeddings.append(np.zeros((min(len(batch), len(img_list)-batch_i-1), embeddings[-1].shape[-1]), dtype=embeddings[-1].dtype))

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

    assert embeddings.shape[0] == len(img_list), f"Embeddings shape {embeddings.shape} does not match number of images {len(img_list)}"

    
    if model_name == "siglip":
        save_folder = os.path.join(dataset_path, 'full_image_embeddings_siglip')
    else:
        save_folder = os.path.join('/mnt/usb_ssd/opencity-data/results/', 'denhaag-clip-bbox/full_image_embeddings')
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "embeddings.npy"), embeddings)
    # for i, e in tqdm(enumerate(embeddings), desc="Saving embeddings"):
    #     np.save(os.path.join(save_folder, f"{i}.npy"), e)
