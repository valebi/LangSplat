import glob
import json
import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
import cv2




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()

    json_path = args.json_path
    dataset_path = args.dataset_path
    img_folder = os.path.join(dataset_path, "color")

    image_files = glob.glob(os.path.join(img_folder, "*.jpg"))
    image_files.sort()

    print(f"Found {len(image_files)} images in {img_folder}")

    # clean up and check which images are available
    json_list = []
    for image_path in image_files:
        # image = cv2.imread(image_path)
        # if image is None:
        #     print(f"Error loading image {image_path}")
        #     os.remove(image_path)
        #     os.remove(image_path.replace("color", "depth").replace(".jpg", ".npy"))
        #     os.remove(image_path.replace("color", "pose").replace(".jpg", ".txt"))
        # else:
        json_list.append(image_path.replace(img_folder, "").replace(".jpg", ".json"))

    n_lines_per_response = None

    import re
    data_numpy = []
    data_dict = {}
    n_failed = 0
    for json_file in json_list:
        response = None
        try:
            with open(json_path + json_file) as f:
                response = json.load(f)
            ranking = response["choices"][0]["message"]["content"].split("\n")
            ranking = [r for r in ranking if r.strip() != ""]
            if n_lines_per_response is None:
                n_lines_per_response = len(ranking)
                print(f"Expecting {n_lines_per_response} lines in response")
            elif len(ranking) != n_lines_per_response:
                print(ranking)
                raise ValueError(f"Expected {n_lines_per_response} lines in response, but got {len(ranking)}")
            
            
            result_dict = {}
            result_list = []
            for text in ranking:
                match = re.match(r'^\s*[-]?\s*([a-zA-Z\s]+):\s*(\d+)', text)
                if match:
                    description = match.group(1).strip()  # captures the text part before the number
                    number = float(match.group(2))          # captures the number
                    result_dict[description] = number
                    result_list.append(number)
                else:
                    raise ValueError(f"Could not parse description and number from text: {text}")
            data_numpy.append(result_list)
            data_dict[json_file] = result_dict
        except (FileNotFoundError, ValueError, KeyError) as e:
            n_failed += 1
            # print(json_path + json_file)
            # print(response)
            data_numpy.append([np.NaN] * n_lines_per_response)
            data_dict[json_file] = {}
        

    print(f"Failed to parse {n_failed} json files")
    # print(data_dict)
    # print(data_numpy)
    data_numpy = np.array(data_numpy).astype(np.float16)
    print("Keys:", data_dict[list(data_dict.keys())[0]].keys())
    save_folder = os.path.join(dataset_path, 'full_image_embeddings_gpt')
    print(f"Saving embeddings to {save_folder}")
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "embeddings.npy"), data_numpy)
    with open(os.path.join(save_folder, "embeddings.json"), "w") as f:
        json.dump(data_dict, f)



