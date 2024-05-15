
from openclip_encoder import OpenCLIPNetwork
import torch 
import numpy as np
import json
import os

def encode_queries(queries, out_file, model="clip"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = OpenCLIPNetwork(device)
    
    prompt_embed = clip_model.encode_text(queries, "cuda").detach().cpu().numpy()
    embed_dict = {q: list(map(list, prompt_embed[i].astype(float).reshape(-1,1))) for i, q in enumerate(queries)}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(embed_dict, f)
    

if __name__ == "__main__":
    encode_queries(["building", "trees, bushes or vegetation", "sports field", "bridge", "canal, lake or the sea", "road", "skyscraper"], "../eval_result/clip_text_embeddings/prompts.json", model="clip")
    encode_queries(["densely populated area", "sparsely populated area", "district with high crime rate", "district with low crime rate", "dangerous neighborhood", "low-income district", "high-income district",  "touristic neighborhood", "industrial area", "old town","expensive neighborhood", ], "../eval_result/clip_text_embeddings/social_dynamics_prompts.json", model="clip")
    encode_queries(["object", "things", "stuff", "texture"], "../eval_result/clip_text_embeddings/canonical.json", model="clip")
    encode_queries(["aerial image", "satellite image"], "../eval_result/clip_text_embeddings/social_dynamics_canonical.json", model="clip")