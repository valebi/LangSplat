            # from transformers import AutoModel

            # clip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")

from transformers import AutoProcessor, AutoModel
from PIL import Image
import glob
import numpy as np
import torch

if __name__ == "__main__":
    model = "google/siglip-base-patch16-224"
    images = glob.glob("/home/bieriv/LangSplat/LangSplat/data/buenos-aires-samples/images/*.png")
    images = [Image.open(img).resize((224, 224)).convert('RGB') for img in images]
   # images = [torch.from_numpy(np.asarray(img).transpose(2,0,1)).cuda().float() for img in images]

    processor = AutoProcessor.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    get_emb = model.get_text_features

   # print(images[0].shape)

    cropped_img_processed = processor(images=images, return_tensors="pt")
    # cropped_img_processed = next(iter(cropped_img_processed.values()))[0]# there has to be a better way
    print(model.layers)
    model.cuda()
    with torch.no_grad():
        features = model.get_image_features(cropped_img_processed["pixel_values"].cuda())
        print(features.shape)
