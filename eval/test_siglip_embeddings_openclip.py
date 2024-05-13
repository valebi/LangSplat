import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms # works on open-clip-torch>=2.23.0, timm>=0.9.8
import glob
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image_paths = sorted(glob.glob("/home/bieriv/LangSplat/LangSplat/data/buenos-aires-samples/images/*.png"))
    images = [Image.open(img).resize((224, 224)).convert('RGB') for img in image_paths]
    texts = ["densely populated area", "expensive neighborhood", "touristic neighborhood", "dangerous neighborhood", "industrial area", "old town"]


    siglip = False
    if siglip:
        checkpoint = 'hf-hub:timm/ViT-B-16-SigLIP'
        # checkpoint = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
        model, preprocess = create_model_from_pretrained(checkpoint)
        tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
        texts_tok = tokenizer(texts, context_length=model.context_length)
        images = torch.stack(list(map(preprocess, images)),axis=0) #.unsqueeze(0)
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts_tok)
    else:
        checkpoint = 'laion2b_s34b_b88k'
        model_type = "ViT-B-16"
        model, _, preprocess = create_model_and_transforms(
                "ViT-B-16",
                pretrained=checkpoint,
                precision="fp16",
        ) 
        tokenizer = get_tokenizer(model_type)
        images = torch.stack(list(map(preprocess, images))).cuda().half()
        texts_tok = tokenizer(texts, context_length=model.context_length).cuda()
        model = model.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts_tok)
        


    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    text_probs = image_features @ text_features.T
    # text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)
    text_probs = F.normalize(text_probs, dim=0).T

    # plot the raw images in a nx2 grid for reference
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        plt.subplot(len(image_paths)//2, 2, i+1)
        plt.imshow(img)
        plt.title(path.split('/')[-1].split(".")[0])
    plt.savefig("eval_result/raw_images.png")


    # create heatmap
    fig, ax = plt.subplots()
    cax = ax.matshow(text_probs.cpu().numpy(), cmap='hot')
    # label axes 
    ax.set_yticklabels([""] + texts)
    ax.set_xticklabels([""]+[path.split('/')[-1].split(".")[0] for path in image_paths])
    # angle 
    plt.xticks(rotation=45, rotation_mode="anchor")
    print()
    plt.title(f"norm. dot product {checkpoint}")
    fig.colorbar(cax)
    # make sure labels not cut off left
    plt.tight_layout()

    plt.savefig(f"eval_result/similarity_heatmap_{checkpoint.replace('/', '__')}.png")

    print(text_probs.shape)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)


    

    