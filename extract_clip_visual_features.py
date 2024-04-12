import argparse
from tqdm import trange
import os
from PIL import Image
from transformers import CLIPVisionModelWithProjection, AutoProcessor
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args() 
    dataset = args.dataset

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_folder_list = os.listdir(dataset)
    feature_dict = {}
    for i in trange(len(image_folder_list)):
        image_folder = image_folder_list[i]
        image_list = os.listdir(os.path.join(dataset, image_folder))
        img_list = []
        for image in image_list:
            image_path = os.path.join(dataset, image_folder, image)
            img = Image.open(image_path)
            img_list.append(img)
        
        inputs = processor(images=img_list, return_tensors='pt')
        for key in inputs: inputs[key] = inputs[key].cuda()
        outputs = model(**inputs)
        pooler_output = outputs.image_embeds.detach().cpu()
        feature_dict[image_folder] = pooler_output

    torch.save(feature_dict, "{}_clip_visual_features.pt".format(dataset))

if __name__ == '__main__':
    main()