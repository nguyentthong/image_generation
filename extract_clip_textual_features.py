import argparse
from tqdm import trange
import os
from PIL import Image
from transformers import CLIPTextModel, AutoTokenizer
import torch
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args() 
    dataset = args.dataset

    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    query_list = []
    with open("{}_query_list.jsonl".format(dataset), 'r') as f: lines = f.readlines()
    for line in lines:
        query_list.append(json.loads(line))

    feature_dict = {}
    for i in trange(len(query_list)):
        query = query_list[i]['query']
        query_id = query_list[i]['query_id']
        
        inputs = tokenizer([query], return_tensors='pt')
        for key in inputs: inputs[key] = inputs[key].cuda()
        outputs = model(**inputs)
        pooler_output = outputs.pooler_output.detach().cpu()
        feature_dict[query_id] = pooler_output

    torch.save(feature_dict, "{}_clip_textual_features.pt".format(dataset))

if __name__ == '__main__':
    main()