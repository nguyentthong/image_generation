import argparse
import os
import json
from tqdm import trange
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda")

    query_file_path = "{}_query_list.jsonl".format(args.dataset)
    os.makedirs("{}".format(args.dataset), exist_ok=True)
    with open(query_file_path, 'r') as f: lines = f.readlines()
    query_list = [json.loads(line) for line in lines]
    processed_query_id_set = set(os.listdir("{}".format(args.dataset)))
    num_images = 6

    with open("{}_error_list.txt".format(args.dataset), 'a') as f:
        for i in trange(len(query_list)):
            query_id = query_list[i]['query_id']
            query = query_list[i]['query']
            output_dir = os.path.join("{}".format(args.dataset), query_id)
            if query_id in processed_query_id_set and len(os.listdir(output_dir)) == num_images: continue
            prompt = [query] * num_images
            
            count = 0
            valid = 0
            while True:
                results = pipe(prompt)
                if True in results['nsfw_content_detected']:
                    if count == 0:
                        f.write(query_id + '\n')
                        f.flush()
                    count += 1
                elif True not in results['nsfw_content_detected']: 
                    valid = 1
                    break
                if count == 5: break

            if valid:
                os.makedirs(output_dir, exist_ok=True)
                images = results.images
                for j in range(num_images):
                    images[j].save(os.path.join(output_dir, "image_{}.png".format(j)))


if __name__ == '__main__':
    main()