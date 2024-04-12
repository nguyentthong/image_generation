import argparse
import os
import json
from tqdm import trange
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    dataset = args.dataset
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda")

    query_file_path = "{}_query_list.jsonl".format(dataset)
    os.makedirs("{}".format(dataset), exist_ok=True)
    with open(query_file_path, 'r') as f: lines = f.readlines()
    query_list = [json.loads(line) for line in lines]
    processed_query_id_set = set(os.listdir("{}".format(dataset)))
    num_images = 6
    with open("{}_error_list.txt".format(dataset), 'r') as f: query_id_list = f.readlines()
    query_id_list = [query_id.strip() for query_id in query_id_list]
    query_id_set = set(query_id_list)

    with open("{}_more_error_list.txt".format(dataset), 'a') as f:
        for i in trange(len(query_list)):
            query_id = query_list[i]['query_id']
            query = query_list[i]['query']
            output_dir = os.path.join("{}".format(dataset), query_id)
            if query_id not in query_id_set: continue
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
                if count == 20: break

            if valid:
                os.makedirs(output_dir, exist_ok=True)
                images = results.images
                for j in range(num_images):
                    images[j].save(os.path.join(output_dir, "image_{}.png".format(j)))


if __name__ == '__main__':
    main()