import argparse
from PIL import Image
from tqdm import tqdm
import cv2
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    dataset = args.dataset
    error_list = []
    with open("{}_error_list.txt".format(dataset), 'r') as f: lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        images = os.listdir(os.path.join(dataset, line.strip()))
        for image in images:
            img = cv2.imread(os.path.join(dataset, line.strip(), image), 0)
            if cv2.countNonZero(img) == 0: 
                error_list.append(line.strip())
                break
            
    with open("{}_error_list.txt".format(dataset), 'w') as f:
        for query_id in error_list: 
            f.write(query_id)
            f.write('\n')


if __name__ == '__main__':
    main()