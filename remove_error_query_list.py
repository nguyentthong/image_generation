import argparse
import os
import shutil
from tqdm import trange

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args() 
    dataset = args.dataset

    with open("{}_more_error_list.txt".format(dataset), 'r') as f:
        lines = f.readlines()
    
    for i in trange(len(lines)):
        folder = lines[i].strip()
        if os.path.exists(os.path.join(dataset, folder)):
            shutil.rmtree(os.path.join(dataset, folder))


if __name__ == '__main__':
    main()