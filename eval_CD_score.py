import os
import json
import numpy as np
from argparse import ArgumentParser

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from skimage.io import imread
import statistics

def main():
    parser = ArgumentParser()
    parser.add_argument('--pr',type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument("--num_samples", type=int, default=4)

    args = parser.parse_args()

    num_images = 16
    pr_dir = args.pr


    model, preprocess = clip.load("ViT-L/14", device="cuda")

    for bi in range(args.num_samples):
        name = args.name + f'-{bi}'
        img_pr_list = imread(os.path.join(pr_dir, f'{bi}.png'))
        img_pr_list = np.array_split(img_pr_list, num_images, 1)

        input_text_name = args.name.replace('_', ' ') ## replace _ to space
        modified_string = ''.join([i for i in input_text_name if not i.isdigit()])
        print (modified_string)
        ref_text = clip.tokenize([f'An image of {modified_string}']).cuda()
        with torch.no_grad():
            text_features = model.encode_text(ref_text)

        similarity_list = []
        semantic_var_list = []

        # decomposed guidance
        for idx, imgs in enumerate(img_pr_list):
            view_i = preprocess(Image.fromarray(imgs)).unsqueeze(0).cuda()

            with torch.no_grad():
                image_features = model.encode_image(view_i)  # 1x768

            if idx == 0:
                reference_features = image_features

            neg_cosine_sim = 1 - F.cosine_similarity(reference_features, image_features, dim=1)
            text_cosine_sim = max(F.cosine_similarity(text_features, image_features, dim=1).item(), 0) * 100
            similarity_list.append(neg_cosine_sim.item())
            semantic_var_list.append(text_cosine_sim)

        D = sum(similarity_list)
        S_var = statistics.variance(semantic_var_list)
        msg=f'{name:<15}\t{D:.5f}\t{S_var:.5f}'
        print(msg)
        with open(os.path.abspath(os.path.join(pr_dir, '../', 'CD_score.log')), 'a') as f:
            f.write(msg+'\n')

if __name__=="__main__":
    main()