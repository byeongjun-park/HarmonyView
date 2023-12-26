import argparse
import torch
from skimage.io import imread
from RAFT.raft import RAFT

from RAFT.flow_utils import *

DEVICE = 'cuda'

def convert_to_rgb(img):
    return img[:, :3] * (img[:, 3:]/255) + (255 - img[:, 3:])

def run(args, pr_path, gt_path, num_samples):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()


    with torch.no_grad():
        img_gt_list = [imread(os.path.join(gt_path, f'{k:03}.png')) for k in range(16)]
        img_gt_list.insert(0, img_gt_list[-1])
        img_gt_list.append(img_gt_list[1])
        flow_fwd_list, flow_bwd_list = [], []
        for i in range(1, 17):
            img_bw = torch.from_numpy(img_gt_list[i - 1]).permute(2, 0, 1).float()[None].cuda()
            img_ref = torch.from_numpy(img_gt_list[i]).permute(2, 0, 1).float()[None].cuda()
            img_fw = torch.from_numpy(img_gt_list[i + 1]).permute(2, 0, 1).float()[None].cuda()
            img_bw, img_ref, img_fw = convert_to_rgb(img_bw), convert_to_rgb(img_ref), convert_to_rgb(img_fw)
            _, flow_fwd = model(img_ref, img_fw, iters=20, test_mode=True)
            _, flow_bwd = model(img_ref, img_bw, iters=20, test_mode=True)

            flow_fwd_list.append(flow_fwd)
            flow_bwd_list.append(flow_bwd)

        for bi in range(num_samples):
            errors_flow = 0
            img_pr_list = imread(os.path.join(pr_path, f'{bi}.png'))
            img_pr_list = np.array_split(img_pr_list, 16, 1)
            img_pr_list.insert(0, img_pr_list[-1])
            img_pr_list.append(img_pr_list[1])
            name = args.name + f'-{bi}'
            for i in range(1, 17):
                img_bw = torch.from_numpy(img_pr_list[i-1]).permute(2, 0, 1).float()[None].cuda()
                img_ref = torch.from_numpy(img_pr_list[i]).permute(2, 0, 1).float()[None].cuda()
                img_fw = torch.from_numpy(img_pr_list[i+1]).permute(2, 0, 1).float()[None].cuda()

                _, flow_fwd = model(img_ref, img_fw, iters=20, test_mode=True)
                _, flow_bwd = model(img_ref, img_bw, iters=20, test_mode=True)

                errors_flow += 0.5 * (torch.abs(flow_fwd_list[i-1] - flow_fwd).mean() + torch.abs(flow_bwd_list[i-1] - flow_bwd).mean())
            msg = f'{name:<15}\t{(errors_flow/16):.4f}'
            print(msg)
            with open(os.path.abspath(os.path.join(pr_path, '../', 'consistency.log')), 'a') as f:
                f.write(msg + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=str, help='Predict Image path')
    parser.add_argument("--gt", type=str, help='GT Image path')
    parser.add_argument("--name", type=str, help='Image path')
    parser.add_argument('--model', help="restore RAFT checkpoint", default='RAFT/weights/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    pr_path = os.path.join(args.pr)
    gt_path = os.path.join(args.gt)

    run(args, pr_path, gt_path, args.num_samples)
