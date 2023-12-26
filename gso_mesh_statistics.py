import numpy as np
import argparse
import os

def run_mesh_stastics(log_path, stat_path, num_samples, num_objects):
    results = np.zeros((1, 2))
    mean_distance = 0
    mean_iou = 0

    with open(log_path, 'r') as f:
        data = f.readlines()
        for i, line in enumerate(data):
            if i % num_samples == 0:
                best_distance = 10000
                best_iou = 0

            tmp_data = line.split('\t')
            tmp_distance = float(tmp_data[1])
            tmp_iou = float(tmp_data[2].split('\n')[0])

            mean_distance += tmp_distance
            mean_iou += tmp_iou

            if best_distance > tmp_distance:
                best_distance = tmp_distance

            if best_iou < tmp_iou:
                best_iou = tmp_iou

            if i % num_samples == num_samples-1:
                results[0, 0] += best_distance
                results[0, 1] += best_iou

        results = results/num_objects

    msg = f'Best Chamfer_Distance {results[0, 0]:.5f}\tBest Volume_IoU {results[0, 1]:.5f}'
    print(msg)
    with open(stat_path, 'a') as f:
        f.write(msg + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default='output_gso_renderer')
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_objects", type=int, default=30)

    args = parser.parse_args()

    log_path = os.path.join(args.log, 'geometry.log')
    stat_path = os.path.join(args.log, 'statistics.log')
    run_mesh_stastics(log_path, stat_path, args.num_samples, args.num_objects)
