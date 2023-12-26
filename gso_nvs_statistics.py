import numpy as np
import argparse
import os

def run_nvs_statistics(log_path, stats_log_path, num_samples, num_objects):
    results = np.zeros((4, 3))

    with open(log_path, 'r') as f:
        data = f.readlines()
        for i, line in enumerate(data):
            if i % num_samples == 0:
                tmp_results = np.zeros((3, num_samples))

            tmp_data = line.split('\t')
            tmp_psnr = float(tmp_data[1])
            tmp_ssim = float(tmp_data[2])
            tmp_lpips = float(tmp_data[3].split('\n')[0])

            tmp_results[0, i % num_samples] = tmp_psnr
            tmp_results[1, i % num_samples] = tmp_ssim
            tmp_results[2, i % num_samples] = tmp_lpips

            if i % num_samples == num_samples-1:
                results[0] += np.max(tmp_results, 1)
                results[1] += np.mean(tmp_results, 1)
                results[2] += np.min(tmp_results, 1)
                results[3] += np.var(tmp_results, 1)

    results = results/num_objects

    with open(stats_log_path, 'a') as f:
        msg = f'Max: PSNR {results[0, 0]:.4f}\tSSIM {results[0, 1]:.4f}\tLPIPS {results[0, 2]:.4f}'
        print (msg)
        f.write(msg + '\n')

        msg = f'Mean: PSNR {results[1, 0]:.4f}\tSSIM {results[1, 1]:.4f}\tLPIPS {results[1, 2]:.4f}'
        print (msg)
        f.write(msg + '\n')

        msg = f'Min: PSNR {results[2, 0]:.4f}\tSSIM {results[2, 1]:.4f}\tLPIPS {results[2, 2]:.4f}'
        print (msg)
        f.write(msg + '\n')

        msg = f'Var: PSNR {results[3, 0]:.8f}\tSSIM {results[3, 1]:.8f}\tLPIPS {results[3, 2]:.8f}'
        print (msg)
        f.write(msg + '\n')

def run_consistency_stastics(log_path, stats_log_path, num_samples, num_objects):
    results = np.zeros((4, 1))

    with open(log_path, 'r') as f:
        data = f.readlines()
        for i, line in enumerate(data):
            if i % num_samples == 0:
                tmp_results = np.zeros((1, num_samples))

            tmp_data = line.split('\t')
            tmp_flow = float(tmp_data[1].split('\n')[0])

            tmp_results[0, i % num_samples] = tmp_flow

            if i % num_samples == num_samples-1:
                results[0] += np.max(tmp_results, 1)
                results[1] += np.mean(tmp_results, 1)
                results[2] += np.min(tmp_results, 1)
                results[3] += np.var(tmp_results, 1)

        results = results/num_objects
    with open(stats_log_path, 'a') as f:
        msg = f'E_flow : Max={results[0, 0]:.4f}\t Mean={results[1, 0]:.4f}\t Min={results[2, 0]:.4f}\t Var={results[3, 0]:.8f}'
        print (msg)
        f.write(msg + '\n')

def run_CD_stastics(log_path, stats_log_path, num_samples, num_objects):
    with open(log_path, 'r') as f:
        data = f.readlines()
        D = 0
        Svar = 0
        for i, line in enumerate(data):
            tmp_data = line.split('\t')
            tmp_D = float(tmp_data[1])
            tmp_Svar = float(tmp_data[2].split('\n')[0])
            D += tmp_D
            Svar += tmp_Svar

        D = D/(num_samples * num_objects)
        Svar = Svar/(num_samples * num_objects)
    with open(stats_log_path, 'a') as f:
        msg = f'CD Score : {D/Svar:.4f}'
        print (msg)
        f.write(msg + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default='output_gso')
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_objects", type=int, default=30)
    args = parser.parse_args()

    nvs_log_path = os.path.join(args.log, 'nvs.log')
    consistency_log_path = os.path.join(args.log, 'consistency.log')
    CD_log_path = os.path.join(args.log, 'CD_score.log')
    stats_log_path = os.path.join(args.log, 'statistics.log')

    run_nvs_statistics(nvs_log_path, stats_log_path, args.num_samples, args.num_objects)
    run_consistency_stastics(consistency_log_path, stats_log_path, args.num_samples, args.num_objects)
    run_CD_stastics(CD_log_path, stats_log_path, args.num_samples, args.num_objects)
