import numpy as np
import torch
from dem.models.components.distribution_distances import (
    compute_full_dataset_distribution_distances,
)
from dem.energies.gmm_energy import GMM
from dem.energies.multi_double_well_energy import MultiDoubleWellEnergy
import pickle
import argparse
import os
import matplotlib.pyplot as plt


# def compute_gaussian_tvd(samples1, samples_test, bins=200):
    
#     H_data, x_edges, y_edges = np.histogram2d(
#         samples_test[:, 0], samples_test[:, 1], bins=bins
#     )
#     H_gen, _, _ = np.histogram2d(
#         samples1[:, 0], samples1[:, 1], bins=[x_edges, y_edges]
#     )
#     H_data_norm = H_data / H_data.sum()
#     H_gen_norm = H_gen / H_gen.sum()
#     total_var = 0.5 * np.abs(H_data_norm - H_gen_norm).sum()
#     return total_var


# def compute_symmetric_gaussian_tvd(samples1, samples_test, bins=200):
#     tvd1 = compute_gaussian_tvd(samples1, samples_test, bins)
#     tvd2 = compute_gaussian_tvd(samples_test, samples1, bins)
#     return (tvd1 + tvd2) / 2


def compute_total_var_energy(energy_function, generated_samples, data_set):
    generated_samples_energy = (
        energy_function(generated_samples).cpu().numpy().reshape(-1),
    )
    data_set_energy = energy_function(data_set).cpu().numpy().reshape(-1)
    
    
    H_data_set, x_data_set = np.histogram(generated_samples_energy, bins=200)
    H_generated_samples, _ = np.histogram(data_set_energy, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )
    return total_var


def compute_total_var_dist(energy_function, generated_samples, data_set):
    generated_samples_dists = (
        energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
    )
    data_set_dists = energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)
    H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
    H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )
    return total_var


def get_all_metric(energy_func, generated_samples):
    test_set = energy_func.sample_test_set(-1, full=True)
    # compute the total variation distance
    try:
        total_var = compute_total_var_dist(energy_func, energy_func.unnormalize(generated_samples), test_set)
        metric_dict = {'tv_dist': total_var}
        total_var = compute_total_var_energy(energy_func, energy_func.unnormalize(generated_samples), test_set)
        metric_dict = {'tv_energy': total_var}
    except:
        total_var = compute_total_var_energy(energy_func, energy_func.unnormalize(generated_samples), test_set)
        metric_dict = {'tv_energy': total_var}
    idx = torch.randperm(len(generated_samples))[:10000]
    names, dists = compute_full_dataset_distribution_distances(
        energy_func.unnormalize(generated_samples)[idx, None],
        test_set[:, None],
        energy_func,
    )
    for name, dist in zip(names, dists):
        metric_dict[name] = dist
    return metric_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='mog')
    parser.add_argument('--sample_path', type=str, default=None)
    parser.add_argument('--normalize_sample', action='store_true')
    parser.add_argument('--save_des', type=str, required=True)

    args = parser.parse_args()
    
    base_dir = './saved_metrics_figs/'+args.save_des
    # if os.path.exists(base_dir):
    #     print(f"Directory {base_dir} already exists. Please choose a different name.")
    #     exit(1)
    # else:
    #     os.makedirs(base_dir)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    if args.target == 'mog':
        energy = GMM()
        plotting_bounds = (-1.4 * 40, 1.4 * 40)
        target_type = 'mog'
    elif args.target =='mog80':
        energy = GMM(
        dimensionality=2,
        n_mixes=80,
        loc_scaling=80,
        data_normalization_factor=100,
        test_set_size=2000)
        plotting_bounds = (-1.4 * 80, 1.4 * 80)
        target_type = 'mog'
    elif args.target == 'mog120':
        energy = GMM(
        dimensionality=2,
        n_mixes=120,
        loc_scaling=120,
        data_normalization_factor=150,
        test_set_size=3000)
        plotting_bounds = (-1.4 * 120, 1.4 * 120)
        target_type = 'mog'
    elif args.target == 'dw':
        energy = MultiDoubleWellEnergy(
        dimensionality=8,
        n_particles=4,
        data_path="data/test_split_DW4.npy",
        data_path_train="data/train_split_DW4.npy",
        data_path_val="data/val_split_DW4.npy",)
        target_type = 'dw'
        
    else:
        raise NotImplementedError(f"Target {args.target} not implemented")
    
    if args.normalize_sample:
        samples = energy.normalize(torch.load(args.sample_path+'/samples_100000.pt')).cpu()
    else:
        samples = torch.load(args.sample_path+'/samples_100000.pt').cpu()
    
    sampled_samples = samples[torch.randint(0, len(samples), (len(energy._test_set),))]
    if target_type == 'mog':
        energy.get_single_dataset_fig(energy.unnormalize(sampled_samples), '', plotting_bounds=plotting_bounds)
        plt.savefig(base_dir + '/{}_fig.pdf'.format(args.save_des))
        energy.get_single_dataset_fig(energy._test_set, '',plotting_bounds=plotting_bounds)
        plt.savefig(base_dir + '/gt.pdf')
    elif target_type == 'dw':
        energy.get_dataset_fig(energy.unnormalize(sampled_samples))
        plt.savefig(base_dir + '/{}_fig.pdf'.format(args.save_des))
    all_metric = get_all_metric(energy, samples)
    # save the metrics
    with open(base_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(all_metric, f)

    