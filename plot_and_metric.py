import numpy as np
import torch
from dem.models.components.distribution_distances import (
    compute_full_dataset_distribution_distances,
)
from dem.energies.gmm_energy import GMM
from dem.energies.multi_double_well_energy import MultiDoubleWellEnergy
from dem.energies.lennardjones_energy import LennardJonesEnergy
import pickle
import argparse
import os
import matplotlib.pyplot as plt


def compute_gaussian_tvd(samples1, samples_test, bins):
    
    H_data, x_edges, y_edges = np.histogram2d(
        samples_test[:, 0], samples_test[:, 1], bins=bins
    )
    H_gen, _, _ = np.histogram2d(
        samples1[:, 0], samples1[:, 1], bins=[x_edges, y_edges]
    )
    H_data_norm = H_data / H_data.sum()
    H_gen_norm = H_gen / H_gen.sum()
    total_var = 0.5 * np.abs(H_data_norm - H_gen_norm).sum()
    return total_var



def compute_total_var_energy(energy_function, generated_samples, data_set, bins):
    generated_samples_energy = energy_function(generated_samples).detach().cpu().numpy().reshape(-1)
    
    data_set_energy = energy_function(data_set).detach().cpu().numpy().reshape(-1)
    
    H_data_set, x_data_set = np.histogram(data_set_energy, bins=bins)
    H_generated_samples, _ = np.histogram(generated_samples_energy, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )
    return total_var


def compute_total_var_dist(energy_function, generated_samples, data_set, bins):
    generated_samples_dists = energy_function.interatomic_dist(generated_samples).detach().cpu().numpy().reshape(-1)
    data_set_dists = energy_function.interatomic_dist(data_set).detach().cpu().numpy().reshape(-1)
    H_data_set, x_data_set = np.histogram(data_set_dists, bins=bins)
    H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )
    return total_var

def get_all_metric(energy_func, generated_samples, bins, metric_dict, update_metric):
    test_set = energy_func.sample_test_set(-1, full=True)

    # 'tv_energy', 'tv_dist', 'tv_sample'
    if update_metric is not None:
        if update_metric == 'tv_energy':
            metric_dict[update_metric] = compute_total_var_energy(energy_func, energy_func.unnormalize(generated_samples), test_set, bins)
        elif update_metric == 'tv_dist':
            metric_dict[update_metric] = compute_total_var_dist(energy_func, energy_func.unnormalize(generated_samples), test_set, bins)
        elif update_metric == 'tv_sample':
            metric_dict[update_metric] = compute_gaussian_tvd(energy_func.unnormalize(generated_samples), test_set, bins)
        else:
            raise NotImplementedError
    else:
        # compute the total variation distance
        if energy_func.name=='gmm':
            total_var = compute_total_var_energy(energy_func, energy_func.unnormalize(generated_samples), test_set, bins)
            metric_dict['tv_energy'] = total_var
            
            total_val = compute_gaussian_tvd(
                energy_func.unnormalize(generated_samples),
                test_set,
                bins
            )
            metric_dict['tv_sample'] = total_val
        else:
            total_var = compute_total_var_dist(energy_func, energy_func.unnormalize(generated_samples), test_set, bins)
            metric_dict['tv_dist'] = total_var
            total_var = compute_total_var_energy(energy_func, energy_func.unnormalize(generated_samples), test_set, bins)
            metric_dict['tv_energy'] = total_var
            
        idx = torch.randperm(len(generated_samples))[:10000]
        names, dists = compute_full_dataset_distribution_distances(
            energy_func.unnormalize(generated_samples)[idx, None],
            test_set[:, None],
            energy_func,
            NAMES = [
            "1-Wasserstein",
            "2-Wasserstein",
        ]
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
    parser.add_argument('--update_metric', type=str, default=None, choices=['tv_energy', 'tv_dist', 'tv_sample'], help='update the metric for which may be not exist or not correct')

    args = parser.parse_args()
    
    base_dir = './saved_metrics_figs/'+args.save_des
    if args.update_metric is None:
        if os.path.exists(base_dir):
            original_base_dir = base_dir
            repeat_counter = 1
            
            while True:
                new_dir = f"{original_base_dir}_repeat_{repeat_counter}"
                if not os.path.exists(new_dir):
                    base_dir = new_dir
                    break
                repeat_counter += 1
            
            print(f"Directory {original_base_dir} already exists. Creating {base_dir} instead.")

        os.makedirs(base_dir)
        existed_all_metric = {}
    else:
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"The metrics file does not exist at {base_dir}")
        with open(base_dir + '/metrics.pkl', 'rb') as f:
            existed_all_metric = pickle.load(f)
    
    if args.target == 'mog':
        energy = GMM(test_set_size=10000)
        plotting_bounds = (-1.4 * 40, 1.4 * 40)
        target_type = 'mog'
    elif args.target =='mog80':
        energy = GMM(
        dimensionality=2,
        n_mixes=80,
        loc_scaling=80,
        data_normalization_factor=100,
        test_set_size=10000)
        plotting_bounds = (-1.4 * 80, 1.4 * 80)
        target_type = 'mog'
    elif args.target == 'mog120':
        energy = GMM(
        dimensionality=2,
        n_mixes=120,
        loc_scaling=120,
        data_normalization_factor=150,
        test_set_size=10000)
        plotting_bounds = (-1.4 * 120, 1.4 * 120)
        target_type = 'mog'
    elif args.target == 'dw':
        energy = MultiDoubleWellEnergy(
        dimensionality=8,
        n_particles=4,
        data_path="data/test_split_DW4.npy",
        data_path_train="data/train_split_DW4.npy",
        data_path_val="data/val_split_DW4.npy")
        target_type = 'dw'
    elif args.target == 'lj13':
        energy = LennardJonesEnergy(
        dimensionality=39,
        n_particles=13,
        data_path="data/test_split_LJ13-1000.npy",
        data_path_train="data/train_split_LJ13-1000.npy",
        data_path_val="data/val_split_LJ13-1000.npy",
        data_path_test="data/test_split_LJ13-1000.npy",
        )
        target_type = 'lj13'
    elif args.target == 'lj55':
        energy = LennardJonesEnergy(
        dimensionality=165,
        n_particles=55,
        data_path="data/test_split_LJ55-1000-part1.npy",
        data_path_train="data/train_split_LJ55-1000-part1.npy",
        data_path_val="data/val_split_LJ55-1000-part1.npy",
        data_path_test="data/test_split_LJ55-1000-part1.npy",
        )
        target_type = 'lj55'
        
    else:
        raise NotImplementedError(f"Target {args.target} not implemented")
    
    samples = torch.load(args.sample_path,weights_only=True).cpu()

    bins = int(np.sqrt(len(energy._test_set)))
    samples = samples.to(torch.float32)
    if target_type == 'mog':
        valid_mask = ~torch.isnan(samples).any(dim=1) & ~torch.isinf(samples).any(dim=1)
        samples = samples[valid_mask]
        samples = torch.clamp(samples, 
                                    -energy.data_normalization_factor, 
                                    energy.data_normalization_factor)

    if args.normalize_sample:
        samples = energy.normalize(samples)
    
    sampled_samples = samples[torch.randint(0, len(samples), (len(energy._test_set),))]
    
    if args.update_metric is None:
        if target_type == 'mog':
            energy.get_single_dataset_fig(energy.unnormalize(sampled_samples), '', plotting_bounds=plotting_bounds)
            plt.savefig(base_dir + '/{}_fig.pdf'.format(args.save_des))
            energy.get_single_dataset_fig(energy._test_set, '',plotting_bounds=plotting_bounds)
            plt.savefig(base_dir + '/gt.pdf')
        # elif target_type == 'dw':
        else:
            energy.get_dataset_fig(energy.unnormalize(sampled_samples))
            plt.savefig(base_dir + '/{}_fig.pdf'.format(args.save_des))
            
    all_metric = get_all_metric(energy, sampled_samples, bins, existed_all_metric, args.update_metric)
    # beautifully print all_metric, only contains: 'tv' or 'Wassertein'
    print('Metrics:')
    for key, value in all_metric.items():
        if key.startswith('tv') or key.endswith('Wasserstein'):
            print(f"{key}: {value:.4f}")
    
    # save the metrics
    with open(base_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(all_metric, f)

    