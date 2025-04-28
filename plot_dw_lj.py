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


method_map = {
    'idem': 'iDEM',
    'fab': 'FAB',
    'dikl': 'DiKL',
    'snis': 'Ours',
}


def get_dataset_fig_dw(energy_func, samples_dict, save_path=None):
    # 设置NIPS风格的字体和样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Sans']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 设置全局字体大小
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    linewidth=5
    
    plt.rc('font', size=SMALL_SIZE)          # 默认字体大小
    plt.rc('axes', titlesize=BIGGER_SIZE)    # 坐标轴标题字体大小
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # 坐标轴标签字体大小
    plt.rc('xtick', labelsize=SMALL_SIZE)    # x轴刻度标签字体大小
    plt.rc('ytick', labelsize=SMALL_SIZE)    # y轴刻度标签字体大小
    plt.rc('legend', fontsize=SMALL_SIZE)    # 图例字体大小
    plt.rc('figure', titlesize=BIGGER_SIZE)  # 图标题字体大小
    
    # 使用科学风格的颜色方案
    colors ={
        'gt': '#1f77b4',  # 蓝色用于Ground Truth
        'snis': '#d62728',  # 鲜红色
        # 'idem': '#2ca02c',  # 绿色
        # 'dikl': '#9467bd',  # 紫色
        # '#8c564b',  # 棕色
        'idem': '#e377c2',  # 粉色
        'dikl': '#ff7f0e',  # 橙色
        # 'fab': '#17becf',  # 青色
        'fab': '#bcbd22',  # 橄榄色
    }
    ground_truth_color = colors['gt']
    
    test_data_smaller = energy_func.sample_test_set(10000)
    dist_test = energy_func.interatomic_dist(test_data_smaller).detach().cpu()
    energy_test = -energy_func(test_data_smaller).detach().cpu()
    min_energy = -26
    max_energy = 0
    bins = int(np.sqrt(len(test_data_smaller)))

    
    # 创建两个独立的图形，使用更高质量的设置
    fig_dist = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.set_xlabel("Interatomic Distance", fontweight='bold')
    ax_dist.set_ylabel("Probability Density", fontweight='bold')
    ax_dist.spines['top'].set_visible(False)  # 移除上边框
    ax_dist.spines['right'].set_visible(False)  # 移除右边框
    ax_dist.grid(True, linestyle='--', alpha=0.3)  # 添加网格线
    ax_dist.set_title("Distribution of Interatomic Distances on DW-4", fontweight='bold')
    
    fig_energy = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.set_xlabel("Energy", fontweight='bold')
    ax_energy.set_ylabel("Probability Density", fontweight='bold')
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['right'].set_visible(False)
    ax_energy.grid(True, linestyle='--', alpha=0.3)
    ax_energy.set_title("Distribution of Energy Values on DW-4", fontweight='bold')

    # 绘制第一个图：原子间距离分布
    ax_dist.hist(
        dist_test.view(-1),
        bins=bins,
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=linewidth,
        color=ground_truth_color,
        label="Ground Truth",
    )    
    
    # 绘制第二个图：能量分布
    ax_energy.hist(
        energy_test.cpu(),
        bins=bins,
        density=True,
        alpha=0.7,
        range=(min_energy, max_energy),
        color=ground_truth_color,
        histtype="step",
        linewidth=linewidth,
        label="Ground Truth",
    )
    method_order = ['fab', 'idem', 'dikl', 'snis']
    
    # 为每个方法添加直方图，使用不同的颜色
    for i, method in enumerate(method_order):
        samples = samples_dict[method]
        dist_samples = energy_func.interatomic_dist(samples).detach().cpu()
        energy_samples = -energy_func(samples).detach().cpu()
        
        method_color = colors[method]  # 不同方法使用不同颜色

        # 在第一个图中添加原子间距离分布
        ax_dist.hist(
            dist_samples.view(-1),
            bins=bins,
            alpha=0.7,
            density=True,
            histtype="step",
            linewidth=linewidth,
            color=method_color,
            label=method_map[method],
        )
        
        # 在第二个图中添加能量分布
        ax_energy.hist(
            energy_samples.cpu(),
            bins=bins,
            density=True,
            alpha=0.7,
            range=(min_energy, max_energy),
            color=method_color,
            histtype="step",
            linewidth=linewidth,
            label=method_map[method],
        )
    
    # 优化图例
    ax_dist.legend(loc='upper left', frameon=True, framealpha=0.9, 
                  edgecolor='lightgray', fancybox=True)
    ax_energy.legend(loc='upper right', frameon=True, framealpha=0.9, 
                    edgecolor='lightgray', fancybox=True)
    
    # 调整布局
    fig_dist.tight_layout()
    fig_energy.tight_layout()
    
    # 如果提供了保存路径，则分别保存两个图，使用高质量设置
    if save_path:
        dist_path = f"{save_path}_distance.pdf"  # 使用PDF获得更好的矢量图质量
        energy_path = f"{save_path}_energy.pdf"
        fig_dist.savefig(dist_path, bbox_inches='tight', dpi=300)
        fig_energy.savefig(energy_path, bbox_inches='tight', dpi=300)
    return (fig_dist, ax_dist), (fig_energy, ax_energy)



def get_dataset_fig_lj(energy_func, samples_dict, save_path=None):
    # 设置NIPS风格的字体和样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Sans']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 设置全局字体大小
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    linewidth=5
    
    plt.rc('font', size=SMALL_SIZE)          # 默认字体大小
    plt.rc('axes', titlesize=BIGGER_SIZE)    # 坐标轴标题字体大小
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # 坐标轴标签字体大小
    plt.rc('xtick', labelsize=SMALL_SIZE)    # x轴刻度标签字体大小
    plt.rc('ytick', labelsize=SMALL_SIZE)    # y轴刻度标签字体大小
    plt.rc('legend', fontsize=SMALL_SIZE)    # 图例字体大小
    plt.rc('figure', titlesize=BIGGER_SIZE)  # 图标题字体大小
    
    # 定义高对比度颜色
    high_contrast_colors ={
        'gt': '#1f77b4',  # 蓝色用于Ground Truth
        'snis': '#d62728',  # 鲜红色
        # 'idem': '#2ca02c',  # 绿色
        # 'dikl': '#9467bd',  # 紫色
        # '#8c564b',  # 棕色
        'idem': '#e377c2',  # 粉色
        'dikl': '#ff7f0e',  # 橙色
        # 'fab': '#17becf',  # 青色
        'fab': '#bcbd22',  # 橄榄色
    }
    ground_truth_color = high_contrast_colors['gt']
    # 根据粒子数设置能量范围和分箱数
    if energy_func.n_particles == 13:
        min_energy = -60
        max_energy = 0
        bins=100
        test_data_smaller = energy_func.sample_test_set(10000)

    elif energy_func.n_particles == 55:
        min_energy = -380
        max_energy = -180
        bins=50
        test_data_smaller = energy_func.sample_test_set(1000)

    dist_test = energy_func.interatomic_dist(test_data_smaller).detach().cpu()
    energy_test = -energy_func(test_data_smaller).detach().cpu()  # 修复多余的detach()
    
    # bins = int(np.sqrt(len(test_data_smaller)))
    
    # 创建两个独立的图形
    fig_dist = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.set_xlabel("Interatomic Distance", fontweight='bold')
    ax_dist.set_ylabel("Probability Density", fontweight='bold')
    ax_dist.spines['top'].set_visible(False)  # 移除上边框
    ax_dist.spines['right'].set_visible(False)  # 移除右边框
    ax_dist.grid(True, linestyle='--', alpha=0.3)  # 添加网格线
    ax_dist.set_title("Distribution of Interatomic Distances", fontweight='bold')
    
    # 添加LJ粒子数量到标题
    if hasattr(energy_func, 'n_particles'):
        ax_dist.set_title(f"Distribution of Interatomic Distances on LJ-{energy_func.n_particles}", fontweight='bold')
    
    fig_energy = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.set_xlabel("Energy", fontweight='bold')
    ax_energy.set_ylabel("Probability Density", fontweight='bold')
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['right'].set_visible(False)
    ax_energy.grid(True, linestyle='--', alpha=0.3)
    
    # 添加LJ粒子数量到标题
    if hasattr(energy_func, 'n_particles'):
        ax_energy.set_title(f"Distribution of Energy Values on LJ-{energy_func.n_particles}", fontweight='bold')
    else:
        ax_energy.set_title("Distribution of Energy Values", fontweight='bold')

    # 绘制第一个图：原子间距离分布
    ax_dist.hist(
        dist_test.view(-1),
        bins=bins,
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=linewidth,
        color=ground_truth_color,
        label="Ground Truth",
    )
    
    # 绘制第二个图：能量分布
    ax_energy.hist(
        energy_test.cpu(),
        bins=bins,
        density=True,
        alpha=0.7,
        range=(min_energy, max_energy),
        color=ground_truth_color,
        histtype="step",
        linewidth=linewidth,
        label="Ground Truth",
    )
    if energy_func.n_particles == 13:
        method_order = ['fab', 'idem', 'dikl', 'snis']
    else:
        method_order = ['fab', 'idem', 'snis']
    
    # 为每个方法添加直方图，使用不同的颜色
    for i, method in enumerate(method_order):
        samples = samples_dict[method]
        dist_samples = energy_func.interatomic_dist(samples).detach().cpu()
        energy_samples = -energy_func(samples).detach().cpu()  # 修复多余的detach()
        
        method_color = high_contrast_colors[method]  # 不同方法使用不同颜色

        # 在第一个图中添加原子间距离分布
        ax_dist.hist(
            dist_samples.view(-1),
            bins=bins,
            alpha=0.7,
            density=True,
            histtype="step",
            linewidth=linewidth,
            color=method_color,
            label=method_map[method],
        )
        
        # 在第二个图中添加能量分布 - 使用方法名称而非固定的"generated data"
        if not (method == 'fab' and energy_func.n_particles == 55):
            ax_energy.hist(
                energy_samples.cpu(),
                bins=bins,
                density=True,
                alpha=0.7,
                range=(min_energy, max_energy),
                color=method_color,
                histtype="step",
                linewidth=linewidth,
                label=method_map[method],  # 使用方法名称而不是"generated data"
            )
    
    # 优化图例
    ax_dist.legend(loc='upper right', frameon=True, framealpha=0.9, 
                  edgecolor='lightgray', fancybox=True)
    ax_energy.legend(loc='upper right', frameon=True, framealpha=0.9, 
                    edgecolor='lightgray', fancybox=True)
    
    # 调整布局
    fig_dist.tight_layout()
    fig_energy.tight_layout()
    
    # 如果提供了保存路径，则分别保存两个图
    if save_path:
        dist_path = f"{save_path}_distance.pdf"
        energy_path = f"{save_path}_energy.pdf"
        fig_dist.savefig(dist_path, bbox_inches='tight', dpi=300, format='pdf')
        fig_energy.savefig(energy_path, bbox_inches='tight', dpi=300, format='pdf')
    return (fig_dist, ax_dist), (fig_energy, ax_energy)

def get_samples_for_methods(target, energy):
    if target == 'dw':
        idem_samples = torch.load('/home/wangchenguang/code/DEM/dem_results/logs/train/runs/2025-04-19_09-39-30_dw4_idem/samples_100000.pt', weights_only=True).cpu()
        fab_samples = energy.normalize(torch.load('/home/wangchenguang/code/DEM/fab_results/dw4_0/samples_100000.pt', weights_only=True).cpu())
        dikl_samples = energy.normalize(torch.load('/home/wangchenguang/code/DEM/dikl_results/dw4/samples_100000.pt', weights_only=True).cpu())
        snis_samples = torch.load('/home/wangchenguang/code/DEM/dem_results/logs/train/runs/2025-04-13_15-01-43_dw4_snis-2/samples_100000.pt', weights_only=True).cpu()
        samples_dict = {
            'idem': idem_samples,
            'fab': fab_samples,
            'dikl': dikl_samples,
            'snis': snis_samples
        }
    elif target == 'lj13':
        idem_samples = torch.load('/home/wangchenguang/code/DEM/dem_results/logs/train/runs/2025-04-17_00-00-22_lj13_idem/samples_100000.pt', weights_only=True).cpu()
        fab_samples = energy.normalize(torch.load('/home/wangchenguang/code/DEM/fab_results/lj13_0/samples_100000.pt', weights_only=True).cpu())
        dikl_samples = energy.normalize(torch.load('/home/wangchenguang/code/DEM/dikl_results/lj13/samples_100000.pt', weights_only=True).cpu())
        snis_samples = torch.load('/home/wangchenguang/code/DEM/dem_results/logs/train/runs/2025-04-23_08-40-54_lj13_snis5_bf50/samples_100000.pt', weights_only=True).cpu()
        samples_dict = {
            'idem': idem_samples,
            'fab': fab_samples,
            'dikl': dikl_samples,
            'snis': snis_samples
        }
    elif target == 'lj55':
        idem_samples = torch.load('/home/wangchenguang/code/DEM/dem_results/logs/eval/runs/2025-04-21_12-58-55_lj55_idem/samples_10000.pt', weights_only=True).cpu()
        fab_samples = energy.normalize(torch.load('/home/wangchenguang/code/DEM/fab_results/lj55_0/samples_10000.pt', weights_only=True).cpu())
        snis_samples = torch.load('/home/wangchenguang/code/DEM/dem_results/logs/train/runs/2025-04-22_14-06-55_lj55_snis/samples_10000.pt', weights_only=True).cpu()
        samples_dict = {
            'idem': idem_samples,
            'fab': fab_samples,
            'snis': snis_samples
        }
    else:
        raise NotImplementedError(f"Target {target} not implemented")
    return samples_dict


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='dw')
    parser.add_argument('--save_des', type=str, required=True)

    args = parser.parse_args()
    
    if args.target == 'dw':
        energy = MultiDoubleWellEnergy(
        dimensionality=8,
        n_particles=4,
        data_path="data/test_split_DW4.npy",
        data_path_train="data/train_split_DW4.npy",
        data_path_val="data/val_split_DW4.npy")
        target_type = 'dw'
        get_fig_fun = get_dataset_fig_dw
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
        get_fig_fun = get_dataset_fig_lj
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
        get_fig_fun = get_dataset_fig_lj
        
    else:
        raise NotImplementedError(f"Target {args.target} not implemented")
    
    samples_dict = get_samples_for_methods(args.target, energy)
    fig, axs = get_fig_fun(energy, samples_dict, save_path=f'paper_figs/{args.target}_{args.save_des}')
    