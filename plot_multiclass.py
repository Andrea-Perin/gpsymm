#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import jax.numpy as jnp
from pathlib import Path
import einops as ein

from utils.conf import load_config
from utils.plot_utils import cm
plt.style.use('myplots.mlpstyle')


def main():
    parser = argparse.ArgumentParser(description='Create plots from results file')
    parser.add_argument('results_dir', type=str, help='Path to the folder containing numpy result files')
    parser.add_argument('output_path', type=str, help='Directory to save the plots')
    parser.add_argument('--alpha', type=float, default=.1,
                       help='Transparency of scatter dots (default: .1)')
    args = parser.parse_args()

    # Load config and results
    cfg = load_config()
    angles = cfg['params']['rotations']
    res_dir = Path(args.results_dir)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load the three files (shape: 'angle test digit seed')
    regression_res = np.load(res_dir / 'regression_predictions.npy')
    training_res = np.load(res_dir / 'training_predictions.npy')
    spectral_res = np.load(res_dir / 'spectral_predictions.npy')

    # regression: for every digit, check how many seeds are positive
    reg_acc = ein.reduce(
        (regression_res > 0).astype(float),
        'angle test digit seed -> angle digit', 'mean'
    )
    # training: for every digit, check how many seeds are correct
    train_acc = ein.reduce(
        (training_res == jnp.arange(10)[:, None]).astype(float),
        'angle test digit seed -> angle digit', 'mean'
    )
    # spectral: count how many results are positive for each digit
    spectral_acc = ein.reduce(
        (spectral_res == 1).astype(float),
        'angle test digit seed -> angle digit', 'mean'
    )

    # compute averages and confidence intervals
    reg_avg = ein.reduce(reg_acc, 'angle digit -> angle', 'mean')
    reg_c95 = 1.96 * ein.reduce(reg_acc, 'angle digit -> angle', jnp.std) / jnp.sqrt(10)
    train_avg = ein.reduce(train_acc, 'angle digit -> angle', 'mean')
    train_c95 = 1.96 * ein.reduce(train_acc, 'angle digit -> angle', jnp.std) / jnp.sqrt(10)
    spectral_avg = ein.reduce(spectral_acc, 'angle digit -> angle', 'mean')
    spectral_c95 = 1.96 * ein.reduce(spectral_acc, 'angle digit -> angle', jnp.std) / jnp.sqrt(10)

    # Plot 1
    fig = plt.figure(figsize=(17*cm, 5*cm))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(1, 3),
        axes_pad=0.1,
        label_mode="L",
        share_all=True,
        # cbar_location="right",
        # cbar_mode="single",
        # cbar_size="5%",
        # cbar_pad=0.1,
        aspect=False
    )

    grid[0].set_xscale('log', base=2)
    grid[0].set_xticks(angles)
    grid[0].set_ylabel('Accuracy', fontsize=10)
    grid[1].set_xlabel(r'$N_{{rot}}$', fontsize=10)

    grid[0].set_title('Trained', fontsize=10)
    grid[0].text(-0, 1.05, r'$\mathbf{(A)}$', transform=grid[0].transAxes, fontsize=10)
    grid[0].plot(angles, train_avg, '-o')
    grid[0].fill_between(angles, train_avg-train_c95, train_avg+train_c95, alpha=.2)

    grid[1].set_title('Exact (NTK)', fontsize=10)
    grid[1].text(-0, 1.05, r'$\mathbf{(B)}$', transform=grid[1].transAxes, fontsize=10)
    grid[1].plot(angles, reg_avg, '-o')
    grid[1].fill_between(angles, reg_avg-reg_c95, reg_avg+reg_c95, alpha=.2)

    grid[2].set_title('Spectral (NTK)', fontsize=10)
    grid[2].text(-0, 1.05, r'$\mathbf{(C)}$', transform=grid[2].transAxes, fontsize=10)
    grid[2].plot(angles, spectral_avg, '-o')
    grid[2].fill_between(angles, spectral_avg-spectral_c95, spectral_avg+spectral_c95, alpha=.2)

    plt.tight_layout(pad=0.4)
    plt.savefig(out_dir / 'multiclass.pdf')
    # plt.show()
    plt.close()

    # Plot 2
    fig = plt.figure(figsize=(8*cm, 5*cm))
    ax = fig.add_subplot(111)
    tickspos = list(range(len(angles)))
    ax.set_xticks(tickspos, angles)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_xlabel(r'$N_{{rot}}$', fontsize=10)
    ax.plot(train_avg, '-o', label='Trained')
    ax.fill_between(tickspos, train_avg-train_c95, train_avg+train_c95, alpha=.2)
    ax.plot(spectral_avg, '-o', label='Spectral (NTK)')
    ax.fill_between(tickspos, spectral_avg-spectral_c95, spectral_avg+spectral_c95, alpha=.2)
    ax.legend(fontsize=8)
    plt.tight_layout(pad=0.4)
    plt.savefig(out_dir / 'multiclass_single_panel.pdf')
    # plt.show()
    plt.close()

if __name__ == '__main__':
    main()
