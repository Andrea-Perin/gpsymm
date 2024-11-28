#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import jax.numpy as jnp
from pathlib import Path


from utils.conf import load_config
from utils.plot_utils import cm, add_inner_title, semaphore
plt.style.use('myplots.mlpstyle')


def plot_panels(results, n_rotations, rot_idx, output_path, alpha):
    """Create and save plots B through E from the results."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Unpack results
    (deltasq, spectral_errors, empirical_errors, lambda_last,
     lambda_avg_no_last, lambda_avg, proj_radius, avg_angle, emp_counts) = results

    # Additional processing
    lambda_last *= jnp.sqrt(jnp.array(n_rotations)[:, None])

    # PANEL B
    fig = plt.figure(figsize=(5.75*cm, 5*cm))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
        cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
    grid[0].set_title(f"$N_{{rot}}={{{n_rotations[rot_idx]}}}$", fontsize=10)
    grid[0].set_xlabel("Empirical error (NTK)")
    grid[0].set_ylabel("Spectral error")
    im = grid[0].scatter(
        empirical_errors[rot_idx],
        spectral_errors[rot_idx],
        c=emp_counts[rot_idx],
        marker='.', alpha=alpha, s=2,
        cmap=semaphore,
    )
    cbar = grid.cbar_axes[0].colorbar(
        plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()),
        label='Num. correct',
        ticks=[0.5, 1.5, 2.5],
        alpha=1.0
    )
    cbar.ax.tick_params(size=0)
    cbar.ax.set_yticklabels((0, 1, 2))
    cmax = max(empirical_errors[rot_idx].max(), spectral_errors[rot_idx].max())
    grid[0].plot([0, cmax], [0, cmax], color='black', alpha=1, lw=.75, ls='--')
    grid[0].set_xlim((0, None))
    grid[0].set_ylim((0, None))
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path / f'panelB_{n_rotations[rot_idx]}.pdf')
    plt.close()

    # PANEL C
    ll = lambda_last[rot_idx]
    lnl = lambda_avg[rot_idx]
    spec_errs = spectral_errors[rot_idx]
    emp_errs = empirical_errors[rot_idx]

    xi = np.linspace(ll.min(), ll.max(), 100)
    yi = np.linspace(lnl.min(), lnl.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi_spec = griddata((ll, lnl), spec_errs, (xi, yi), method='linear')
    zi_emp = griddata((ll, lnl), emp_errs, (xi, yi), method='linear')

    fig = plt.figure(figsize=(5*cm, 10*cm))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(2, 1),
        axes_pad=0.05,
        label_mode="L",
        share_all=True,
        aspect=False,
        cbar_location="top",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05
    )

    vmin = min(np.nanmin(zi_spec), np.nanmin(zi_emp))
    vmax = max(np.nanmax(zi_spec), np.nanmax(zi_emp))
    titlestr = f'$N_{{rot}}={n_rotations[rot_idx]}$'

    im1 = grid[0].contourf(xi, yi, zi_spec, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
    grid[0].set_xlabel(r'$\lambda^{-1}_{N}$')
    grid[0].set_ylabel(r'$\langle\lambda^{-1}\rangle$')
    add_inner_title(grid[0], 'Spectral error', loc='upper right')
    add_inner_title(grid[0], titlestr, loc='lower right')

    for tick in grid[0].get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment('center')

    im2 = grid[1].contourf(xi, yi, zi_emp, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
    grid[1].set_xlabel('$\lambda^{-1}_{N}$')
    grid[1].set_ylabel(r'$\langle\lambda^{-1}\rangle$')
    add_inner_title(grid[1], 'Empirical error', loc='upper right')
    add_inner_title(grid[1], titlestr, loc='lower right')

    for tick in grid[1].get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment('center')

    cbar = grid.cbar_axes[0].colorbar(im1)
    cbar.set_label('Error magnitude')

    plt.subplots_adjust(top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.savefig(output_path / f'panelC_{n_rotations[rot_idx]}.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # PANEL D
    fig = plt.figure(figsize=(5.75*cm, 5*cm))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
        cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
    grid[0].set_title(f"$N_{{rot}}={{{n_rotations[rot_idx]}}}$", fontsize=10)
    grid[0].set_ylabel(r"$\Delta^2$")
    grid[0].set_xlabel(r"$\lambda_N$")
    im = grid[0].scatter(
        (1/lambda_last[rot_idx]),
        deltasq[rot_idx],
        c=jnp.log(spectral_errors[rot_idx]),
        marker='.', alpha=alpha, s=2)
    cbar = grid.cbar_axes[0].colorbar(
        plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()), alpha=1.0
    )
    cbar.ax.set_title(r'$\log\varepsilon_s$', fontsize=10)
    grid[0].yaxis.set_tick_params(rotation=90)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path / f'panelD_{n_rotations[rot_idx]}.pdf')
    plt.close()

    # PANEL E
    fig = plt.figure(figsize=(5.75*cm, 5*cm))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
        cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
    grid[0].set_title(f"$N_{{rot}}={{{n_rotations[rot_idx]}}}$", fontsize=10)
    grid[0].set_ylabel(r"$1/\rho$")
    grid[0].set_xlabel(r"$\langle\lambda^{-1}\rangle$")
    im = grid[0].scatter(
        lambda_avg[rot_idx],
        1/proj_radius[rot_idx],
        c=jnp.log(spectral_errors[rot_idx]),
        marker='.', alpha=alpha, s=2)
    cbar = grid.cbar_axes[0].colorbar(
        plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()), alpha=1.0
    )
    cbar.ax.set_title(r'$\log\varepsilon_s$', fontsize=10)
    grid[0].yaxis.set_tick_params(rotation=90)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path / f'panelE_{n_rotations[rot_idx]}.pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create plots from results file')
    parser.add_argument('results_file', type=str, help='Path to the numpy results file')
    parser.add_argument('output_path', type=str, help='Directory to save the plots')
    parser.add_argument('--rot-idx', type=int, default=2,
                       help='Index of rotation to plot (default: 2)')
    parser.add_argument('--alpha', type=float, default=.1,
                       help='Transparency of scatter dots (default: .1)')
    args = parser.parse_args()

    # Load config and then results
    cfg = load_config()
    n_rotations = cfg['params']['rotations']
    results = np.load(args.results_file)

    # Create plots
    plot_panels(results, n_rotations, args.rot_idx, args.output_path, alpha=args.alpha)


if __name__ == '__main__':
    main()
