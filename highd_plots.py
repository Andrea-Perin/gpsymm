# %% This thing produces just the plots from the given data
import numpy as np
from highd import N_ROTATIONS
from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, xshift_img, kronmap
from gp_utils import kreg, circulant_error, make_circulant, extract_components
from plot_utils import cm, cloudplot, add_spines

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patheffects import withStroke
plt.style.use('myplots.mlpstyle')


# things to plot:
    # * empirical vs spectral
    # * error phase space
    # * deltasq vs lambda_N
    # * avg_angle vs lambda_avg
    #

# parameters
N_ROTATIONS = [4, 8, 16, 32, 64]
ROT_IDX = 1
# set paths
out_path = Path('images') / 'highd_trained'
data_path = Path('results') / 'highd_trained_adafactor'


# %% Load data
losses = np.load( data_path / 'losses.npy' )
theory_values = np.load( data_path / 'theory_values.npy' )
preds = np.load( data_path / 'predictions.npy' )

# %%
x = np.arange(10)
y = np.arange(10)
plt.scatter(x, y, c=x**2+y**2, cmap=mycmap)
plt.colorbar()
plt.show()
# %%





# %% Spectral vs. empirical error
semaphore = plt.color.ListedColormap(['red', 'yellow', 'green'])
fig = plt.figure(figsize=(5.75*cm, 5*cm))
grid = ImageGrid(
    fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
grid[0].set_title(f"$N_{{rot}}={{{N_ROTATIONS[ROT_IDX]}}}$", fontsize=10)
grid[0].set_xlabel("Empirical error")
grid[0].set_ylabel("Spectral error")
im = grid[0].scatter(
    empirical_errors[rot_idx],
    spectral_errors[rot_idx],
    c=emp_counts[rot_idx],
    marker='.', alpha=.1, s=2,
    cmap=semaphore, norm=norm
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
grid[0].set_xticks([0., 0.5, 1., 1.5])
grid[0].set_xlim((0, None))
grid[0].set_ylim((0, None))
plt.tight_layout(pad=0.4)
plt.savefig(out_path / f'panelB_semaphore_{N_ROTATIONS[rot_idx]}.pdf')
plt.show()
# %% PANEL C: v2
fig = plt.figure(figsize=(5.75*cm, 5*cm))
grid = ImageGrid(
    fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
# grid[0].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
grid[0].set_title(f"$N_{{rot}}={{{N_ROTATIONS[rot_idx]}}}$", fontsize=10)
grid[0].set_ylabel(r"$\Delta^2$")
grid[0].set_xlabel(r"$\lambda_N$")
im = grid[0].scatter(
    (1/lambda_last[rot_idx]),
    deltasq[rot_idx],
    c=jnp.log(spectral_errors[rot_idx]),
    marker='.', alpha=.1, s=2)
cbar = grid.cbar_axes[0].colorbar(
    plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()), alpha=1.0
)
cbar.ax.set_title(r'$\log\varepsilon_s$', fontsize=10)
grid[0].yaxis.set_tick_params(rotation=90)
plt.tight_layout(pad=0.4)
# plt.savefig(out_path / f'panelC_{N_ROTATIONS[rot_idx]}.pdf')
plt.show()
