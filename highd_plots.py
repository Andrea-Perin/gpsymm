# %% This thing produces just the plots from the given data
import numpy as np
from scipy.interpolate import griddata
import einops as ein
from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, xshift_img, kronmap
from gp_utils import kreg, circulant_error, make_circulant, extract_components
from plot_utils import cm, cloudplot, add_spines, semaphore, add_inner_title

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
data_path = Path('results') / 'highd_trained_3layers'

# %% Load data
losses = np.load( data_path / 'losses.npy' )[ROT_IDX]

preds = np.load( data_path / 'predictions.npy' )[ROT_IDX]
correct_p = preds[..., 0] > 0
correct_m = preds[..., 1] < 0
corr_count = np.sum( np.column_stack([correct_p, correct_m]), axis=1 )
emp_err = np.mean( np.abs(preds - np.array([+1., -1.])), axis=-1 )

theory_values = ein.rearrange(
    np.load( data_path / 'theory_values.npy' )[ROT_IDX],
    'pair value -> value pair'
)
sp_err, lambda_n, lambda_avg, deltasq, avg_angle = theory_values


# %% Spectral vs. empirical error (sempaphore plot)
fig = plt.figure(figsize=(5.75*cm, 5*cm))
grid = ImageGrid(
    fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
grid[0].set_title(f"$N_{{rot}}={{{N_ROTATIONS[ROT_IDX]}}}$", fontsize=10)
grid[0].set_xlabel("Empirical error")
grid[0].set_ylabel("Spectral error")
im = grid[0].scatter( emp_err, sp_err, c=corr_count, marker='.', alpha=.5, s=2, cmap=semaphore )
cbar = grid.cbar_axes[0].colorbar(
    plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()),
    label='Num. correct',
    ticks=[0.5, 1.5, 2.5],
    alpha=1.0
)
cbar.ax.tick_params(size=0)
cbar.ax.set_yticklabels((0, 1, 2))

cmax = max(emp_err.max(), sp_err.max())
grid[0].plot([0, cmax], [0, cmax], color='black', alpha=1, lw=.75, ls='--')
grid[0].set_xticks([0., 0.5, 1., 1.5])
grid[0].set_xlim((0, None))
grid[0].set_ylim((0, None))
plt.tight_layout(pad=0.4)
# plt.savefig(out_path / f'panelB_semaphore_{N_ROTATIONS[ROT_IDX]}.pdf')
plt.show()


# %% Error phase space
def add_inner_title(ax, title, loc, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=4)],
                size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    return ax


# Create interpolation grids
xi = np.linspace(lambda_n.min(), lambda_n.max(), 100)
yi = np.linspace(lambda_avg.min(), lambda_avg.max(), 100)
xi, yi = np.meshgrid(xi, yi)
# Interpolate both errors
zi_sp = griddata((lambda_n, lambda_avg), sp_err, (xi, yi), method='linear')
zi_emp = griddata((lambda_n, lambda_avg), emp_err, (xi, yi), method='linear')
# Create figure with ImageGrid
titlestr = f'$N_{{rot}}={N_ROTATIONS[ROT_IDX]}$'
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
# Plot spectral errors
vmin = min(np.nanmin(zi_sp), np.nanmin(zi_emp))
vmax = max(np.nanmax(zi_sp), np.nanmax(zi_emp))
im1 = grid[0].contourf(xi, yi, zi_sp, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
grid[0].set_xlabel(r'$\lambda^{-1}_{N}$')
grid[0].set_ylabel(r'$\langle\lambda^{-1}\rangle$')
add_inner_title(grid[0], 'Spectral error', loc='upper right')
add_inner_title(grid[0], titlestr, loc='lower right')
for tick in grid[0].get_yticklabels():
    tick.set_rotation(90)
    tick.set_verticalalignment('center')
# Plot empirical errors
im2 = grid[1].contourf(xi, yi, zi_emp, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
grid[1].set_xlabel('$\lambda^{-1}_{N}$')
grid[1].set_ylabel(r'$\langle\lambda^{-1}\rangle$')
add_inner_title(grid[1], 'Empirical error', loc='upper right')
add_inner_title(grid[1], titlestr, loc='lower right')
for tick in grid[1].get_yticklabels():
    tick.set_rotation(90)
    tick.set_verticalalignment('center')
# Add colorbar
cbar = grid.cbar_axes[0].colorbar(im2)
cbar.set_label('Error magnitude')
# Adjust the layout manually
# plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.85)
plt.subplots_adjust(top=1, bottom=0, wspace=0, hspace=0)
plt.tight_layout(pad=0)
plt.show()

# %%
