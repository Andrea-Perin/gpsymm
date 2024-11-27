# %% Investigations on circularizations of empirical cov
import numpy as np
from scipy.interpolate import griddata
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, PRNGKeyArray
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path

from utils.mnist_utils import load_images, load_labels, normalize_mnist
from utils.data_utils import make_rotation_orbit
from utils.gp_utils import kreg, circulant_error, make_circulant, extract_components
from utils.plot_utils import cm, add_spines, semaphore

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patheffects import withStroke
plt.style.use('myplots.mlpstyle')


# %% Parameters
SEED = 124
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = [2, 4, 8, 16, 32, 64]
rot_idx = 1
N_PAIRS = 500
REG = 1e-4
N_PCS = 3
n_hidden_layers = 1
out_path = Path(f'images/mlp_{n_hidden_layers}')
out_path.mkdir(parents=True, exist_ok=True)



img_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
lab_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
images = load_images(img_path=img_path)
labels = load_labels(lab_path=lab_path)
# make_orbit = kronmap(three_shear_rotate, 2)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
# network and NTK
W_std, b_std = 1., 1.
layer = nt.stax.serial(
    nt.stax.Dense(512, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
)
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.serial(*([layer] * n_hidden_layers)),
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


# %% PANEL A to start
label_a, label_b = 4, 7
digit_selector = 10
angles = jnp.linspace(0, 360, angles_panel_a:=360, endpoint=False)
digit_a = images[labels == label_a][digit_selector:digit_selector+1]
digit_b = images[labels == label_b][:1]
digit_a_orbits = make_rotation_orbit(digit_a, angles)
digit_b_orbits = make_rotation_orbit(digit_b, angles)
digit_a_orbits = normalize_mnist(digit_a_orbits[0])
digit_b_orbits = normalize_mnist(digit_b_orbits[0])
data, ps = ein.pack( (digit_a_orbits, digit_b_orbits), '* n w h' )
data = ein.rearrange(data, 'd n w h -> (n d) (w h)')
u, s, vh = jnp.linalg.svd(data,  full_matrices=True)
pcs = ein.einsum(u[:, :N_PCS], s[:N_PCS], 'i j, j -> i j')
# Main scatter plot
fig = plt.figure(figsize=(5.75*cm, 5*cm))
ax = fig.add_subplot(111, projection='3d')
ax.set_position((0, 0, 1, 1))
step = 360//(N_ROTATIONS[rot_idx])
looped_a, ps = ein.pack((pcs[::N_ROTATIONS[rot_idx]*2], pcs[:1]), '* d')
looped_b, ps = ein.pack((pcs[1::N_ROTATIONS[rot_idx]*2], pcs[1:2]), '* d')
ax.plot(looped_a[:, 0], looped_b[:, 1], looped_a[:, 2], 'o--', lw=.3, markersize=1.5, label=label_a)
ax.plot(looped_b[:, 0], looped_b[:, 1], looped_b[:, 2], 'o--', lw=.3, markersize=1.5, label=label_b)
# show PCA basis elements
vmax = max(float(vh[:3].max()), float(-vh[:3].min()))
vmin = -vmax
inset_pc1 = ax.inset_axes((.25, 0.1, 0.15, 0.15), transform=ax.transAxes)
inset_pc1.set_title("PC1", fontsize=10, y= -.5, va='center', transform=inset_pc1.transAxes)
add_spines(inset_pc1)
inset_pc1.imshow(-vh[0].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)  # adjust index as needed

inset_pc2 = ax.inset_axes((.75, 0.1, 0.15, 0.15))
inset_pc2.set_title("PC2", fontsize=10, y=-.5, va='center', transform=inset_pc2.transAxes)
add_spines(inset_pc2)
inset_pc2.imshow(vh[1].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)  # adjust index as needed

inset_pc3 = ax.inset_axes((.9, 0.5, 0.15, 0.15))
inset_pc3.set_title("PC3", fontsize=10, y=-.5, va='center', transform=inset_pc3.transAxes,
    path_effects=[withStroke(linewidth=3, foreground='w')])
add_spines(inset_pc3)
inset_pc3.imshow(vh[2].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)  # adjust index as needed
# Show digits
inset1 = ax.inset_axes((0.3, 0.75, 0.15, 0.15))
inset1.set_axis_off()
inset1.imshow(digit_a[0], cmap='gray')
inset2 = ax.inset_axes((0.70, 0.75, 0.15, 0.15))
inset2.set_axis_off()
inset2.imshow(digit_b[0], cmap='gray')
# Plot images in insets
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout(pad=0)
plt.savefig(out_path / f'panelA_{N_ROTATIONS[rot_idx]}.pdf', bbox_inches='tight') #, pad_inches=0)
# plt.show()


# %% EVERYTHING ELSE
def get_data(
    key: PRNGKeyArray,
    n_rotations: int,
    n_pairs: int = N_PAIRS,
    collision_rate: float = .2,
) -> Float[Array, 'pair (angle digit) (width height)']:
    n_pairs_actual = int(n_pairs * (1+collision_rate))
    key_a, key_b = jr.split(key)
    angles = jnp.linspace(0, 2*jnp.pi, n_rotations, endpoint=False)
    idxs_A, idxs_B = jr.randint(key, minval=0, maxval=len(images), shape=(2, n_pairs_actual,))
    # remove same-digit pairs
    labels_A, labels_B = labels[idxs_A], labels[idxs_B]
    collision_mask = (labels_A == labels_B)
    idxs_A, idxs_B = idxs_A[~collision_mask][:n_pairs], idxs_B[~collision_mask][:n_pairs]
    #
    images_A, images_B = images[idxs_A], images[idxs_B]
    orbits_A = make_rotation_orbit(images_A, angles)
    orbits_B = make_rotation_orbit(images_B, angles)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * angle width height')
    data = normalize_mnist(ein.rearrange(data, 'pair digit angle width height -> (pair digit angle) width height'))
    return ein.rearrange(
        data,
        '(pair digit angle) width height -> pair (angle digit) (width height)',
        pair=n_pairs, digit=2, angle=n_rotations
    )


@jax.jit
def get_deltasq(data: Float[Array, 'pair (angle digit) (width height)']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (angle digit) wh -> digit pair angle wh', digit=2)
    projs = ein.reduce(rearranged, 'digit pair angle wh -> digit pair wh', 'mean')
    return ein.einsum((projs[0] - projs[1])**2, 'pair wh -> pair')


@jax.jit
def get_projected_radius(data: Float[Array, 'pair (angle digit) (width height)']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (angle digit) wh -> digit pair angle wh', digit=2)
    centers = ein.reduce(rearranged, 'digit pair angle wh -> digit pair wh', 'mean')
    radii_sq = ein.einsum((rearranged[..., 0, :] - centers)**2, 'digit pair wh -> digit pair')
    return ein.reduce(jnp.sqrt(radii_sq), 'digit pair -> pair', 'mean')


@jax.jit
def get_symm_empirical_error(
    k: Float[Array, 'n n'],
    ys: Float[Array, 'n 1'],
    reg: float=REG
) -> Float[Array, '2']:
    # for first class
    cov_components = extract_components(k, 0)
    y_train = jnp.delete(ys, 0, axis=0)
    y_pred_0, _ = kreg(*cov_components, y_train, reg=reg)
    # for second class
    cov_components = extract_components(k, 1)
    y_train = jnp.delete(ys, 1, axis=0)
    y_pred_1, _ = kreg(*cov_components, y_train, reg=reg)
    return jnp.concat((y_pred_0, y_pred_1))


def make_custom_circulant(k: Float[Array, 'n n']) -> Float[Array, 'n n']:
    """
    Convert a covariance matrix to a circulant matrix by doing weird things.
    Here we hypothesize that the matrix has the following appearance:

       | A0 | B0 | A1 | B1  ...
    ---+----+----+----+---- ...
    A0 | 1  | L  | AR |     ...
    ---+----+----+----+---- ...
    B0 |    | 1  | R  | BR  ...
    ---+----+----+----+---- ...
    ...

    In this function, we make the matrix circulant by:
        * turning the odd diagonals into their mean
        * turning the even diagonals into their min or their max

    Args:
    k (jnp.ndarray): Input covariance matrix.

    Returns:
    jnp.ndarray: Circulant matrix derived from the input covariance matrix.
    """
    idxs = jnp.arange(len(k), dtype=int)
    aligned = jax.vmap(jnp.roll)(k, -idxs)
    means = jnp.mean(aligned, axis=0)
    # mins = jnp.min(aligned, axis=0)
    # composed = jnp.where( idxs % 2, means, mins )
    maxs = jnp.max(aligned, axis=0)
    composed = jnp.where( idxs % 2, means, maxs )
    out = jax.vmap(jnp.roll, in_axes=(None, 0))(composed, idxs)
    return out


results_names = ( 'deltasq', 'sp_errs', 'em_errs', 'lambda_last', 'lambda_avg_no_last', 'lambda_avg', 'proj_radius', 'avg_angle', 'counts')
results_shape = (len(results_names), len(N_ROTATIONS), N_PAIRS)
results = np.empty(results_shape)
keys = jr.split(RNG, len(N_ROTATIONS))
for idx, (n_rot, key) in tqdm(enumerate(zip(N_ROTATIONS, keys)), total=len(N_ROTATIONS)):
    data = get_data(key, n_rot)
    deltasq = get_deltasq(data)
    proj_radius = get_projected_radius(data)
    kernels = jax.vmap(kernel_fn)(data, data).ntk
    # computation of empirical errors, done as average over both classes
    ys = jnp.array([+1., -1.]*n_rot)[:, None]
    empirical_results = jax.vmap(get_symm_empirical_error, in_axes=(0, None))(kernels, ys)
    empirical_errors = ein.reduce(jnp.abs(empirical_results-ys[:2]), 'n p 1 -> n', 'mean')
    empirical_correct_preds = ein.reduce( jnp.sign(empirical_results) == ys[:2], 'n d 1 -> n', 'sum' )
    # computation of spectral errors
    ckernels = jax.vmap(make_circulant)(kernels)
    spectral_errors = jax.vmap(circulant_error)(ckernels)
    # avg angle? (AR+BR)/2
    avg_angle = ein.reduce(ckernels[:, 0, 2::2], 'n k -> n', 'mean')
    # computation of elements of the inverse spectrum. NOTE THE REGULARIZATION
    # CONSTANT: see https://numpy.org/doc/stable/reference/routines.fft.html#normalization
    # isp = 1/jnp.abs(jax.vmap(orthofft)(ckernels[:, 0]) + REG*jnp.sqrt(2*n_rot))
    isp = 1/jnp.abs(jax.vmap(jnp.fft.fft)(ckernels[:, 0]))
    lambda_last = isp[:, n_rot]
    lambda_avg_no_last = ein.reduce(jnp.delete(isp, n_rot, axis=1), 'n d -> n', 'mean')
    lambda_avg = ein.reduce(isp, 'n d -> n', 'mean')
    # loading of results
    results[:, idx] = deltasq, spectral_errors, empirical_errors, lambda_last, lambda_avg_no_last, lambda_avg, proj_radius, avg_angle, empirical_correct_preds

deltasq, spectral_errors, empirical_errors, lambda_last, lambda_avg_no_last, lambda_avg, proj_radius, avg_angle, emp_counts = results
# additional division of lambda_last by sqrt(N)
# This makes relationships "more linear" in the following plots
# We are still unsure of the reason for this behavior
lambda_last *= jnp.sqrt(jnp.array(N_ROTATIONS)[:, None])

# %% PANEL C
ll = lambda_last[rot_idx]
lnl = lambda_avg[rot_idx]
spec_errs = spectral_errors[rot_idx]
emp_errs = empirical_errors[rot_idx]

# Create interpolation grids
xi = np.linspace(ll.min(), ll.max(), 100)
yi = np.linspace(lnl.min(), lnl.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate both errors
zi_spec = griddata((ll, lnl), spec_errs, (xi, yi), method='linear')
zi_emp = griddata((ll, lnl), emp_errs, (xi, yi), method='linear')


def add_inner_title(ax, title, loc, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
                size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    return at

# Create figure with ImageGrid
titlestr = f'$N_{{rot}}={N_ROTATIONS[rot_idx]}$'
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
vmin = min(np.nanmin(zi_spec), np.nanmin(zi_emp))
vmax = max(np.nanmax(zi_spec), np.nanmax(zi_emp))
im1 = grid[0].contourf(xi, yi, zi_spec, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
grid[0].set_xlabel(r'$\lambda^{-1}_{N}$')
grid[0].set_ylabel(r'$\langle\lambda^{-1}\rangle$')
add_inner_title(grid[0], 'Spectral error', loc='upper right')
add_inner_title(grid[0], titlestr, loc='lower right')
# grid[0].set_title('Spectral error', fontsize=10)
for tick in grid[0].get_yticklabels():
    tick.set_rotation(90)
    tick.set_verticalalignment('center')
# Plot empirical errors
im2 = grid[1].contourf(xi, yi, zi_emp, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
grid[1].set_xlabel('$\lambda^{-1}_{N}$')
grid[1].set_ylabel(r'$\langle\lambda^{-1}\rangle$')
add_inner_title(grid[1], 'Empirical error', loc='upper right')
add_inner_title(grid[1], titlestr, loc='lower right')
# grid[1].set_title('Empirical error', fontsize=10)
for tick in grid[1].get_yticklabels():
    tick.set_rotation(90)
    tick.set_verticalalignment('center')
# Add colorbar
cbar = grid.cbar_axes[0].colorbar(im1)
cbar.set_label('Error magnitude')
# Adjust the layout manually
# plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.85)
plt.subplots_adjust(top=1, bottom=0, wspace=0, hspace=0)
plt.tight_layout(pad=0)
plt.savefig(out_path / f'panelC_{N_ROTATIONS[rot_idx]}.pdf', bbox_inches='tight', pad_inches=0)
# plt.show()

# %% PANEL B: v2
fig = plt.figure(figsize=(5.75*cm, 5*cm))
grid = ImageGrid(
    fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
grid[0].set_title(f"$N_{{rot}}={{{N_ROTATIONS[rot_idx]}}}$", fontsize=10)
grid[0].set_xlabel("Empirical error (NTK)")
grid[0].set_ylabel("Spectral error")
im = grid[0].scatter(
    empirical_errors[rot_idx],
    spectral_errors[rot_idx],
    c=emp_counts[rot_idx],
    marker='.', alpha=.1, s=2,
    # cmap=semaphore
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
# grid[0].set_xticks([0., 0.5, 1., 1.5])
grid[0].set_xlim((0, None))
grid[0].set_ylim((0, None))
plt.tight_layout(pad=0.4)
plt.savefig(out_path / f'panelB_{N_ROTATIONS[rot_idx]}.pdf')
# plt.show()
# %% PANEL D
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
plt.savefig(out_path / f'panelD_{N_ROTATIONS[rot_idx]}.pdf')
# plt.show()

# %% PANEL E
fig = plt.figure(figsize=(5.75*cm, 5*cm))
grid = ImageGrid(
    fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
grid[0].set_title(f"$N_{{rot}}={{{N_ROTATIONS[rot_idx]}}}$", fontsize=10)
grid[0].set_ylabel(r"$1/\rho$")
grid[0].set_xlabel(r"$\langle\lambda^{-1}\rangle$")
im = grid[0].scatter(
    lambda_avg[rot_idx],
    1/proj_radius[rot_idx],
    c=jnp.log(spectral_errors[rot_idx]),
    marker='.', alpha=.1, s=2)
cbar = grid.cbar_axes[0].colorbar(
    plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()), alpha=1.0
)
cbar.ax.set_title(r'$\log\varepsilon_s$', fontsize=10)
grid[0].yaxis.set_tick_params(rotation=90)
plt.tight_layout(pad=0.4)
plt.savefig(out_path / f'panelE_{N_ROTATIONS[rot_idx]}.pdf')


# %% Linear regression over spectral vs empirical
ys = spectral_errors[rot_idx]
xs = empirical_errors[rot_idx]
common_mask = jnp.isnan(ys) | jnp.isnan(xs)
ys = ys[~common_mask]
xs = xs[~common_mask]
alpha = ((ys.sum()) * (xs**2).sum() - xs.sum() * (xs*ys).sum()) / ((N_PAIRS * (xs**2).sum()) - (xs.sum())**2)
beta = (N_PAIRS * (xs*ys).sum() - (xs.sum() * ys.sum())) / (N_PAIRS * (xs**2).sum() - (xs.sum())**2)
print(alpha)
print(beta)
# %%

# %%
# plt.show()
# # %% PANEL D: v2
# fig = plt.figure(figsize=(6*cm, 5*cm))
# grid = ImageGrid(
#     fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, label_mode="L", share_all=True,
#     cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
# grid[0].set_title(f"$N_{{rot}}={{{N_ROTATIONS[rot_idx]}}}$", fontsize=10)
# grid[0].set_xlabel(r"$1/\rho$")
# grid[0].set_ylabel(r"$\langle\lambda^{-1}\rangle$")
# im = grid[0].scatter(
#     (1/lambda_last)[rot_idx],
#     proj_radius[rot_idx],
#     c=jnp.log(spectral_errors[rot_idx]),
#     marker='.', alpha=.1, s=2)
# cbar = grid.cbar_axes[0].colorbar(
#     plt.cm.ScalarMappable(norm=im.norm, cmap=im.get_cmap()), alpha=1.0
# )
# cbar.ax.set_title(r'$\log\varepsilon_s$', fontsize=10)
# grid[0].yaxis.set_tick_params(rotation=90)
# plt.tight_layout(pad=0.4)
# # plt.savefig(out_path / 'panelD_v2.pdf')
# plt.show()

# # %% PANEL B
# fig = plt.figure(figsize=(12*cm, 5*cm))
# grid = ImageGrid(
#     fig, 111, nrows_ncols=(1, 2), axes_pad=0.05, label_mode="L", share_all=True, aspect=False,
#     cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
# # Empirical errors
# jittered_n = ein.repeat(jnp.array(N_ROTATIONS), 'n -> n p', p=N_PAIRS)
# jittered_n += jr.normal(key=RNG, shape=jittered_n.shape) * .5
# scatter1 = grid[0].scatter(jittered_n, empirical_errors, s=1, alpha=.1, c=jnp.log(deltasq))
# grid[0].set_xticks(N_ROTATIONS)
# grid[0].set_title("Empirical error", fontsize=10)
# grid[0].set_xlabel("$N_{rot}$",)
# grid[0].set_ylabel("Error magnitude")
# # Spectral errors
# scatter2 = grid[1].scatter(jittered_n, spectral_errors, s=1, alpha=.1, c=jnp.log(deltasq))
# grid[1].set_xticks(N_ROTATIONS)
# grid[1].set_xlabel("$N_{rot}$",)
# grid[1].set_title("Spectral error", fontsize=10)
# # Shared xlabel
# # Add colorbar
# cbar = grid.cbar_axes[0].colorbar(
#     plt.cm.ScalarMappable(norm=scatter2.norm, cmap=scatter2.get_cmap()),
#     alpha=1.0
# )
# cbar.set_label('$\log\Delta^2$')
# plt.set_cmap('viridis')
# plt.tight_layout()
# # plt.savefig(out_path / 'panelB_v1.pdf')
# plt.show()

# # %% PANEL C
# fig = cloudplot(
#     empirical_errors,
#     spectral_errors,
#     jnp.log(deltasq),
#     xlabel='Empirical error',
#     ylabel='Spectral error',
#     clabel="$\log\Delta^2$",
#     titles=[f"$N_{{rots}}={{{n}}}$" for n in N_ROTATIONS],
#     figsize=(17*cm, 5*cm)
# )
# cmax = max(empirical_errors.max(), spectral_errors.max())
# for ax in fig.get_axes()[:len(N_ROTATIONS)]:
#     ax.plot([0, cmax], [0, cmax], color='black', alpha=1, lw=.75, ls='--')
# fig.supxlabel('Empirical error', y=.12, fontsize=10)
# fig.supylabel('Spectral error', x=.05, y=0.59, fontsize=10)
# plt.tight_layout()
# # plt.savefig(out_path / 'panelC_v1.pdf')
# plt.show()

# # %% PANEL D
# fig = cloudplot(
#     deltasq,
#     (1/lambda_last),
#     jnp.log(spectral_errors),
#     xlabel=(XLAB:='$\Delta^2$'),
#     ylabel=(YLAB:='$\lambda_N$'),
#     clabel=r"log(spectral err.)",
#     titles=[f"$N_{{rots}}={{{n}}}$" for n in N_ROTATIONS],
#     figsize=(17*cm, 5*cm)
# )
# fig.supxlabel(XLAB, y=.15)
# fig.supylabel(YLAB, x=0.03, y=.55)
# plt.tight_layout()
# # plt.savefig(out_path / 'panelD_v1.pdf')
# plt.show()
