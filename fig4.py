# %% Investigations on circularizations of empirical cov
import numpy as np
from scipy.interpolate import griddata
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, Scalar, PyTree, Int, PRNGKeyArray, UInt8
from typing import Tuple
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft

from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, xshift_img, kronmap
from gp_utils import kreg, circulant_error, make_circulant, extract_components
from plot_utils import cm, cloudplot

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('myplots.mlpstyle')


# %% Parameters
SEED = 123
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = [4, 8, 16, 32, 64]
N_PAIRS = 10_000
REG = 1e-5
N_PCS = 3


img_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
lab_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
images = normalize_mnist(load_images(img_path=img_path))
labels = load_labels(lab_path=lab_path)
make_orbit = kronmap(three_shear_rotate, 2)
# Network and NTK
w_std, b_std = 1., 1.
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(512, W_std=w_std, b_std=b_std),
    nt.stax.Relu(),
    nt.stax.Dense(1, W_std=w_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


# %% PANEL A to start
label_a, label_b = 4, 7
angles = jnp.linspace(0, 1, angles_panel_a:=360, endpoint=False) * jnp.pi * 2
digit_a = images[labels == label_a][:1]
digit_b = images[labels == label_b][:1]
digit_a_orbits = make_orbit(digit_a, angles)
digit_b_orbits = make_orbit(digit_b, angles)
data, ps = ein.pack( (digit_a_orbits[0], digit_b_orbits[0]), '* n w h' )
data = ein.rearrange(data, 'd n w h -> (n d) (w h)')
u, s, vh = jnp.linalg.svd(data,  full_matrices=True)
pcs = ein.einsum(u[:, :N_PCS], s[:N_PCS], 'i j, j -> i j')

fig = plt.figure(figsize=(5*cm, 5*cm))
ax = fig.add_subplot(111, projection='3d')
# Main scatter plot
ax.scatter(pcs[::2, 0], pcs[::2, 1], pcs[::2, 2], s=2, label=label_a)
ax.scatter(pcs[1::2, 0], pcs[1::2, 1], pcs[1::2, 2], s=2, label=label_b)
ax.text2D(0.20, 0.05, "PC1", transform=ax.transAxes)
ax.text2D(0.80, 0.1, "PC2", transform=ax.transAxes)
ax.text2D(1., 0.55, "PC3", transform=ax.transAxes)
# Create two inset axes
# Parameters are [loc, width%, height%] relative to parent axes
inset1 = ax.inset_axes([0.15, 0.75, 0.15, 0.15])
inset2 = ax.inset_axes([0.60, 0.75, 0.15, 0.15])
# Plot images in insets
inset1.imshow(digit_a[0], cmap='gray')  # adjust index as needed
inset2.imshow(digit_b[0], cmap='gray')  # adjust index as needed
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
inset1.set_xticks([])
inset1.set_yticks([])
inset2.set_xticks([])
inset2.set_yticks([])
# plt.tight_layout()
plt.show()


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
    orbits_A = make_orbit(images_A, angles)
    orbits_B = make_orbit(images_B, angles + jnp.pi/n_rotations)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * angle width height')
    return ein.rearrange(data, 'pair digit angle width height -> pair (angle digit) (width height)')


@jax.jit
def get_deltasq(data: Float[Array, 'pair (angle digit) (width height)']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (angle digit) wh -> digit pair angle wh', digit=2)
    projs = ein.reduce(rearranged, 'digit pair angle wh -> digit pair wh', 'mean')
    return ein.einsum((projs[0] - projs[1])**2, 'pair wh -> pair')


@jax.jit
def get_symm_empirical_error(
    k: Float[Array, 'n n'],
    ys: Float[Array, 'n 1'],
    reg: float=REG
) -> Scalar:
    error = 0.
    # for first class
    cov_components = extract_components(k, 0)
    y_train = ys[1:]
    y_pred, _ = kreg(*cov_components, y_train, reg=reg)
    error += jnp.abs(ys[0] - y_pred).squeeze()
    # for second class
    cov_components = extract_components(k, -1)
    y_train = ys[:-1]
    y_pred, _ = kreg(*cov_components, y_train, reg=reg)
    error += jnp.abs(ys[-1] - y_pred).squeeze()
    return error / 2.


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


results_names = ( 'deltasq', 'sp_errs', 'em_errs', )
results_shape = (len(results_names), len(N_ROTATIONS), N_PAIRS)
results = np.empty(results_shape)
keys = jr.split(RNG, len(N_ROTATIONS))
for idx, (n_rot, key) in tqdm(enumerate(zip(N_ROTATIONS, keys)), total=len(N_ROTATIONS)):
    data = get_data(key, n_rot)
    deltasq = get_deltasq(data)
    kernels = jax.vmap(kernel_fn)(data, data).ntk
    # computation of empirical errors, done as average over both classes
    ys = jnp.array([+1., -1.]*n_rot)[:, None]
    empirical_errors = jax.vmap(get_symm_empirical_error, in_axes=(0, None))(kernels, ys)
    # computation of spectral errors
    ckernels = jax.vmap(make_circulant)(kernels)
    spectral_errors = jax.vmap(circulant_error)(ckernels)
    # loading of results
    results[:, idx] = deltasq, spectral_errors, empirical_errors

deltasq, spectral_errors, empirical_errors = results

# %%
fig = cloudplot(
    empirical_errors,
    spectral_errors,
    jnp.log(deltasq),
    xlabel='Empirical errors',
    ylabel='Spectral errors',
    clabel="$\log\Delta^2$",
    titles=[f"$N_{{rots}}={{{n}}}$" for n in N_ROTATIONS],
    figsize=(15*cm, 4*cm)
)
cmax = max(empirical_errors.max(), spectral_errors.max())
for ax in fig.get_axes()[:len(N_ROTATIONS)]:
    ax.plot([0, cmax], [0, cmax], color='black', alpha=1, lw=.75, ls='--')
plt.show()
