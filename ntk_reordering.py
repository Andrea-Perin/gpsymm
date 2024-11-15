# %% NTK with reordering
import numpy as np
from scipy.interpolate import griddata
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, Integer, Scalar, PyTree, Int, PRNGKeyArray, UInt8
from typing import Tuple
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path

from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, xshift_img, kronmap
from gp_utils import kreg, circulant_error, make_circulant, extract_components
from plot_utils import cm, cloudplot, add_spines

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patheffects import withStroke
plt.style.use('myplots.mlpstyle')


# %% Parameters
SEED = 124
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = [8, ]
N_PAIRS = 100
REG = 1e-4
out_path = Path('images/highd')
out_path.mkdir(parents=True, exist_ok=True)


img_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
lab_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
images = normalize_mnist(load_images(img_path=img_path))
labels = load_labels(lab_path=lab_path)
make_orbit = kronmap(three_shear_rotate, 2)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
find_min_idx = ft.partial(jnp.argmin, axis=1)
# network and NTK
W_std, b_std = 1., 1.
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(512, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


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
    orbits_B = make_orbit(images_B, angles) # + jnp.pi/n_rotations)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * angle width height')
    return ein.rearrange(data, 'pair digit angle width height -> pair (angle digit) (width height)')


def get_shifts(orbit: Float[Array, 'pair (angle digit) (w h)']) -> Integer[Array, 'pair']:
    smt = ein.rearrange(orbit, 'pair (angle digit) wh -> pair digit angle wh', digit=2)
    pairwise_dists = ein.einsum((smt[:, 0, :1] - smt[:, 1])**2, 'pair angle wh -> pair angle')
    shift_idx = jnp.argmin(pairwise_dists, axis=-1)
    return shift_idx


def reorder(data: Float[Array, 'pair (angle digit) (width height)'], shifts: Integer[Array, 'pair']
) -> Float[Array, 'pair (angle digit) (width height)']:
    #
    data = ein.rearrange(data, 'pair (angle digit) wh -> pair digit angle wh', digit=2)
    orbit_roller = ft.partial(jnp.roll, axis=0)
    orbit_1 = data[:, 0, ...]
    orbit_2 = jax.vmap(orbit_roller)(data[:, 1, ...], -shifts)
    ordered_orbits, ps = ein.pack( (orbit_1, orbit_2), "pair * angle wh" )
    return ein.rearrange(ordered_orbits, 'pair digit angle wh -> pair (angle digit) wh')


# @jax.jit
def get_symm_empirical_error(
    k: Float[Array, 'n n'],
    ys: Float[Array, 'n 1'],
    idx: int,
    reg: float=REG
) -> Float[Array, '2']:
    # for first class
    cov_components = extract_components(k, 0)
    y_train = jnp.delete(ys, 0, axis=0)
    y_pred_0, _ = kreg(*cov_components, y_train, reg=reg)
    # for second class
    cov_components = extract_components(k, 1+2*idx)
    y_train = jnp.delete(ys, 1+2*idx, axis=0)
    y_pred_1, _ = kreg(*cov_components, y_train, reg=reg)
    return jnp.concat((y_pred_0, y_pred_1))


results_names = ( 'deltasq', 'sp_errs', 'em_errs', 'lambda_last', 'lambda_avg_no_last', 'lambda_avg', 'proj_radius', 'avg_angle', 'counts')
results_shape = (len(results_names), len(N_ROTATIONS), N_PAIRS)
results = np.empty(results_shape)
keys = jr.split(RNG, len(N_ROTATIONS))
for idx, (n_rot, key) in tqdm(enumerate(zip(N_ROTATIONS, keys)), total=len(N_ROTATIONS)):
    # standard
    data = get_data(key, n_rot)
    kernels_before = jax.vmap(kernel_fn)(data, data).ntk
    # reordered
    shifts = get_shifts(data)
    ordered_data = reorder(data, shifts)
    kernels_after = jax.vmap(kernel_fn)(ordered_data, ordered_data).ntk
    #
    # computation of empirical errors, done as average over both classes
    ys = jnp.array([+1., -1.]*n_rot)[:, None]
    # empirical_results_before = jax.vmap(get_symm_empirical_error, in_axes=(0, None, 0))(kernels_before, ys, shifts)
    empirical_results_before = jnp.array(
        [get_symm_empirical_error(k, ys, s) for k, s in zip(kernels_before, shifts)]
    )
    empirical_errors_before = ein.reduce(jnp.abs(empirical_results_before-ys[:2]), 'n p 1 -> n', 'mean')
    empirical_correct_preds_before = ein.reduce( jnp.sign(empirical_results_before) == ys[:2], 'n d 1 -> n', 'sum' )
    #
    # empirical_results_after = jax.vmap(get_symm_empirical_error, in_axes=(0, None))(kernels_after, ys, jnp.zeros(N_PAIRS, dtype=int))
    empirical_results_after = jnp.array(
        [get_symm_empirical_error(k, ys, s) for k, s in zip(kernels_after, jnp.zeros_like(shifts))]
    )
    empirical_errors_after = ein.reduce(jnp.abs(empirical_results_after-ys[:2]), 'n p 1 -> n', 'mean')
    empirical_correct_preds_after = ein.reduce( jnp.sign(empirical_results_after) == ys[:2], 'n d 1 -> n', 'sum' )
    #
    # spectrals
    ckernels_before = jax.vmap(make_circulant)(kernels_before)
    spectral_errors_before = jax.vmap(circulant_error)(ckernels_before)
    ckernels_after = jax.vmap(make_circulant)(kernels_after)
    spectral_errors_after = jax.vmap(circulant_error)(ckernels_after)

    avg_angle_after = ckernels_after[:, 0, 2]
    avg_lambda_after = jnp.mean(1/jnp.abs(jax.vmap(orthofft)(ckernels_after[:, 0])+REG), axis=-1)
    break


# empirical before and after
fig, ax = plt.subplots()
ax.scatter(
    spectral_errors_before,
    spectral_errors_after
)
ax.plot([0, 1.5], [0, 1.5], ls='--')
ax.set_xlabel('empirical before')
ax.set_ylabel('empirical after')
plt.show()

fig, ax = plt.subplots()
ax.scatter(
    avg_lambda_after,
    avg_angle_after
)
ax.set_xlabel('lambda')
ax.set_ylabel('(A+B)/2')
plt.show()
