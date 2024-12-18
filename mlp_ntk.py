# %% Investigations on circularizations of empirical cov
import numpy as np
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, PRNGKeyArray
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path
import argparse

from utils.conf import load_config
from utils.mnist_utils import load_images, load_labels, normalize_mnist
from utils.data_utils import make_rotation_orbit
from utils.gp_utils import kreg, circulant_error, make_circulant, extract_components


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MLP NTK analysis')
parser.add_argument('--n-hidden', type=int, default=1,
                   help='number of hidden layers (default: 1)')
args = parser.parse_args()

# %% Parameters
cfg = load_config('config.toml')
SEED = cfg['params']['seed']
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = cfg['params']['rotations']
N_PAIRS = cfg['params']['n_pairs']

# %% Paths
img_path = Path(cfg['paths']['img_path'])
lab_path = Path(cfg['paths']['lab_path'])


# %% File specific stuff
REG = 1e-10
W_std, b_std = 1., 1.
res_path = Path(cfg['paths']['res_path']) / f'mlp_{args.n_hidden}'
res_path.mkdir(parents=True, exist_ok=True)
out_path = Path(f'images/mlp_{args.n_hidden}')
out_path.mkdir(parents=True, exist_ok=True)


images = load_images(img_path=img_path)
labels = load_labels(lab_path=lab_path)
# make_orbit = kronmap(three_shear_rotate, 2)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
# network and NTK
layer = nt.stax.serial(
    nt.stax.Dense(512, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
)
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.serial(*([layer] * args.n_hidden)),
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
    angles = jnp.linspace(0, 360, n_rotations, endpoint=False)
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

np.save(res_path / 'results', results)
