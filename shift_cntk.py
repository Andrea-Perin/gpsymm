# gap cntk over shifts
import numpy as np
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, PyTree, PRNGKeyArray
from typing import Tuple
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path
import argparse

from utils.conf import load_config
from utils.mnist_utils import load_images, load_labels, normalize_mnist
from utils.data_utils import kronmap, xshift_img
from utils.gp_utils import make_circulant, circulant_error, extract_components, kreg


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run CNN NTK analysis')
parser.add_argument('--gap', action='store_true',
                    help='whether to use Global Average Pooling layer (default: False)')
parser.add_argument('--demean', action='store_true',
                    help='whether to have 0-mean inputs (default: False)')
args = parser.parse_args()


# %% Parameters
cfg = load_config('config.toml')
SEED = cfg['params']['seed']
RNG = jr.PRNGKey(SEED)
N_SHIFTS = cfg['params']['shifts']
N_PAIRS = cfg['params']['n_pairs']

# %% Paths
img_path = Path(cfg['paths']['img_path'])
lab_path = Path(cfg['paths']['lab_path'])
res_path = Path(cfg['paths']['res_path'])

# %% File-specific stuff
IS_GAP = args.gap
DEMEAN = args.demean
REG = 1e-5
W_std = 1.
b_std = 1.
res_path = res_path / ('cntk_shift_' + ('gap' if IS_GAP else 'fc') + '_' + ('demean' if DEMEAN else ''))
res_path.mkdir(parents=True, exist_ok=True)


# %%
images = load_images(img_path=img_path)
labels = load_labels(lab_path=lab_path)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
in_shape = (1, *images[0].shape, 1)
# network and NTK
conv =  nt.stax.serial(
    nt.stax.Conv(out_chan=64, filter_shape=(3, 3), padding='CIRCULAR', W_std=W_std, b_std=None),
    nt.stax.Relu()
)
pool = nt.stax.GlobalAvgPool() if IS_GAP else nt.stax.Identity()
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    conv,
    pool,
    nt.stax.Flatten(),
    nt.stax.Dense(1, W_std=W_std, b_std=None)
)
kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=1)
make_xshift_orbit = kronmap(xshift_img, 2)


def init_params(key: PRNGKeyArray, in_shape: Tuple[int]) -> Tuple[PRNGKeyArray, PyTree]:
    key, ens_key = jr.split(key)
    ens_key = jr.split(ens_key, N_PAIRS)
    out_shape, params = jax.vmap(init_fn, in_axes=(0, None))(ens_key, in_shape)
    return key, params


# %% EVERYTHING ELSE
def get_data(
    key: PRNGKeyArray,
    n_shifts: int,
    n_pairs: int = N_PAIRS,
    collision_rate: float = .2+20/N_PAIRS,
) -> Float[Array, 'pair (shift digit) width height 1']:
    n_pairs_actual = int(n_pairs * (1+collision_rate))
    key_a, key_b = jr.split(key)
    shifts = jnp.linspace(0, 1, n_shifts, endpoint=False)
    idxs_A, idxs_B = jr.randint(key, minval=0, maxval=len(images), shape=(2, n_pairs_actual,))
    # remove same-digit pairs
    labels_A, labels_B = labels[idxs_A], labels[idxs_B]
    collision_mask = (labels_A == labels_B)
    idxs_A, idxs_B = idxs_A[~collision_mask][:n_pairs], idxs_B[~collision_mask][:n_pairs]
    #
    images_A, images_B = images[idxs_A], images[idxs_B]
    orbits_A = make_xshift_orbit(images_A, shifts)
    orbits_B = make_xshift_orbit(images_B, shifts)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * shift width height')
    data = normalize_mnist(ein.rearrange(data, 'p d s w h -> (p d s) w h'))
    if DEMEAN:
        data -= jnp.mean(data, axis=(-1, -2), keepdims=True)
    return ein.rearrange(
        data,
        '(pair digit shift) width height -> pair (shift digit) width height 1',
        digit=2, angle=n_shifts, pair=len(orbits_A)
    )


@jax.jit
def get_deltasq(data: Float[Array, 'pair (shift digit) width height 1']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (shift digit) width height 1 -> digit pair shift (width height)', digit=2)
    projs = ein.reduce(rearranged, 'digit pair shift wh -> digit pair wh', 'mean')
    return ein.einsum((projs[0] - projs[1])**2, 'pair wh -> pair')


@jax.jit
def get_projected_radius(data: Float[Array, 'pair (shift digit) width height 1']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (shift digit) w h 1 -> digit pair shift (w h)', digit=2)
    centers = ein.reduce(rearranged, 'digit pair shift wh -> digit pair wh', 'mean')
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


results_names = ( 'deltasq', 'sp_errs', 'em_errs', 'lambda_last', 'lambda_avg_no_last', 'lambda_avg', 'proj_radius', 'avg_angle', 'counts')
results_shape = (len(results_names), len(N_SHIFTS), N_PAIRS)
results = np.empty(results_shape)
keys = jr.split(RNG, len(N_SHIFTS))
for idx, (n_shifts, key) in tqdm(enumerate(zip(N_SHIFTS, keys)), total=len(N_SHIFTS)):
    data = get_data(key, n_shifts)
    deltasq = get_deltasq(data)
    proj_radius = get_projected_radius(data)
    kernels = jnp.stack([kernel_fn(orb, orb).ntk for orb in data])
    # computation of empirical errors, done as average over both classes
    ys = jnp.array([+1., -1.]*n_shifts)[:, None]
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
    isp = 1/jnp.abs(jax.vmap(orthofft)(ckernels[:, 0]) + REG*jnp.sqrt(2*n_shifts))
    lambda_last = isp[:, n_shifts]
    lambda_avg_no_last = ein.reduce(jnp.delete(isp, n_shifts, axis=1), 'n d -> n', 'mean')
    lambda_avg = ein.reduce(isp, 'n d -> n', 'mean')
    # loading of results
    results[:, idx] = deltasq, spectral_errors, empirical_errors, lambda_last, lambda_avg_no_last, lambda_avg, proj_radius, avg_angle, empirical_correct_preds

np.save(res_path / 'results', results)
