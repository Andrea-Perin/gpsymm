# %% multiple seeds
import numpy as np
import jax
from jax import numpy as jnp, random as jr
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path
import argparse

from utils.conf import load_config
from utils.mnist_utils import load_images, load_labels, normalize_mnist
from utils.data_utils import kronmap, make_rotation_orbit
from utils.gp_utils import circulant_error, make_circulant
from utils.plot_utils import cm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('myplots.mlpstyle')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MLP NTK analysis with multiple seeds')
parser.add_argument('--n-hidden', type=int, default=1,
                   help='number of hidden layers (default: 1)')
parser.add_argument('--rot-idx', type=int, default=2,
    help='Select how many rotations (default: 2)')
args = parser.parse_args()


# %% Parameters
cfg = load_config('config.toml')
SEED = cfg['params']['seed']
RNG = jr.PRNGKey(SEED)
ANGLES = cfg['params']['rotations']  # [2, 4, 8, 16, 32, 64]
NUM_SEEDS = cfg['params']['n_seeds']
N_TESTS = cfg['params']['n_pts_multiple_seeds']

# %% Paths
out_path = Path(cfg['paths']['out_path']) / f'multiple_seeds_{args.n_hidden}'
out_path.mkdir(parents=True, exist_ok=True)
img_path = Path(cfg['paths']['img_path'])
lab_path = Path(cfg['paths']['lab_path'])

# %% Other params
# TEMP = 0.  # for softmax purposes, not used
REG = 1e-10
N_CLASSES = 10  # Instead of having a magic number 10 around the codebase

W_std, b_std = 1., 1.
layer = nt.stax.serial(
    nt.stax.Dense(512, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
)
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.serial(*([layer] * args.n_hidden)),  # Use args.n_hidden here
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


mycat = lambda x, y: jnp.concatenate((x, y), axis=0)
def concat_interleave(oa, ob):
    return ein.rearrange(
        jnp.concatenate((oa, ob), axis=0),
        '(digit angle) wh -> (angle digit) wh', digit=2
    )


def regress(points, nangles):
    whole_kernel = kernel_fn(points, points).ntk
    idxs = jnp.arange(0, NUM_SEEDS*nangles, nangles)
    k11 = jnp.delete( jnp.delete(whole_kernel, idxs, axis=0), idxs, axis=1 )
    k12 = jnp.delete( whole_kernel[:, idxs], idxs, axis=0, )
    k22 = whole_kernel[idxs][:, idxs]
    ys = jnp.array([+1.] * (NUM_SEEDS) * (nangles-1) + [-1.] * NUM_SEEDS * nangles)[:, None]
    sol = jax.scipy.linalg.solve(k11 + REG*jnp.eye(len(k11)), k12, assume_a='pos')
    mean = ein.einsum(sol, ys, 'train test, train d-> test d')
    # var = k22 - ein.einsum(sol, k12, 'train t1, train t2 -> t1 t2')
    return mean


images = load_images(img_path=img_path)
labels = load_labels(lab_path=lab_path)
results = np.empty((len(ANGLES), N_TESTS, 3))
colors_pairings = np.empty((len(ANGLES), N_TESTS))
colors_semaphore = np.empty((len(ANGLES), N_TESTS))
for angle_idx, nangles in enumerate(ANGLES):
    angles = jnp.linspace(0, 360, nangles, endpoint=False)
    RNG, test_key = jr.split(RNG)
    test_keys = jr.split(test_key, N_TESTS)
    for test_idx, key in tqdm(zip(range(N_TESTS), test_keys)):
        # pick data
        key, kab, ka, kb = jr.split(key, num=4)
        class_a, class_b = jnp.sort(jr.choice(kab, N_CLASSES, replace=False, shape=(2,)))
        colors_pairings[angle_idx, test_idx] = N_CLASSES*class_a+class_b  # just a way to encode pairs as colors
        idxs_a = jr.choice(ka, jnp.argwhere(labels == class_a), replace=False, shape=(NUM_SEEDS,))[:, 0]
        idxs_b = jr.choice(kb, jnp.argwhere(labels == class_b), replace=False, shape=(NUM_SEEDS,))[:, 0]

        orbits_a = make_rotation_orbit(images[idxs_a], angles)
        orbits_a = ein.rearrange(orbits_a, 'seed angle w h -> (seed angle) w h')
        orbits_a = normalize_mnist(orbits_a)
        orbits_a = ein.rearrange(orbits_a, '(seed angle) w h -> seed angle (w h)', seed=NUM_SEEDS, angle=nangles)

        orbits_b = make_rotation_orbit(images[idxs_b], angles)
        orbits_b = ein.rearrange(orbits_b, 'seed angle w h -> (seed angle) w h')
        orbits_b = normalize_mnist(orbits_b)
        orbits_b = ein.rearrange(orbits_b, '(seed angle) w h -> seed angle (w h)', seed=NUM_SEEDS, angle=nangles)

        # First with weird
        orbit_pairs = kronmap(concat_interleave, 2)(orbits_a, orbits_b)
        out = jax.vmap(jax.vmap(kernel_fn), in_axes=(1,))(orbit_pairs).ntk
        flatout = ein.rearrange(out, 'sa sb i j -> (sa sb) i j')
        k_circ_flat = jax.vmap(make_circulant)(flatout)

        # average of errors
        sp_err_flat = jax.vmap(circulant_error)(k_circ_flat)
        sp_err = ein.rearrange(sp_err_flat, '(sa sb) -> sa sb', sa=NUM_SEEDS, sb=NUM_SEEDS)
        sp_err = ein.reduce(sp_err, 'sa sb ->', 'mean')

        # Then with "normal"
        all_points, ps = ein.pack( (orbits_a, orbits_b), '* wh' )
        pred_a = regress(all_points, nangles)
        err_a = jnp.mean(1-jnp.mean(pred_a))
        all_points, ps = ein.pack( (orbits_b, orbits_a), '* wh' )
        pred_b = regress(all_points, nangles)
        err_b = jnp.mean(1-jnp.mean(pred_b))
        colors_semaphore[angle_idx, test_idx] = jnp.mean( jnp.concat((pred_a, pred_b)) > 0 )

        results[angle_idx, test_idx] = (sp_err, err_a, err_b)

# %%
figsize = (17*cm, 5*cm)
fig = plt.figure(figsize=figsize)
grid = ImageGrid(fig, 111,
                nrows_ncols=(1, len(ANGLES)),
                axes_pad=0.1,
                share_all=True,
                aspect=True,  # This forces square aspect ratio
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.15,
                )

assert (not jnp.any(jnp.isnan(results)))
maxres = jnp.max(results)
for angle_idx, ax in enumerate(grid):
    sc = ax.scatter(
        np.mean(results[angle_idx, :, 1:], axis=1),
        results[angle_idx, :, 0],
        c=colors_semaphore[angle_idx],
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        s=2
    )
    ax.plot([0, maxres], [0, maxres], '--', c='k', lw=.3)
    ax.set_xlim([0, 2])
    ax.set_xticks([0, 1, 2])
    ax.set_ylim([0, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_title(f'$N_{{rot}}={{{ANGLES[angle_idx]}}}$', fontsize=10)

# Add labels
fig.text(0.5, 0.1, 'Symm. empirical error (NTK)', ha='center', fontsize=10)
fig.text(0.0, 0.5, 'Spectral error', va='center', rotation='vertical', fontsize=10)

# Colorbar
grid.cbar_axes[0].colorbar(sc, format=lambda x, _: f'{100*x:.0f}%')
grid.cbar_axes[0].set_ylabel(r'Corr. class \%', fontsize=10)

plt.tight_layout()
plt.savefig(out_path / f'multiple_seeds_{args.n_hidden}.pdf')


# %% Single panel version
figsize = (6*cm, 5*cm)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, aspect='equal')

assert (not jnp.any(jnp.isnan(results)))
maxres = jnp.max(results)

sc = ax.scatter(
    np.mean(results[args.rot_idx, :, 1:], axis=1),
    results[args.rot_idx, :, 0],
    c=colors_semaphore[args.rot_idx],
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    s=2
)
ax.plot([0, maxres], [0, maxres], '--', c='k', lw=.3)
ax.set_xlim([0, 2])
ax.set_xticks([0, 1, 2])
ax.set_ylim([0, 2])
ax.set_yticks([0, 1, 2])
ax.set_title(f'$N_{{rot}}={{{ANGLES[args.rot_idx]}}}$', fontsize=10)

# Add labels
ax.set_xlabel('Empirical error (NTK)', fontsize=10)
ax.set_ylabel('Spectral error', fontsize=10)

# Colorbar
cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
cbar.set_label(r'Corr. class \%', fontsize=10)
labels = [0, 50, 100]
cbar.ax.set_yticklabels(labels, rotation=90)

plt.tight_layout()
plt.savefig(out_path / f'multiple_seeds_{args.n_hidden}_{args.rot_idx}.pdf')
plt.close()

# %% JUNK

# considering interactions between seeds of same class
# orbit_pairs = kronmap(concat_interleave, 2)(orbits_a, orbits_b)
# k = kernel_fn(
#     orbit_pairs,
#     ein.rearrange(orbit_pairs, 'sa sb a wh -> sb sa a wh')
# ).ntk
# kflat = ein.rearrange(k, 'sa sb sap sbp i j -> (sa sb sap sbp) i j', sa=NUM_SEEDS, sb=NUM_SEEDS, sap=NUM_SEEDS, sbp=NUM_SEEDS)
# kflatsymm = (kflat + kflat.transpose((0, 2, 1)))/2
# kcirc = jax.vmap(make_circulant)(kflatsymm)
# sp_err = jax.vmap(circulant_error)(kcirc)
# sp_err = jnp.mean(sp_err)


# k_mean = ein.reduce(k, '... i j -> i j', 'mean')
# k_circ = make_circulant(k_mean)
# sp_err = circulant_error(k_circ)

# maximizing the error
# sp_err_flat = jax.vmap(circulant_error)(k_circ_flat)
# sp_err = ein.rearrange(sp_err_flat, '(sa sb) -> sa sb', sa=NUM_SEEDS, sb=NUM_SEEDS)
# sp_err_a = ein.reduce(sp_err, 'sa sb -> sa', 'max')
# sp_err_a = ein.reduce(sp_err_a, 'sa ->', 'mean')
# sp_err_b = ein.reduce(sp_err, 'sa sb -> sb', 'max')
# sp_err_b = ein.reduce(sp_err_b, 'sb ->', 'mean')
# sp_err = (sp_err_a + sp_err_b)/2

# errors of average (weighted)
# k_circ = ein.rearrange(k_circ_flat, '(sa sb) i j -> sa sb i j', sa=NUM_SEEDS, sb=NUM_SEEDS)
# k_circ_powers = ein.rearrange(
#     jax.vmap(circulant_error)(k_circ_flat),
#     '(sa sb) -> sa sb', sa=NUM_SEEDS, sb=NUM_SEEDS
# )
# # k_circ_powers = ein.einsum(k_circ[..., 0]**2, 'sa sb i -> sa sb')
# # k_circ_powers = k_circ[..., 0, 1]**2
# a_soft = NUM_SEEDS * jax.nn.softmax(TEMP * k_circ_powers, axis=-1)[..., None, None]
# b_soft = NUM_SEEDS * jax.nn.softmax(TEMP * k_circ_powers, axis=0)[..., None, None]
# avg_over_b = ein.reduce( k_circ  * a_soft, 'sa sb i j -> i j', 'mean' )
# avg_over_a = ein.reduce( k_circ  * b_soft, 'sa sb i j -> i j', 'mean' )
# sp_err = ( circulant_error(avg_over_a) + circulant_error(avg_over_b) ) / 2

# Errors of average (unweighted)
# k_weird = ein.reduce(out, 'seeda seedb i j -> i j', 'mean')
# k_weird_circ = make_circulant(k_weird)
# sp_err = circulant_error(k_weird_circ)
