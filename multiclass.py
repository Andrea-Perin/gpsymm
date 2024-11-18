# %% Multiclass stuff
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
from gp_utils import kreg, circulant_error, make_circulant, extract_components, circulant_predict
from plot_utils import cm, cloudplot, add_spines

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patheffects import withStroke
plt.style.use('myplots.mlpstyle')


# %% Parameters
SEED = 12
RNG = jr.PRNGKey(SEED)
NUM_ANGLES = 8
NUM_SEEDS = 10
N_TESTS = 300
REG = 1e-4
CLASSES_PER_TEST = 3  # how many classes to use per test

N_IMGS = 60_000
N_CLASSES = 10  # how many classes in MNIST
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
    nt.stax.Dense(512, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


mycat = lambda x, y: jnp.concatenate((x, y), axis=0)
def concat_interleave(oa, ob):
    return ein.rearrange(
        jnp.concatenate((oa, ob), axis=0),
        '(digit angle) wh -> (angle digit) wh', digit=2
    )


def peelmap(fn, num):
    for n in range(num):
        fn = jax.vmap(fn)
    return fn


def k_one_vs_many(orbits, idx):
    orbit_a = orbits[idx]
    orbit_b, ps = ein.pack((orbits[:idx], orbits[idx+1:]), '* seed angle wh')
    orbit_pairs = jax.vmap(kronmap(concat_interleave, 2), in_axes=(None, 0))(orbit_a, orbit_b)
    out = peelmap(kernel_fn, 3)(orbit_pairs).ntk
    return ein.rearrange(out, 'seedb seeda cls i j -> cls seeda seedb i j')


angles = jnp.linspace(0, 2*jnp.pi, NUM_ANGLES, endpoint=False)
# all_orbits = make_orbit(images, angles)
results = np.empty( (N_TESTS, 2*CLASSES_PER_TEST) )
RNG, test_key = jr.split(RNG)
test_keys = jr.split(test_key, N_TESTS)

for iteration, key in tqdm(zip(range(N_TESTS), test_keys)):
    # pick data
    key, kab = jr.split(key)
    classes = jr.choice(kab, N_CLASSES, replace=False, shape=(CLASSES_PER_TEST, ))
    key, k_classes = jr.split(key)
    k_classes = jr.split(key, CLASSES_PER_TEST)
    idxs = [jr.choice(k, jnp.argwhere(labels == c), replace=False, shape=(NUM_SEEDS,)) for k, c in zip(k_classes, classes)]
    flat_idxs = ein.rearrange(jnp.array(idxs), 'cls seed 1 -> (cls seed)')
    orbits = make_orbit(images[flat_idxs], angles)
    orbits = ein.rearrange( orbits, '(cls seed) angle w h -> cls seed angle (w h)',
        cls=CLASSES_PER_TEST, seed=NUM_SEEDS )
    # now we compute class vs class kernel matrices
    orbit_pairs = kronmap(kronmap(concat_interleave, 2), 2)(orbits, orbits)
    kernels = peelmap(kernel_fn, 4)(orbit_pairs).ntk
    avg_k = ein.reduce(kernels, 'ca cb sa sb i j -> ca cb i j', 'mean')
    avg_kc = peelmap(make_circulant, 2)(avg_k)
    preds = peelmap(circulant_predict, 4)(kernels[..., 0])
    # remove the second diagonal
    preds, ps = ein.pack( [jnp.delete(p, i, axis=0) for i, p in enumerate(preds)], '* cb sa sb' )
    avg_preds = ein.reduce(preds, 'ca cb sa sb -> ca sa sb', 'mean')



    # select one of the classes; use that as reference, and all the others as "rest of the world"
    ks = jnp.array(
        [k_one_vs_many(orbits, c) for c in range(CLASSES_PER_TEST)]
    )
    # candidate_ks = ein.reduce(ks, 'refcls othercls seeda seedb i j -> refcls othercls i j', 'mean')
    # ksc = jax.vmap(jax.vmap(make_circulant), in_axes=(1,))(candidate_ks)
    # sp_err = jax.vmap(jax.vmap(circulant_error), in_axes=(1,))(ksc)
    # sp_err = ein.reduce(sp_err, 'refcls othercls -> refcls', 'max')
    # sp_err = jax.vmap(jax.vmap(jax.vmap(circulant_error, in_axes=(0,)), in_axes=(1,)), in_axes=(2,))(ks)
    # sp_err = ein.reduce(sp_err, 'refcls othercls -> refcls', 'max')

    # flat_k = ein.reduce(ks, 'r o sa sb i j -> (r o) i j', 'mean')
    # flat_kc = jax.vmap(make_circulant)(flat_k)
    # sp_err = jax.vmap(circulant_error)(flat_kc)
    # # sp_err = ein.rearrange(sp_err, '(r o sa sb) -> r o sa sb',
    # #     r=CLASSES_PER_TEST, o=CLASSES_PER_TEST-1, sa=NUM_SEEDS, sb=NUM_SEEDS
    # # )
    # sp_err = ein.rearrange(sp_err, '(r o) -> r o', r=CLASSES_PER_TEST, o=CLASSES_PER_TEST-1)
    # # sp_err = ein.reduce(sp_err, 'r o sa sb -> r o sa', 'mean')
    # sp_err = ein.reduce(sp_err, 'r o -> r', 'max')
    # # sp_err = ein.reduce(sp_err, 'r o -> r', 'mean')

    flat_k = ein.rearrange(ks, 'r o sa sb i j -> (r o sa sb) i j')
    flat_kc = jax.vmap(make_circulant)(flat_k)
    avg_kc = ein.reduce(flat_kc, '(r o sa sb) i j -> r i j', 'mean', o=CLASSES_PER_TEST-1, sa=NUM_SEEDS, sb=NUM_SEEDS)
    breakpoint()
    sp_pred = jax.vmap(circulant_predict)(avg_kc[..., 0])


    # sp_err = jax.vmap(circulant_error)(flat_kc)
    # sp_err = ein.rearrange(sp_err, '(r o sa sb) -> r o sa sb',
    #     r=CLASSES_PER_TEST, o=CLASSES_PER_TEST-1, sa=NUM_SEEDS, sb=NUM_SEEDS
    # )
    # # sp_err = ein.rearrange(sp_err, '(r o) -> r o', r=CLASSES_PER_TEST, o=CLASSES_PER_TEST-1)
    # sp_err = ein.reduce(sp_err, 'r o sa sb -> r o sa', 'mean')
    # sp_err = ein.reduce(sp_err, 'r o sa -> r o', 'mean')
    # sp_err = ein.reduce(sp_err, 'r o -> r', 'mean')


    # solve regression
    one_vs_rest_pred = []
    for cls in range(CLASSES_PER_TEST):
        all_points = ein.rearrange(
            jnp.roll(orbits, -cls, axis=0),
            'cls seed angle wh -> (cls seed angle) wh'
        )
        whole_kernel = kernel_fn(all_points, all_points).ntk
        # LINEAR REGRESSION (on unrotated samples from class A)
        # we are removing the first angle from JUST class A
        idxs = jnp.arange(0, NUM_SEEDS*NUM_ANGLES, NUM_ANGLES)
        k11 = jnp.delete( jnp.delete(whole_kernel, idxs, axis=0), idxs, axis=1 )
        k12 = jnp.delete( whole_kernel[:, idxs], idxs, axis=0, )
        k22 = whole_kernel[idxs][:, idxs]
        ys = jnp.array([+1.] * (NUM_SEEDS) * (NUM_ANGLES-1) + [-1.] * NUM_SEEDS * NUM_ANGLES * (CLASSES_PER_TEST - 1))[:, None]
        sol = jax.scipy.linalg.solve(k11 + REG*jnp.eye(len(k11)), k12, assume_a='pos')
        pred = ein.einsum(sol, ys, 'train test, train d -> test d')
        breakpoint()
        one_vs_rest_pred.append(pred)

    # log results
    results[iteration] = (*sp_pred.tolist(), *one_vs_rest_pred)


# %%
fig, axs = plt.subplots(nrows=1, ncols=CLASSES_PER_TEST, figsize=(4, 4), sharex=True, sharey=True)
fig.supxlabel('spectral')
fig.supylabel('empirical')
for idx, ax in enumerate(axs.flatten()):
    ax.scatter(results[:, idx], results[:, CLASSES_PER_TEST+idx], alpha=.1)
    ax.plot([0, 1.5], [0, 1.5], ls='--', color='k')
# plt.savefig('multiclass_plot_1.png')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(results[:, :CLASSES_PER_TEST].mean(1), results[:, CLASSES_PER_TEST:].mean(1), alpha=.1)
ax.plot([0, 1.5], [0, 1.5], ls='--', color='k')
plt.savefig('multiclass_plot_2.png')
plt.show()
