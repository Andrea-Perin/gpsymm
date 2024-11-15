# %% multiple seeds
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
NUM_ANGLES = 8
NUM_SEEDS = 16
N_TESTS = 1024
REG = 1e-4
N_IMGS = 60_000
N_CLASSES = 10
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


def regress(points):
    whole_kernel = kernel_fn(points, points).ntk
    idxs = jnp.arange(0, NUM_SEEDS*NUM_ANGLES, NUM_ANGLES)
    k11 = jnp.delete( jnp.delete(whole_kernel, idxs, axis=0), idxs, axis=1 )
    k12 = jnp.delete( whole_kernel[:, idxs], idxs, axis=0, )
    k22 = whole_kernel[idxs][:, idxs]
    ys = jnp.array([+1.] * (NUM_SEEDS) * (NUM_ANGLES-1) + [-1.] * NUM_SEEDS * NUM_ANGLES)[:, None]
    sol = jax.scipy.linalg.solve(k11 + REG*jnp.eye(len(k11)), k12, assume_a='pos')
    mean = ein.einsum(sol, ys, 'train test, train d-> test d')
    var = k22 - ein.einsum(sol, k12, 'train t1, train t2 -> t1 t2')
    return jnp.mean(jnp.abs(1 -mean))


angles = jnp.linspace(0, 2*jnp.pi, NUM_ANGLES, endpoint=False)
all_orbits = make_orbit(images, angles)

results = np.empty( (N_TESTS, 3) )
colors = []
RNG, test_key = jr.split(RNG)
test_keys = jr.split(test_key, N_TESTS)
for iteration, key in tqdm(zip(range(N_TESTS), test_keys)):
    # pick data
    key, kab, ka, kb = jr.split(key, num=4)
    class_a, class_b = jnp.sort(jr.choice(kab, N_CLASSES, replace=False, shape=(2,)))
    colors.append( (class_a, class_b) )
    idxs_a = jr.choice(ka, jnp.argwhere(labels == class_a), replace=False, shape=(NUM_SEEDS,)).squeeze()
    idxs_b = jr.choice(kb, jnp.argwhere(labels == class_b), replace=False, shape=(NUM_SEEDS,)).squeeze()
    orbits_a = ein.rearrange(all_orbits[idxs_a], 'seed angle w h -> seed angle (w h)')
    orbits_b = ein.rearrange(all_orbits[idxs_b], 'seed angle w h -> seed angle (w h)')

    # First with weird
    orbit_pairs = kronmap(concat_interleave, 2)(orbits_a, orbits_b)
    out = jax.vmap(jax.vmap(kernel_fn), in_axes=(1,))(orbit_pairs).ntk
    k_weird = ein.reduce(out, 'seeda seedb i j -> i j', 'mean')
    k_weird_circ = make_circulant(k_weird)
    k_weird_err = circulant_error(k_weird_circ)

    # Then with "normal"
    all_points, ps = ein.pack( (orbits_a, orbits_b), '* wh' )
    err_a = regress(all_points)
    all_points, ps = ein.pack( (orbits_b, orbits_a), '* wh' )
    err_b = regress(all_points)

    results[iteration] = (k_weird_err, err_a, err_b)

# %%
colors = jnp.array([a*10+b for a, b in colors])
fig, ax = plt.subplots()
ax.scatter( results[:, 0], jnp.mean(results[:, 1:], axis=1), c=colors )
ax.set_xlabel('spectral')
ax.set_ylabel('regression')
ax.plot([0, 1.5], [0, 1.5], ls='--')
plt.show()
# %%
fig, ax = plt.subplots()
ax.scatter( results[:, 0], results[:, 1], c=colors )
ax.set_xlabel('spectral')
ax.set_ylabel('regression')
ax.plot([0, 1.5], [0, 1.5], ls='--')
plt.show()

fig, ax = plt.subplots()
ax.scatter( results[:, 0], results[:, 2], c=colors )
ax.set_xlabel('spectral')
ax.set_ylabel('regression')
ax.plot([0, 1.5], [0, 1.5], ls='--')
plt.show()
