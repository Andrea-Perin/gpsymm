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

ANGLES = [8, 4, 2]
NUM_SEEDS = 13
N_TESTS = 300
REG = 1e-4
CLASSES_PER_TEST = 10  # how many classes to use per test

N_IMGS = 60_000
N_CLASSES = 10  # how many classes in MNIST
out_path = Path('results/m_classification')
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


spectral_preds = np.empty( (len(ANGLES), N_TESTS, CLASSES_PER_TEST, NUM_SEEDS) )
regression_preds = np.empty( (len(ANGLES), N_TESTS, CLASSES_PER_TEST, NUM_SEEDS) )
for ia, nangles in enumerate(ANGLES):
    angles = jnp.linspace(0, 2*jnp.pi, nangles, endpoint=False)
    RNG, test_key = jr.split(RNG)
    test_keys = jr.split(test_key, N_TESTS)

    for iteration, key in tqdm(zip(range(N_TESTS), test_keys), total=N_TESTS):
        # pick data
        key, kab = jr.split(key)
        classes = jr.choice(kab, N_CLASSES, replace=False, shape=(CLASSES_PER_TEST, ))
        classes = jnp.sort(classes)
        key, k_classes = jr.split(key)
        k_classes = jr.split(key, CLASSES_PER_TEST)
        idxs = [jr.choice(k, jnp.argwhere(labels == c), replace=False, shape=(NUM_SEEDS,)) for k, c in zip(k_classes, classes)]
        flat_idxs = ein.rearrange(jnp.array(idxs), 'cls seed 1 -> (cls seed)')
        orbits = make_orbit(images[flat_idxs], angles)
        orbits = ein.rearrange( orbits, '(cls seed) angle w h -> cls seed angle (w h)', cls=CLASSES_PER_TEST, seed=NUM_SEEDS )

        # spectral
        orbit_pairs = kronmap(kronmap(concat_interleave, 2), 2)(orbits, orbits)
        kernels = peelmap(kernel_fn, 4)(orbit_pairs).ntk
        ckernels = peelmap(make_circulant, 4)(kernels)
        sp_preds = peelmap(circulant_predict, 4)(ckernels[..., 0])
        avg_sp_preds = ein.reduce(sp_preds, 'clsa clsb sa sb -> clsa clsb sa', 'mean')
        # remove diagonal on first two axes (we would be comparing class a to itself)
        avg_sp_preds, ps = ein.pack( [jnp.delete(p, i, axis=0) for i, p in enumerate(avg_sp_preds)], '* clsb sa' )
        corr_avg_preds = avg_sp_preds > 0
        corr_preds = ein.reduce(corr_avg_preds, 'clsa clsb sa -> clsa sa', jnp.all)
        spectral_preds[ia, iteration] = corr_preds

        # regression
        one_vs_rest_pred = []
        for cls in range(CLASSES_PER_TEST):
            all_points = ein.rearrange(
                jnp.roll(orbits, -cls, axis=0),
                'cls seed angle wh -> (cls seed angle) wh'
            )
            whole_kernel = kernel_fn(all_points, all_points).ntk
            # LINEAR REGRESSION (on unrotated samples from class A)
            # we are removing the first angle from JUST class A
            idxs = jnp.arange(0, NUM_SEEDS*nangles, nangles)
            k11 = jnp.delete( jnp.delete(whole_kernel, idxs, axis=0), idxs, axis=1 )
            k12 = jnp.delete( whole_kernel[:, idxs], idxs, axis=0, )
            k22 = whole_kernel[idxs][:, idxs]
            ys = jnp.array([+1.] * (NUM_SEEDS) * (nangles-1) + [-1.] * NUM_SEEDS * nangles * (CLASSES_PER_TEST - 1))[:, None]
            sol = jax.scipy.linalg.solve(k11 + REG*jnp.eye(len(k11)), k12, assume_a='pos')
            pred = ein.einsum(sol, ys, 'train test, train d -> test d')
            one_vs_rest_pred.append(pred)
        regression_preds[ia, iteration] = jnp.array(one_vs_rest_pred).squeeze()

np.save(out_path / 'regression_predictions', regression_preds)
np.save(out_path / 'spectral_predictions', spectral_preds)
