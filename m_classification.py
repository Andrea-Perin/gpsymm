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
import itertools as its
from pathlib import Path
import optax

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

ANGLES = [64, 32, 16, 8, 4, 2]
NUM_SEEDS = 16
N_TESTS = 1
REG = 1e-4
CLASSES_PER_TEST = 10  # how many classes to use per test

N_EPOCHS = 5000
BATCH_SIZE = 1024

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
def net_maker(W_std: float = 1., b_std: float = 1., dropout_rate: float = 0.5, mode: str = 'train'):
    return nt.stax.serial(
        nt.stax.Dense(128, W_std=W_std, b_std=b_std),
        nt.stax.Relu(),
        nt.stax.Dropout(rate=dropout_rate, mode=mode),
        nt.stax.Dense(64, W_std=W_std, b_std=b_std),
        nt.stax.Relu(),
        nt.stax.Dropout(rate=dropout_rate, mode=mode),
        nt.stax.Dense(10, W_std=W_std, b_std=None),
        nt.stax.Relu(),
        nt.stax.Dense(1, W_std=W_std, b_std=b_std)
    )

init_fn, train_apply_fn, kernel_fn = net_maker(W_std=1., b_std=1., dropout_rate=.5)
_, eval_apply_fn, _ = net_maker(W_std=1., b_std=1., mode='test')
kernel_fn = jax.jit(kernel_fn)
optim = optax.adam(learning_rate=5e-2)


def train_apply(params, x, key) -> Float[Array, "10"]:
    net_out = train_apply_fn(params, x, rng=key)
    return jax.nn.log_softmax(net_out)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Scalar:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


def loss(
    params, x: Float[Array, "batch 28*28"], y: Int[Array, " batch"], key: PRNGKeyArray
) -> Float[Array, ""]:
    pred_y = train_apply(params, x, key)
    return cross_entropy(y, pred_y)


def train(
    params,
    train_x,
    train_y,
    optim,
    epochs,
    key
):
    opt_state = optim.init(params)

    @jax.jit
    def make_step(
        params,
        opt_state: PyTree,
        x: Float[Array, "batch 28*28"],
        y: Int[Array, " batch"],
        key: PRNGKeyArray
    ):
        key, knew = jr.split(key)
        loss_value, grads = jax.value_and_grad(loss)(params, x, y, knew)
        updates, opt_state = optim.update( grads, opt_state, params )
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, key

    keys = jr.split(key, epochs)
    losses = []
    for epoch, key in enumerate(keys):
        # shuffle x and y
        key, pkey = jr.split(key)
        perm = jr.permutation(pkey, len(train_x))
        train_x = train_x[perm]
        train_y = train_y[perm]
        for batch_idx in range(len(train_x) // BATCH_SIZE + 1):
            params, opt_state, train_loss, key = make_step(
                params,
                opt_state,
                train_x[BATCH_SIZE*batch_idx: BATCH_SIZE*(batch_idx+1)],
                train_y[BATCH_SIZE*batch_idx: BATCH_SIZE*(batch_idx+1)],
                key
            )
            losses.append(train_loss)
    return params, losses



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
training_preds = np.empty( (len(ANGLES), N_TESTS, CLASSES_PER_TEST, NUM_SEEDS) )
losses_log = []
for ia, nangles in enumerate(ANGLES):
    angles = jnp.linspace(0, 2*jnp.pi, nangles, endpoint=False)
    RNG, test_key = jr.split(RNG)
    test_keys = jr.split(test_key, N_TESTS)

    for test_idx, key in (pbar_out := tqdm(zip(range(N_TESTS), test_keys), total=N_TESTS)):
        # pick data
        pbar_out.set_description('Picking data')
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
        pbar_out.set_description('Spectral')
        orbit_pairs = kronmap(kronmap(concat_interleave, 2), 2)(orbits, orbits)
        kernels = peelmap(kernel_fn, 4)(orbit_pairs).ntk
        ckernels = peelmap(make_circulant, 4)(kernels)
        sp_preds = peelmap(circulant_predict, 4)(ckernels[..., 0])
        avg_sp_preds = ein.reduce(sp_preds, 'clsa clsb sa sb -> clsa clsb sa', 'mean')
        # remove diagonal on first two axes (we would be comparing class a to itself)
        avg_sp_preds, ps = ein.pack( [jnp.delete(p, i, axis=0) for i, p in enumerate(avg_sp_preds)], '* clsb sa' )
        corr_avg_preds = avg_sp_preds > 0
        corr_preds = ein.reduce(corr_avg_preds, 'clsa clsb sa -> clsa sa', jnp.all)
        spectral_preds[ia, test_idx] = corr_preds

        # regression
        pbar_out.set_description('Regression')
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
        regression_preds[ia, test_idx] = jnp.array(one_vs_rest_pred).squeeze()

        # net training on sampled dataset
        pbar_out.set_description('Training')
        for cls in tqdm(range(CLASSES_PER_TEST)):
            all_ys = ein.repeat(
                jnp.roll(classes, -cls),
                'c -> (c n)', n=NUM_SEEDS*nangles
            )
            all_xs = ein.rearrange(
                jnp.roll(orbits, -cls, axis=0),
                'cls seed angle wh -> (cls seed angle) wh'
            )
            key, knet, ktrain = jr.split(key, num=3)
            idxs = jnp.arange(0, NUM_SEEDS*nangles, nangles)
            x_train = jnp.delete(all_xs, idxs, axis=0)
            y_train = jnp.delete(all_ys, idxs, axis=0)
            out_shape, params = init_fn(knet, x_train[0].shape)
            params = params[:-2]  # remove last two layers
            params, losses = train(
                params,
                x_train,
                y_train,
                optim,
                epochs=N_EPOCHS,
                key=ktrain
            )
            losses_log.append(np.array(losses))
            plt.figure()
            plt.plot(np.array(losses))
            plt.title(f'Angle: {nangles}, test: {test_idx}, class: {cls}.')
            plt.savefig(out_path / f'loss_{nangles}_{test_idx}_{cls}.pdf')
            # record prediction of trained net
            net_outs = eval_apply_fn(params, all_xs[idxs], rng=key)  # key here is only needed for API reasons
            training_preds[ia, test_idx, cls] = jnp.argmax(jnp.exp(net_outs), axis=-1)


# %%
np.save(out_path / 'regression_predictions', regression_preds)
np.save(out_path / 'spectral_predictions', spectral_preds)
np.save(out_path / 'training_predictions', training_preds)

# %% Plot
regression_preds = np.load(out_path / 'regression_predictions.npy')
spectral_preds = np.load(out_path / 'spectral_predictions.npy')
training_preds = np.load(out_path / 'training_predictions.npy')
training_preds_corr = training_preds == np.arange(10)[:, None]

# %%
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
reg_vs_angle = 1-ein.reduce((regression_preds > 0).astype(float), 'angle test cls seed -> angle test', 'mean')
sp_vs_angle = 1-ein.reduce((spectral_preds > 0).astype(float), 'angle test cls seed -> angle test', 'mean')
train_vs_angle = 1-ein.reduce(training_preds_corr.astype(float), 'angle test cls seed -> angle test', 'mean')

uncert_reg = jnp.std( jnp.mean(regression_preds > 0, axis=(1, 3)), axis=-1 ) / np.sqrt(CLASSES_PER_TEST)
uncert_sp = jnp.std( jnp.mean(regression_preds > 0, axis=(1, 3)), axis=-1 ) / np.sqrt(CLASSES_PER_TEST)
uncert_train = jnp.std( jnp.mean(training_preds_corr.astype(float), axis=(1, 3)), axis=-1 ) / np.sqrt(CLASSES_PER_TEST)

# reg_vs_angle_std = 2 * jnp.std(regression_preds > 0, axis=(-1, -2, -3)) / np.sqrt(np.array(ANGLES))
# sp_vs_angle_std = 2 * jnp.std(spectral_preds > 0, axis=(-1, -2, -3)) / np.sqrt(np.array(ANGLES))
for ax_ in ax:
    ax_.set_xscale('log')
ax[0].plot(ANGLES, reg_vs_angle, '-o')
# ax[0].fill_between(ANGLES, reg_vs_angle - uncert_reg, reg_vs_angle + uncert_reg, alpha=.2)
ax[0].set_title('regression vs angle')
ax[0].set_xlabel('angle')
ax[0].set_ylabel('error percentage')
ax[1].plot(ANGLES, sp_vs_angle, '-o')
# ax[1].fill_between(ANGLES, sp_vs_angle - uncert_sp, sp_vs_angle + uncert_sp, alpha=.2)
ax[1].set_title('spectral vs angle')
ax[1].set_xlabel('angle')
ax[1].set_ylabel('error percentage')
ax[1].set_ylim([0, None])
ax[2].plot(ANGLES, train_vs_angle, '-o')
# ax[2].fill_between(ANGLES, sp_vs_angle - uncert_sp, sp_vs_angle + uncert_sp, alpha=.2)
ax[2].set_title('train vs angle')
ax[2].set_xlabel('angle')
ax[2].set_ylabel('error percentage')
ax[2].set_ylim([0, None])
plt.show()
