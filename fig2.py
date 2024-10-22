# %%
import jax
from jax import numpy as jnp
import einops as ein
from gp_utils import kreg

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use('./myplots.mlpstyle')
from plot_utils import cm, get_size


# %% plot A: gaussian process on periodic grid
N = 11
L = N/ 4
xs = jnp.linspace(-1, 1, N)[:, None]
ys = jnp.array([(-1)**(i) for i in range(N)])[:, None]
k = jnp.exp(-L**2 * jnp.sum((xs - xs[:, None])**2, axis=-1))
k_train_train = jnp.delete(jnp.delete(k, N//2, axis=0), N//2, axis=1)
k_train_test = jnp.delete(k[:, N//2:N//2+1], N//2, axis=0)
k_test_test = k[N//2:N//2+1, N//2:N//2+1]
y_train = jnp.delete(ys, N//2, axis=0)
y_mean, y_var = kreg(k_train_train, k_train_test, k_test_test, y_train)
error = jnp.abs(ys[N//2] - y_mean)
# fine grid
xt = jnp.delete(xs, N//2, axis=0)
xf = jnp.linspace(-1, 1, 1000)[:, None]
k_train_fine = jnp.exp(-L**2 * jnp.sum((xt[:, None] - xf)**2, axis=-1))
k_fine_fine = jnp.exp(-L**2 * jnp.sum((xf[:, None] - xf)**2, axis=-1))
yfm, yfv = kreg(k_train_train, k_train_fine, k_fine_fine, y_train)
yfm = yfm.flatten()
yfstd = jnp.sqrt(jnp.diag(yfv))
xf = xf.flatten()

# %% PANEL A
figsize = (10*cm, 5*cm)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=figsize)
ax.spines[:].set_visible(False)
ax.set_axis_off()
ax.scatter(xt, y_train, c=[colors[0] if y<0 else colors[1] for y in y_train])
ax.scatter(xs[N//2], ys[N//2], color='black', marker='o')
ax.plot(xf, yfm, color='black', lw=.75)
ax.fill_between(xf, yfm-3*yfstd, yfm+3*yfstd, alpha=.3, color='black')
ax.plot([xs[N//2], xs[N//2]], [ys[N//2], y_mean[0]], color='black', linestyle='--', lw=.75)
ax.annotate( r'$\varepsilon$', (.05, 0) )
x0, y0, x1, y1 = map(float, (xs[0, 0], ys[0, 0], xs[1, 0], ys[1, 0]))
xpred, ypred, ytrue = map(float, (xs[N//2, 0], y_mean[0, 0], ys[N//2, 0]))
ax.annotate( r'+1', (x0, y0), (x0+.2, y0+.5), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", shrinkB=5))
ax.annotate( r'-1', (x1, y1), (x1+.2, y1-.5), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.2", shrinkB=5))
ax.annotate( r'$\hat{y}(x_0)$', (xpred, ypred), (xpred+.2, ypred), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", shrinkB=5))
ax.annotate( r'$y_0$', (xpred, ytrue), (xpred+.2, ytrue-.5), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.2", shrinkB=5))
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.tight_layout()
plt.savefig('images/fig2_panelA.pdf')
plt.show()


# %% PANEL B
figsize = (5*cm, 5*cm)
fig, ax = plt.subplots(figsize=figsize)
ax.set_axis_off()
k_circ = jax.vmap(jnp.roll)(
    ein.repeat(
        ein.reduce(
            jax.vmap(jnp.roll)(k, -jnp.arange(len(k))),
            'rows cols -> cols',
            'mean'),
        'cols -> rows cols',
        rows=len(k)),
    jnp.arange(len(k)))
ax.imshow(k_circ, cmap='cividis')
rect1 = Rectangle( (-.5, N//2-.5), width=N, height=1, linewidth=1, edgecolor=colors[0], facecolor='none')
rect2 = Rectangle( (N//2-.5, -.5), width=1, height=N, linewidth=1, edgecolor=colors[0], facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.text(-1, N//2, r'$k(x_0, \vec{x}_\mathcal{D})$', horizontalalignment='right', verticalalignment='center', rotation=90)
ax.text(N//2, -1, r'$k(\vec{x}_\mathcal{D}, x_0)$', horizontalalignment='center', verticalalignment='bottom')
# plt.tight_layout()
plt.savefig('images/fig2_panelB_circ.pdf')
plt.show()
