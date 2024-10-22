# %%
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array
import einops as ein


def kreg(
    k11: Float[Array, 'train train'],
    k12: Float[Array, 'train test'],
    k22: Float[Array, 'test test'],
    y: Float[Array, 'train 1'],
    reg: float = 1e-5) -> tuple[Float[Array, 'test 1'], Float[Array, 'test 1']]:
    """GP prediction"""
    sol = jax.scipy.linalg.solve( k11 + reg*jnp.eye(len(k11)), k12, assume_a='pos' )
    mean = ein.einsum(sol, y, 'train test, train d-> test d')
    var = k22 - ein.einsum(sol, k12, 'train t1, train t2 -> t1 t2')
    return mean, var

# %% Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 10  # keep this to even values
    L = 2
    reg = 1e-5
    xs = jnp.linspace(-1, 1, N)[:, None]
    ys = jnp.array([-1., 1.] * (N//2))[:, None]
    k = jnp.exp(-L**2 * jnp.sum((xs - xs[:, None])**2, axis=-1))
    k_train_train = jnp.delete(jnp.delete(k, N//2, axis=0), N//2, axis=1)
    k_train_test = jnp.delete(k[:, N//2:N//2+1], N//2, axis=0)
    k_test_test = k[N//2:N//2+1, N//2:N//2+1]
    y_train = jnp.delete(ys, N//2, axis=0)
    y_pred, y_var = kreg(k_train_train, k_train_test, k_test_test, y_train, reg)
    error = jnp.abs(ys[N//2]-y_pred)
    print(f"Prediction error at missing point: {error[0, 0]:.2f} pm {jnp.sqrt(y_var)[0, 0]:.2f}")
    # on a finer grid for plotting
    xt = jnp.delete(xs, N//2, axis=0)
    yt = jnp.delete(ys, N//2, axis=0)
    xf = jnp.linspace(-1, 1, 100)[:, None]
    k_train_fine = jnp.exp(-L**2 * jnp.sum((xf - xt[:, None])**2, axis=-1))
    k_fine_fine = jnp.exp(-L**2 * jnp.sum((xf - xf[:, None])**2, axis=-1))
    yf_mean, yf_var = kreg(k_train_train, k_train_fine, k_fine_fine, y_train, reg)
    yf_std = jnp.sqrt(jnp.diag(yf_var))
    plt.plot(xf, yf_mean)
    plt.scatter(xt, yt)
    plt.fill_between(xf.flatten(), yf_mean.flatten()-yf_std, yf_mean.flatten()+yf_std, alpha=1)
    plt.show()
