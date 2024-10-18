from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float


def three_shear_rotate(img: Float[Array, 'd d'], theta: float) -> Float[Array, 'd d']:
    """
    Perform image rotation using the three-shear method.
    This function performs a quick and interpolation-free rotation of a square,
    single-channel image using the three-shear method. It conserves the pixels
    by shifting them across the image without interpolating them. The function
    supports batch processing via `vmap` for efficient computation on multiple
    images.
    Parameters
    ----------
    img : Float[Array, 'd d']
        A 2D array representing the input image, assumed to be square-shaped
        and single-channel. The image is rotated in-place, where `d` is the 
        dimension of the image (width and height).
    
    theta : float
        The angle of rotation in radians.
    Returns
    -------
    Float[Array, 'd d']
        The rotated image as a 2D array of the same shape as the input `img`. 
        Pixels outside the boundary of the original image are set to 0.
    Notes
    -----
    This method uses three shear transformations to rotate the image without
    interpolation. The process ensures that no pixel values are altered, only 
    repositioned.
    Assumptions
    -----------
    - The input image is square-shaped.
    - The image has a single channel (grayscale).
    
    This method may not work as intended for images that do not meet these
    assumptions.
    TODO
    ----
    - Rewrite this function with `einops` instead of the current reshaping 
      strategy to improve clarity and reduce reliance on manual reshaping.
    References
    ----------
    - See Tom Forsyth's article on the three-shear rotation method:
      https://cohost.org/tomforsyth/post/891823-rotation-with-three
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Float, Array
    >>> img = jnp.ones((100, 100))
    >>> rotated_img = three_shear_rotate(img, jnp.pi / 4)
    >>> rotated_img.shape
    (100, 100)
    """
    w, h = img.shape
    s = w // 2  # assumes square images
    shear_tan = jnp.array([[1, -jnp.tan(theta/2)], [0, 1]])
    shear_sin = jnp.array([[1, 0], [jnp.sin(theta), 1]])
    og_grid = jnp.stack(jnp.meshgrid(
        jnp.arange(-s, s),
        jnp.arange(-s, s))).T
    new_grid = shear_tan @ shear_sin @ shear_tan @ (og_grid.reshape(-1, 2).T)
    qng = new_grid.T.reshape(h, w, 2).astype(int)
    rotated = img[qng[..., 0]+s, qng[..., 1]+s]
    return jnp.where((jnp.abs(qng) > s).any(-1), 0, rotated)


def sgd_dataloader(xs, ys, batch_size, *, key):
    """
    Generates batches of data for Stochastic Gradient Descent (SGD).

    Args:
        xs (jax.numpy.ndarray): Input features.
        ys (jax.numpy.ndarray): Corresponding labels.
        batch_size (int): Size of each batch.
        key (jax.random.PRNGKey): Key for random number generation.

    Yields:
        tuple: A tuple containing the batch of input features and corresponding labels.

    Raises:
        ValueError: If the lengths of input features and labels do not match.

    Notes:
        This function generates batches of data for stochastic gradient descent (SGD).
        It shuffles the dataset at each iteration and yields batches of specified size.

    Example:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>> key = random.PRNGKey(0)
        >>> xs = jnp.array([[1, 2], [3, 4], [5, 6]])
        >>> ys = jnp.array([0, 1, 0])
        >>> dataloader = sgd_dataloader(xs, ys, batch_size=2, key=key)
        >>> for batch_xs, batch_ys in dataloader:
        ...     print(batch_xs, batch_ys)
        [[3 4]
         [1 2]] [1 0]
        [[5 6]] [0]
    """
    if len(xs) != len(ys):
        raise ValueError(f"Wrong data shapes: len(xs)={len(xs)}, len(ys)={len(ys)}")

    dataset_size = xs.shape[0]
    indices = jnp.arange(dataset_size)

    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)

        start = 0
        end = batch_size

        while end < dataset_size:
            batch_perm = perm[start:end]
            yield xs[batch_perm], ys[batch_perm]
            start = end
            end = start + batch_size
