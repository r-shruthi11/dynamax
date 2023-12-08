import jax.numpy as jnp
import jax.random as jr

from dynamax.slds.models import SLDSParams, SLDS
from dynamax.utils.plotting import plot_states_and_timeseries
from dynamax.utils.plotting import COLORS as colors

import matplotlib.pyplot as plt


def simulate_slds(key,
                  num_states=5,
                  latent_dim=2,
                  emission_dim=10,):

    # Generate transition matrix
    p = (jnp.arange(num_states)**10).astype(float)
    p /= p.sum()
    P = jnp.zeros((num_states, num_states))
    for k, p in enumerate(p[::-1]):
        P += jnp.roll(p * jnp.eye(num_states), k, axis=1)

    # Generate dynamics
    rot = lambda theta: jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                               [jnp.sin(theta), jnp.cos(theta)]])

    angles = jnp.linspace(0, 2 * jnp.pi, num_states, endpoint=False)
    theta = -jnp.pi / 25 # rotational frequency
    As = jnp.array([0.8 * rot(theta) for _ in range(num_states)])
    bs = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
    Qs = jnp.tile(0.001 * jnp.eye(latent_dim), (num_states, 1, 1))

    # Generate emissions
    k1, key = jr.split(key)
    C = jr.normal(k1, (emission_dim, latent_dim))
    d = jnp.zeros(emission_dim)
    R = jnp.eye(emission_dim)

    # Pack parameters into an SLDSParams object
    params = SLDSParams(P, As, bs, Qs, C, d, R)
    
    slds = SLDS(num_states, latent_dim, emission_dim)

    # Sample from the model
    num_timesteps = 10000
    zs, xs, ys = slds.sample(key, 
                             params, 
                             num_timesteps)
    
    # Plot the sampled data
    fig = plt.figure(figsize=(8, 8))
    for k in range(num_states):
        plt.plot(*xs[zs==k].T, 'o', color=colors[k],
            alpha=0.75, markersize=3)

    plt.plot(*xs[:1000].T, '-k', lw=0.5, alpha=0.2)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

    # Plot the states and timeseries
    fig, ax = plot_states_and_timeseries(zs, xs)
    ax.set_xlim(0, 200)
    ax.set_title("latent states")
    plt.show()

    fig, ax = plot_states_and_timeseries(zs, ys)
    ax.set_xlim(0, 200)
    ax.set_title("emissions")
    plt.show()

    return zs, xs, ys, slds, params


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    zs, xs, ys, params = simulate_slds(key)