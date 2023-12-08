from jax import vmap
import jax.numpy as jnp
import jax.random as jr

from dynamax.slds.test_models import simulate_slds
from dynamax.slds.laplace import laplace_approximation, block_tridiag_mvn_sample
from tensorflow_probability.substrates import jax as tfp
from dynamax.utils.plotting import plot_states_and_timeseries

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    zs, xs, ys, slds, params = simulate_slds(key)

    # Create distributions that are passed to the laplace approximation
    P = params.transition_matrix
    As = params.dynamics_matrices
    bs = params.dynamics_biases
    Qs = params.dynamics_covs
    C = params.emission_matrix
    d = params.emission_bias
    R = params.emission_cov

    latent_dim = slds.latent_dim
    emission_dim = slds.emission_dim
    num_states = slds.num_states

    # Define log prob functions that close over zs and params
    log_prob = lambda xs, ys: slds.log_prob(ys, zs, xs, params)
    initial_distribution = lambda x0: MVN(jnp.zeros(latent_dim), jnp.eye(latent_dim)).log_prob(x0)
    dynamics_distribution = lambda t, xt, xtp1: MVN(As[zs[t+1]] @ xt + bs[zs[t+1]], Qs[zs[t+1]]).log_prob(xtp1)
    emission_distribution = lambda t, xt, yt: MVN(C @ xt + d, R).log_prob(yt)

    log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = \
    laplace_approximation(log_prob,
                        initial_distribution,
                        dynamics_distribution,
                        emission_distribution,
                        jnp.zeros_like(xs),
                        ys,
                        method="L-BFGS",
                        num_iters=50)

    x_sample = block_tridiag_mvn_sample(jr.PRNGKey(0), J_diag, J_lower_diag, h)
    
    fig, ax = plot_states_and_timeseries(zs, xs)
    ax.set_xlim(0, 200)
    ax.set_title("latent states")

    cov_x = ExxT - jnp.einsum('ti,tj->tij', Ex, Ex)
    fig, ax = plot_states_and_timeseries(zs, Ex, jnp.sqrt(vmap(jnp.diag)(cov_x)))
    ax.set_xlim(0, 200)
    ax.set_title("inferred latent states")

    fig, ax = plot_states_and_timeseries(zs, x_sample)
    ax.set_xlim(0, 200)
    ax.set_title("inferred latent states")