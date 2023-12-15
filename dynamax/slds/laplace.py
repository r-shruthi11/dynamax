import jax
import jax.numpy as jnp
import jax.random as jr

from jax import jit, vmap, lax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance



import optax
import jax.scipy.optimize
from jax import hessian, value_and_grad, jacfwd, jacrev
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_expectations
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_log_normalizer

def _sample_info_gaussian(key, J, h, sample_shape=()):
    # TODO: avoid inversion.
    # see https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/stats.py#L117-L122
    # L = np.linalg.cholesky(J)
    # x = jr.normal(key, h.shape[0])
    # return scipy.linalg.solve_triangular(L,x,lower=True,trans='T') \
    #     + dpotrs(L,h,lower=True)[0]
    cov = jnp.linalg.inv(J)
    loc = jnp.einsum("...ij,...j->...i", cov, h)
    return tfp.distributions.MultivariateNormalFullCovariance(
        loc=loc, covariance_matrix=cov).sample(sample_shape=sample_shape, seed=key)

def block_tridiag_mvn_sample(key, J_diag, J_lower_diag, h):
    # Run the forward filter
    log_Z, (filtered_Js, filtered_hs) = block_tridiag_mvn_log_normalizer(J_diag, J_lower_diag, h)

    # Backward sample
    def _step(carry, inpt):
        x_next, key = carry
        Jf, hf, L = inpt

        # Condition on the next observation
        Jc = Jf
        # hc = hf - jnp.einsum('ni,ij->nj', x_next, L)
        hc = hf - L.T @ x_next

        # Split the key
        key, this_key = jr.split(key)
        x = _sample_info_gaussian(this_key, Jc, hc)
        return (x, key), x

    # Initialize with sample of last timestep and sample in reverse
    last_key, key = jr.split(key)
    x_T = _sample_info_gaussian(last_key, filtered_Js[-1], filtered_hs[-1])

    # inputs = (filtered_Js[:-1][::-1], filtered_hs[:-1][::-1], J_lower_diag[::-1])
    # _, x_rev = lax.scan(_step, (x_T, key), inputs)
    _, x = lax.scan(_step, (x_T, key), (filtered_Js[:-1], filtered_hs[:-1], J_lower_diag), reverse=True)

    # Reverse and concatenate the last time-step's sample
    x = jnp.concatenate((x, x_T[None, ...]), axis=0)

    # Transpose to be (num_samples, num_timesteps, dim)
    return x

def laplace_approximation(log_prob,
                          initial_distribution,
                          dynamics_distribution,
                          emission_distribution,
                          initial_states,
                          emissions,
                          method="BFGS",
                          adam_learning_rate=1e-2,
                          num_iters=10):
    """
    Laplace approximation to the posterior distribution for state space models
    with continuous latent states.

    log_prob: states, emissions -> log prob (scalar)
    initial_distribution: initial state -> log prob (scalar)
    dynamics_distribution: time, curr_state, next_state -> log prob (scalar)
    emission_distribution: time, curr_state, curr_emission -> log prob (scalar)
    x0 (array, (num_timesteps, latent_dim)): Initial guess of state mode.
    data (array, (num_timesteps, obs_dim)): Observation data.
    method (str, optional): Optimization method to use. Choices are
        ["L-BFGS", "BFGS", "Adam"]. Defaults to "L-BFGS".
    learning_rate (float, optional): [description]. Defaults to 1e-3.
    num_iters (int, optional): Only used when optimization method is "Adam."
        Specifies the number of update iterations. Defaults to 50.

    """
    def _compute_laplace_mean(initial_states):
        """Find the mode of the log joint probability for the Laplace approximation.
        """

        scale = initial_states.size
        dim = initial_states.shape[-1]

        if method == "BFGS" or "L-BFGS":
            # scipy minimize expects x to be shape (n,) so we flatten / unflatten
            def _objective(x_flattened):
                x = x_flattened.reshape(-1, dim)
                return -1 * jnp.sum(log_prob(x, emissions)) / scale

            optimize_results = jax.scipy.optimize.minimize(
                _objective,
                initial_states.ravel(),
                method="bfgs" if method == "BFGS" else "l-bfgs-experimental-do-not-rely-on-this",
                options=dict(maxiter=num_iters))

            # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
            x_mode = optimize_results.x.reshape(-1, dim)  # reshape back to (T, D)

        elif method == "Adam":

            params = initial_states
            _objective = lambda x: -1 * jnp.sum(log_prob(x, emissions)) / scale
            optimizer = optax.adam(adam_learning_rate)
            opt_state = optimizer.init(params)

            @jit
            def step(params, opt_state):
                loss_value, grads = value_and_grad(_objective)()
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss_value

            # TODO: Replace with a scan
            for i in range(num_iters):
                params, opt_state, loss_value = step(params, opt_state)
            x_mode = params

        else:
            raise ValueError(f"method = {method} is not recognized. Should be one of ['Adam', 'BFGS']")

        return x_mode

    def _compute_laplace_precision_blocks(states):
        """Get the negative Hessian at the given states for the Laplace approximation.
        """
        # initial distribution
        J_init = -1 * hessian(initial_distribution)(states[0])

        # dynamics
        f = dynamics_distribution
        ts = jnp.arange(len(states))
        J_11 = -1 * vmap(hessian(f, argnums=1))(ts[:-1], states[:-1], states[1:])
        J_22 = -1 * vmap(hessian(f, argnums=2))(ts[:-1], states[:-1], states[1:])
        J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=2), argnums=1))(ts[:-1], states[:-1], states[1:])

        # emissions
        f = emission_distribution
        J_obs = -1 * vmap(hessian(f, argnums=1))(ts, states, emissions)

        # debug only if this flag is set
        # if jax.config.jax_disable_jit:
        #     assert not np.any(np.isnan(J_init)), "nans in J_init"
        #     assert not np.any(np.isnan(J_11)), "nans in J_11"
        #     assert not np.any(np.isnan(J_22)), "nans in J_22"
        #     assert not np.any(np.isnan(J_21)), "nans in J_21"
        #     assert not np.any(np.isnan(J_obs)), "nans in J_obs"

        # combine into diagonal and lower diagonal blocks
        J_diag = J_obs
        J_diag = J_diag.at[0].add(J_init)
        J_diag = J_diag.at[:-1].add(J_11)
        J_diag = J_diag.at[1:].add(J_22)
        J_lower_diag = J_21
        return J_diag, J_lower_diag


    # Find the mean and precision of the Laplace approximation
    mu = _compute_laplace_mean(initial_states)

    # The precision is given by the negative hessian at the mode
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(mu)

    # Compute the linear potential by multiplying a block tridiagonal matrix with a vector
    # We represent the block tridiag matrix with the (T, D, D) array of diagonal blocks
    # and the (T-1, D, D) array of lower diagonal blocks. The vector is represented
    # as a (T, D) array.
    f = vmap(jnp.matmul)
    h = f(J_diag, mu) # (T, D)
    h = h.at[1:].add(f(J_lower_diag, mu[:-1]))
    h = h.at[:-1].add(f(jnp.swapaxes(J_lower_diag, -1, -2), mu[1:]))

    log_normalizer, Ex, ExxT, ExxnT = block_tridiag_mvn_expectations(J_diag, J_lower_diag, h)

    # Returns log_normalizer, Ex, ExxT, ExxnT, and posterior params for sampling
    return log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h