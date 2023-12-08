import dataclasses
import jax.numpy as jnp
import jax.random as jr

from jax import vmap, lax
from jax.nn import one_hot
from jaxtyping import Array, Float
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance

from dynamax import hidden_markov_model as hmm
from dynamax import linear_gaussian_ssm as lds
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params
from dynamax.slds.laplace import block_tridiag_mvn_sample, laplace_approximation
from dynamax.utils.utils import register_pytree_node_dataclass, fit_linear_regression

@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class SLDSParams:
    """
    Container for the model parameters.

    Note: Assume that C, d, R are shared by all states
    Note: Assume that initial distribution is uniform over discrete state
          and the standard normal over the continuous latent state
    """
    transition_matrix : Float[Array, "num_states num_states"]
    dynamics_matrices : Float[Array, "num_states latent_dim latent_dim"]
    dynamics_biases : Float[Array, "num_states latent_dim"]
    dynamics_covs : Float[Array, "num_states latent_dim latent_dim"]
    emission_matrix : Float[Array, "emission_dim latent_dim"]
    emission_bias : Float[Array, "emission_dim"]
    emission_cov : Float[Array, "emission_dim emission_dim"]



class SLDS:
    """
    A switching linear dynamical system
    """
    def __init__(self,
                 num_states : int,
                 latent_dim : int,
                 emission_dim : int):
        self.num_states = num_states
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim

    def log_prob(self, ys, zs, xs, params):
        P = params.transition_matrix
        As = params.dynamics_matrices
        bs = params.dynamics_biases
        Qs = params.dynamics_covs
        C = params.emission_matrix
        d = params.emission_bias
        R = params.emission_cov

        lp = 0.0
        # log p(z_t | z_{t-1})
        lp += tfd.Categorical(probs=P[zs[:-1]]).log_prob(zs[1:]).sum()

        # log p(x_t | A_z x_{t-1} + b_z, Q_z)
        means = jnp.einsum('tde,te->td', As[zs[1:]], xs[:-1]) + bs[zs[1:]]
        lp += MVN(means, Qs[zs[1:]]).log_prob(xs[1:]).sum()

        # log p(y_t | C x_t + d, R)
        means = jnp.einsum('nd,td->tn', C, xs) + d
        lp += MVN(means, R).log_prob(ys).sum()
        return lp

    def sample(self,
               key : jr.PRNGKey,
               params : SLDSParams,
               num_timesteps : int):
        """

        """
        # Shorthand names
        K = self.num_states
        D = self.latent_dim
        N = self.emission_dim
        P = params.transition_matrix
        As = params.dynamics_matrices
        bs = params.dynamics_biases
        Qs = params.dynamics_covs
        C = params.emission_matrix
        d = params.emission_bias
        R = params.emission_cov

        # Sample the first time steps
        k1, k2, k3, key = jr.split(key, 4)
        z0 = tfd.Categorical(probs=jnp.ones(K) / K).sample(seed=k1)
        x0 = MVN(jnp.zeros(D), jnp.eye(D)).sample(seed=k2)
        y0 = MVN(C @ x0 + d, R).sample(seed=k3)

        def _step(carry, key):
            zp, xp, yp = carry
            k1, k2, k3, key = jr.split(key, 4)

            z = tfd.Categorical(probs=P[zp]).sample(seed=k1)
            x = MVN(As[z] @ xp + bs[z], Qs[z]).sample(seed=k2)
            y = MVN(C @ x + d, R).sample(seed=k3)
            return (z, x, y), (zp, xp, yp)

        _, (zs, xs, ys) = lax.scan(_step, (z0, x0, y0), jr.split(key, num_timesteps))
        return zs, xs, ys

    def _fit_map(self, emissions, initial_zs, initial_xs, initial_params, num_iters=100):
        """
        Find the MAP estimate of parameters and states using coordinate ascent.
        """
        K = self.num_states
        D = self.latent_dim
        N = self.emission_dim
        ys = emissions
        T = ys.shape[0]

        def _update_discrete_states(zs, xs, params):
            """
            Update the discrete states to the coordinate-wise maximum using the
            Viterbi algorithm.
            """
            pi0 = jnp.ones(K) / K
            P = params.transition_matrix
            As = params.dynamics_matrices   # (K, D, D)
            bs = params.dynamics_biases     # (K, D)
            Qs = params.dynamics_covs       # (K, D, D)

            # Compute the log likelihoods. In pseudocode:
            # for t in range(T):
            #     for k in range(K):
            #         log_likes[t,k] = MVN(As[k] @ x[t-1] + bs[k], Qs[k]).log_prob(x[t])
            means = jnp.einsum('kde,te->tkd', As, xs[:-1]) + bs
            log_likes = MVN(means, Qs).log_prob(xs[1:][:, None, :]) # (T-1, K)

            # Account for the first timestep
            log_likes = jnp.vstack([
                MVN(jnp.zeros(D), jnp.eye(D)).log_prob(xs[0]) * jnp.ones((1, K)),
                log_likes])

            return hmm.hmm_posterior_mode(pi0, P, log_likes)

        def _update_continuous_states(zs, xs, params):
            As = params.dynamics_matrices   # (K, D, D)
            bs = params.dynamics_biases     # (K, D)
            Qs = params.dynamics_covs       # (K, D, D)
            C = params.emission_matrix
            d = params.emission_bias
            R = params.emission_cov

            # Dynamax uses slightly different indexing. We need As[t] to be the
            # dynamics matrix used to map xs[t] to xs[t+1]. In this code, As[t]
            # maps xs[t-1] to xs[t]. We'll correct for that by padding zs with
            # a dummy value
            zs_pad = jnp.concatenate([zs, jnp.array([0])])[1:]
            lgssm_params = make_lgssm_params(
                initial_mean=jnp.zeros(D),
                initial_cov=jnp.eye(D),
                dynamics_weights=As[zs_pad],
                dynamics_bias=bs[zs_pad],
                dynamics_cov=Qs[zs_pad],
                emissions_weights=C,
                emissions_bias=d,
                emissions_cov=R,
            )

            posterior = lds.lgssm_smoother(lgssm_params, ys)
            return posterior.smoothed_means

        def _update_params(zs, xs, params):
            # Update the discrete state
            zsoh = one_hot(zs, K)
            P = jnp.einsum('ti,tj->ij', zsoh[:-1], zsoh[1:])
            P /= P.sum(axis=1, keepdims=True)

            # Update the dynamics parameters
            xs_pad = jnp.column_stack([xs, jnp.ones((T, 1))])
            ExpxpT = jnp.einsum('tk,td,te->kde', zsoh[1:], xs_pad[:-1], xs_pad[:-1])
            ExpxT = jnp.einsum('tk,td,te->kde', zsoh[1:], xs_pad[:-1], xs[1:])
            ExxT = jnp.einsum('tk,td,te->kde', zsoh[1:], xs[1:], xs[1:])
            Ns = jnp.einsum('tk->k', zsoh[1:])
            Abs, Qs = vmap(fit_linear_regression)(ExpxpT, ExpxT, ExxT, Ns)
            As, bs = Abs[..., :-1], Abs[..., -1]

            # Update emission parameters
            ExxT = jnp.einsum('td,te->de', xs_pad, xs_pad)
            ExyT = jnp.einsum('td,tn->dn', xs_pad, ys)
            EyyT = jnp.einsum('tm,tn->mn', ys, ys)
            Cd, R = fit_linear_regression(ExxT, ExyT, EyyT, T)
            C, d = Cd[..., :-1], Cd[..., -1]

            # Return an updated dataclass of parameters
            return dataclasses.replace(params,
                                       transition_matrix=P,
                                       dynamics_matrices=As,
                                       dynamics_biases=bs,
                                       dynamics_covs=Qs,
                                       emission_matrix=C,
                                       emission_bias=d,
                                       emission_cov=R
                                       )

        # Run the coordinate ascent algorithm!
        # params = initial_params
        # zs = initial_zs
        # xs = initial_xs
        # lps = []
        # for itr in range(num_iters):
        #     zs = _update_discrete_states(zs, xs, params)
        #     xs = _update_continuous_states(zs, xs, params)
        #     # params = _update_params(zs, xs, params)
        #     lps.append(self.log_prob(ys, zs, xs, params))
        # lps = jnp.stack(lps)

        def _step(carry, args):
            zs, xs, params = carry
            lp = self.log_prob(ys, zs, xs, params)
            zs = _update_discrete_states(zs, xs, params)
            xs = _update_continuous_states(zs, xs, params)
            params = _update_params(zs, xs, params)
            return (zs, xs, params), lp

        initial_carry = (initial_zs, initial_xs, initial_params)
        (zs, xs, params), lps = lax.scan(_step, initial_carry, None, length=num_iters)

        return lps, zs, xs, params

    def _fit_laplace_em(self, key, emissions, initial_zs, initial_xs, initial_params,
                        num_iters=100, n_discrete_samples=1):
        """
        Estimate the parameters of the SLDS and an approximate posterior distr.
        over latent states using Laplace EM. Specifically, the approximate
        posterior factors over discrete and continuous latent states. The
        discrete state posterior is a discrete chain graph, and the continuous
        posterior is a linear Gaussian chain. We estimate the continuous posterior
        using a Laplace approximation, which is appropriate when the likelihood
        is log concave in the continuous states.
        """
        K = self.num_states
        D = self.latent_dim
        N = self.emission_dim
        ys = emissions
        T = ys.shape[0]

        def _update_discrete_states(key, params, J_diag, J_lower_diag, h):
            """
            Update the discrete states to the coordinate-wise maximum using the
            Viterbi algorithm.
            """
            As = params.dynamics_matrices   # (K, D, D)
            bs = params.dynamics_biases     # (K, D)
            Qs = params.dynamics_covs       # (K, D, D)

            # sample xs from q(x)
            key, *skeys = jr.split(key, n_discrete_samples+1)
            vmap_block_tridiag_mvn_sample = vmap(block_tridiag_mvn_sample, in_axes=(0, None, None, None))
            x_samples = vmap_block_tridiag_mvn_sample(jnp.array(skeys), J_diag, J_lower_diag, h)

            # TODO: replace with initial state distribution object
            initial_state_distn = jnp.ones(K) / K
            pi0 = jnp.mean(jnp.array(
                [initial_state_distn
                    for x in x_samples]), axis=0)

            # TODO: eventually, transition matrix will depend on x
            # this should be another to call to a function that returns transition matrices
            P = jnp.mean(jnp.array(
                [params.transition_matrix
                    for x in x_samples]), axis=0)

            def _dynamics_likelihood(xs):
                means = jnp.einsum('kde,te->tkd', As, xs[:-1]) + bs
                log_likes = MVN(means, Qs).log_prob(xs[1:][:, None, :]) # (T-1, K)
                # Account for the first timestep
                log_likes = jnp.vstack([
                    MVN(jnp.zeros(D), jnp.eye(D)).log_prob(xs[0]) * jnp.ones((1, K)),
                    log_likes])
                return log_likes
            vmap_dynamics_likelihood = vmap(_dynamics_likelihood)

            log_likes = jnp.mean(vmap_dynamics_likelihood(x_samples), axis=0)

            return hmm.inference.hmm_smoother(pi0, P, log_likes)

            # TODO: Figure out how to return a discrete chain graphical model

        def _update_continuous_states(ys, zs, xs, params):
            P = params.transition_matrix
            As = params.dynamics_matrices   # (K, D, D)
            bs = params.dynamics_biases     # (K, D)
            Qs = params.dynamics_covs       # (K, D, D)
            C = params.emission_matrix
            d = params.emission_bias
            R = params.emission_cov

            # Define log prob functions that close over zs and params
            log_prob = lambda xs, ys: self.log_prob(ys, zs, xs, params) #TODO get access to data

            # TODO : change these to slds object distributions
            # TODO : marginalize over q(z)
            initial_distribution = lambda x0: MVN(jnp.zeros(D), jnp.eye(D)).log_prob(x0)
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

            return Ex, ExxT, ExxnT, J_diag, J_lower_diag, h

        def _update_params(zs, xs, params):
            # Update the discrete state
            zsoh = one_hot(zs, K)
            P = jnp.einsum('ti,tj->ij', zsoh[:-1], zsoh[1:])
            P /= P.sum(axis=1, keepdims=True)

            # Update the dynamics parameters
            xs_pad = jnp.column_stack([xs, jnp.ones((T, 1))])
            ExpxpT = jnp.einsum('tk,td,te->kde', zsoh[1:], xs_pad[:-1], xs_pad[:-1])
            ExpxT = jnp.einsum('tk,td,te->kde', zsoh[1:], xs_pad[:-1], xs[1:])
            ExxT = jnp.einsum('tk,td,te->kde', zsoh[1:], xs[1:], xs[1:])
            Ns = jnp.einsum('tk->k', zsoh[1:])
            Abs, Qs = vmap(fit_linear_regression)(ExpxpT, ExpxT, ExxT, Ns)
            As, bs = Abs[..., :-1], Abs[..., -1]

            # Update emission parameters
            ExxT = jnp.einsum('td,te->de', xs_pad, xs_pad)
            ExyT = jnp.einsum('td,tn->dn', xs_pad, ys)
            EyyT = jnp.einsum('tm,tn->mn', ys, ys)
            Cd, R = fit_linear_regression(ExxT, ExyT, EyyT, T)
            C, d = Cd[..., :-1], Cd[..., -1]

            # Return an updated dataclass of parameters
            return dataclasses.replace(params,
                                       transition_matrix=P,
                                       dynamics_matrices=As,
                                       dynamics_biases=bs,
                                       dynamics_covs=Qs,
                                       emission_matrix=C,
                                       emission_bias=d,
                                       emission_cov=R
                                       )

        # Run the coordinate ascent algorithm!
        # params = initial_params
        # zs = initial_zs
        # xs = initial_xs
        # lps = []
        # for itr in range(num_iters):
        #     zs = _update_discrete_states(zs, xs, params)
        #     xs = _update_continuous_states(zs, xs, params)
        #     # params = _update_params(zs, xs, params)
        #     lps.append(self.log_prob(ys, zs, xs, params))
        # lps = jnp.stack(lps)

        def _step(carry, args):
            zs, xs, params, key = carry
            Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = _update_continuous_states(ys, zs, xs, params)
            xs = Ex # redefine xs as mean
            key, skey = jr.split(key)
            post = _update_discrete_states(skey, params, J_diag, J_lower_diag, h)
            zs = jnp.argmax(post.smoothed_probs, axis=1)
            params = _update_params(zs, xs, params)
            lp = self.log_prob(ys, zs, xs, params)
            return (zs, xs, params, key), lp

        initial_carry = (initial_zs, initial_xs, initial_params, key)
        (zs, xs, params, key), lps = lax.scan(_step, initial_carry, None, length=num_iters)

        return lps, zs, xs, params, key