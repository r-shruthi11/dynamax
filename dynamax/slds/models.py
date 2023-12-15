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
    pi0 : Float[Array, "num_states"]
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
                 emission_dim : int,
                 params : SLDSParams):
        self.num_states = num_states
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim
        self.params = params

    def init_continuous_state_distn(self):
        return MVN(jnp.zeros(self.latent_dim), jnp.eye(self.latent_dim))

    def init_discrete_state_distn(self):
        return tfd.Categorical(probs=self.params.pi0)

    def transition_distn(self, z):
        """
        Currently z can be a scalar or a sequence
        """
        P = self.params.transition_matrix
        return tfd.Categorical(probs=P[z])

    def dynamics_distn(self, z, x):
        As = self.params.dynamics_matrices
        bs = self.params.dynamics_biases
        Qs = self.params.dynamics_covs
        return MVN(As[z] @ x + bs[z], Qs[z])

    def dynamics_distributions(self, zs, xs):
        # TODO: combine this with dynamics_distn
        As = self.params.dynamics_matrices
        bs = self.params.dynamics_biases
        Qs = self.params.dynamics_covs
        means = jnp.einsum('tde,te->td', As[zs], xs) + bs[zs]
        return MVN(means, Qs[zs])

    def emission_distn(self, x):
        C = self.params.emission_matrix
        d = self.params.emission_bias
        R = self.params.emission_cov
        return MVN(C @ x + d, R)

    def emission_distributions(self, xs):
        # TODO: combine this with emission_distn
        C = self.params.emission_matrix
        d = self.params.emission_bias
        R = self.params.emission_cov
        means = jnp.einsum('nd,td->tn', C, xs) + d
        return MVN(means, R)

    def log_prob(self, ys, zs, xs):
        lp = 0.0
        # log p(z_t | z_{t-1})
        lp += self.transition_distn(zs[:-1]).log_prob(zs[1:]).sum()
        # log p(x_t | A_z x_{t-1} + b_z, Q_z)
        lp += self.dynamics_distributions(zs[1:], xs[:-1]).log_prob(xs[1:]).sum()
        # log p(y_t | C x_t + d, R)
        lp += self.emission_distributions(xs).log_prob(ys).sum()
        return lp

    def sample(self,
               key : jr.PRNGKey,
               num_timesteps : int):
        """

        """
        # Sample the first time steps
        k1, k2, k3, key = jr.split(key, 4)
        z0 = self.init_discrete_state_distn().sample(seed=k1)
        x0 = self.init_continuous_state_distn().sample(seed=k2)
        y0 = self.emission_distn(x0).sample(seed=k3)

        def _step(carry, key):
            zp, xp, yp = carry
            k1, k2, k3, key = jr.split(key, 4)

            z = self.transition_distn(zp).sample(seed=k1)
            x = self.dynamics_distn(z, xp).sample(seed=k2)
            y = self.emission_distn(x).sample(seed=k3)
            return (z, x, y), (zp, xp, yp)

        _, (zs, xs, ys) = lax.scan(_step, (z0, x0, y0), jr.split(key, num_timesteps))
        return zs, xs, ys

def _fit_laplace_em(slds, key, emissions, initial_zs, initial_xs,
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
    K = slds.num_states
    D = slds.latent_dim
    N = slds.emission_dim
    ys = emissions
    T = ys.shape[0]

    def _update_discrete_states(slds, key, J_diag, J_lower_diag, h):
        """
        Update the discrete states to the coordinate-wise maximum using the
        Viterbi algorithm.
        """
        # sample xs from q(x)
        key, *skeys = jr.split(key, n_discrete_samples+1)
        vmap_block_tridiag_mvn_sample = vmap(block_tridiag_mvn_sample, in_axes=(0, None, None, None))
        x_samples = vmap_block_tridiag_mvn_sample(jnp.array(skeys), J_diag, J_lower_diag, h)

        pi0 = jnp.mean(jnp.array(
            [slds.params.pi0
                for x in x_samples]), axis=0)

        # TODO: eventually, transition matrix will depend on x
        P = jnp.mean(jnp.array(
            [slds.params.transition_matrix
                for x in x_samples]), axis=0)

        # TODO: change this to call dynamics distribution
        def _dynamics_likelihood(xs):
            As = slds.params.dynamics_matrices   # (K, D, D)
            bs = slds.params.dynamics_biases     # (K, D)
            Qs = slds.params.dynamics_covs       # (K, D, D)
            means = jnp.einsum('kde,te->tkd', As, xs[:-1]) + bs
            log_likes = MVN(means, Qs).log_prob(xs[1:][:, None, :]) # (T-1, K)
            # Account for the first timestep
            log_likes = jnp.vstack([
                slds.init_continuous_state_distn().log_prob(xs[0]) * jnp.ones((1, K)),
                log_likes])
            return log_likes
        vmap_dynamics_likelihood = vmap(_dynamics_likelihood)

        log_likes = jnp.mean(vmap_dynamics_likelihood(x_samples), axis=0)

        return hmm.inference.hmm_smoother(pi0, P, log_likes)

    def _update_continuous_states(slds, ys, zs, xs):

        # Define log prob functions that close over zs and
        log_prob = lambda xs, ys: slds.log_prob(ys, zs, xs) #TODO get access to data

        # TODO : change these to slds object distributions
        # TODO : marginalize over q(z)
        initial_distribution = lambda x0: slds.init_continuous_state_distn().log_prob(x0)
        dynamics_distribution = lambda t, xt, xtp1: slds.dynamics_distn(zs[t+1], xt).log_prob(xtp1)
        emission_distribution = lambda t, xt, yt: slds.emission_distn(xt).log_prob(yt)
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

    def _update_params(slds, zs, xs):

        # Currently, initial state distn is fixed
        pi0 = slds.params.pi0

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

        # Return updated dataclass of parameters
        slds.params = dataclasses.replace(slds.params,
                                    pi0 = pi0,
                                    transition_matrix=P,
                                    dynamics_matrices=As,
                                    dynamics_biases=bs,
                                    dynamics_covs=Qs,
                                    emission_matrix=C,
                                    emission_bias=d,
                                    emission_cov=R
                                    )
        return

    def _step(carry, args):
        zs, xs, key = carry
        Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = _update_continuous_states(slds, ys, zs, xs)
        xs = Ex # redefine xs as mean
        key, skey = jr.split(key)
        post = _update_discrete_states(slds, skey, J_diag, J_lower_diag, h)
        zs = jnp.argmax(post.smoothed_probs, axis=1)
        _update_params(slds, zs, xs)
        lp = slds.log_prob(ys, zs, xs)
        return (zs, xs, key), lp

    initial_carry = (initial_zs, initial_xs, key)
    (zs, xs, key), lps = lax.scan(_step, initial_carry, None, length=num_iters)

    return lps, zs, xs, key