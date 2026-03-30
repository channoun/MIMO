"""
Alpha-stable score diffusion models (future extension).

Placeholder for score-based generative models that use alpha-stable
forward processes instead of the standard Gaussian (Ornstein-Uhlenbeck).

An alpha-stable SDE has the form:
    dX_t = f(X_t, t) dt + g(t) dL_t^alpha

where L_t^alpha is an isotropic alpha-stable Lévy process.

The score ∇_x ln q_t(x) can be estimated via a generalized DSM loss:
    L(θ) = E_{X0, X_t} [ ||s_θ(X_t, t) - ∇_{X_t} ln p_{t|0}(X_t | X0)||^2 ]

For alpha-stable transitions, ∇_{X_t} ln p_{t|0} is not Gaussian and
requires evaluation of the stable density gradient — this is the main
computational challenge.

Status: Long-term research extension. Not implemented for the initial paper.
"""
import warnings
warnings.warn(
    "extensions/stable_score.py: Alpha-stable score diffusion is a long-term "
    "research extension and is not yet implemented. "
    "See CLAUDE.md Section 'Alpha-stable score diffusion priors' for details.",
    FutureWarning,
    stacklevel=2,
)


class AlphaStableScoreNet:
    """
    Placeholder for alpha-stable score diffusion model.

    This class will implement a score network trained with the generalized
    DSM loss for alpha-stable forward processes.

    For now, raises NotImplementedError on instantiation.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AlphaStableScoreNet is not yet implemented. "
            "Use SubGaussianStableNoise + stable_likelihood_score for the "
            "stable noise extension (Section V of CLAUDE.md)."
        )
