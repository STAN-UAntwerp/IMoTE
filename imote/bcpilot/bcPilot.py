"""This module implements PILOT for binary classification"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from pydantic import BaseModel
from sklearn.base import BaseEstimator
from dataclasses import dataclass

from .bcTree import Tree
from .bcSplit import Split

# Set up logging for PILOT
logger = logging.getLogger("pilot")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

REGRESSION_NODES = ["con", "lin", "pcon", "plin"]
NODE_PREFERENCE_ORDER = REGRESSION_NODES + ["pconc"]
DEFAULT_DF_SETTINGS = {"con": 1.0, "lin": 1.5, "pcon": 2.1, "plin": 9.0, "pconc": 1.1}


@dataclass
class BICCandidate:
    """Stores information about a BIC candidate evaluated during split selection."""

    node_type: str
    feature_id: int
    pivot: float
    bic: float
    wrss: float
    intercept_left: float
    slope_left: float
    intercept_right: float
    slope_right: float
    n_left: int = 0
    n_right: int = 0

    def __repr__(self):
        return (
            f"BICCandidate({self.node_type}, feat={self.feature_id}, "
            f"pivot={self.pivot:.3f}, bic={self.bic:.2f})"
        )


# Numerical stability constants
MAX_COEFF_VALUE = 1000.0  # Maximum allowed coefficient value
MIN_DENOM_VALUE = 1e-10  # Minimum denominator to avoid division by zero


def _clip_coefficient(coeff, max_val=MAX_COEFF_VALUE):
    """Clip coefficients to prevent numerical instability"""
    return np.clip(coeff, -max_val, max_val)


class Moments:
    def __init__(
        self,
        x: np.ndarray = np.array([]),
        y: np.ndarray = np.array([]),
        w: np.ndarray = np.array([]),
    ) -> None:
        assert len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        assert w.shape[0] == y.shape[0]

        # Clip y to prevent overflow in squared terms
        y_clipped = np.clip(y, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)
        x_clipped = np.clip(x, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)

        self.n: int = len(y_clipped)
        self.sum_y: float = np.sum(y_clipped)
        self.sum_y2: float = np.sum(y_clipped**2)
        self.sum_wx: float = np.sum(w * x_clipped)
        self.sum_wx2: float = np.sum(w * (x_clipped**2))
        self.sum_wy: float = np.sum(w * y_clipped)
        self.sum_wy2: float = np.sum(w * (y_clipped**2))
        self.sum_wxy: float = np.sum(w * x_clipped * y_clipped)
        self.sum_w: float = np.sum(w)

    def update_pivot(self, x, y, w=None, moment_direction="left"):
        if w is None:
            w = np.ones_like(y)
        if moment_direction == "right":
            # Clip values to prevent overflow
            y_clipped = np.clip(y, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)
            x_clipped = np.clip(x, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)

            self.n -= len(y_clipped)
            self.sum_y -= np.sum(y_clipped)
            self.sum_y2 -= np.sum(y_clipped**2)
            self.sum_wx -= np.sum(w * x_clipped)
            self.sum_wx2 -= np.sum(w * (x_clipped**2))
            self.sum_wy -= np.sum(w * y_clipped)
            self.sum_wy2 -= np.sum(w * (y_clipped**2))
            self.sum_wxy -= np.sum(w * x_clipped * y_clipped)
            self.sum_w -= np.sum(w)
        elif moment_direction == "left":
            # Clip values to prevent overflow
            y_clipped = np.clip(y, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)
            x_clipped = np.clip(x, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)

            self.n += len(y_clipped)
            self.sum_y += np.sum(y_clipped)
            self.sum_y2 += np.sum(y_clipped**2)
            self.sum_wx += np.sum(w * x_clipped)
            self.sum_wx2 += np.sum(w * (x_clipped**2))
            self.sum_wy += np.sum(w * y_clipped)
            self.sum_wy2 += np.sum(w * (y_clipped**2))
            self.sum_wxy += np.sum(w * x_clipped * y_clipped)
            self.sum_w += np.sum(w)
        else:
            raise ValueError("Direction must be 'left' or 'right'")


def _get_BIC(
    n,
    wrss,
    k,
    alpha=1.0,
    use_correction=False,
):
    """
    Compute BIC with optional finite-sample correction.

    Standard BIC: n * log(wrss) + log(n) * (1 + alpha * (k - 1))
    Corrected BIC: adds a correction factor k*log(n) * (n / (n - k - 1)) to penalize
    complexity more heavily when n is small relative to k.
    """
    if wrss <= 0:
        return float("-inf")

    penalty = np.log(n) * (1 + alpha * (k - 1))

    if use_correction:
        # Avoid division by zero or negative denominator
        # If n <= k + 1, the model is definitely too complex for the data
        if n > k + 1:
            penalty *= n / (n - k - 1)
        else:
            penalty = float("inf")

    return n * np.log(wrss) + penalty

def _get_deviance_BIC(
    n: int,
    wrss: float,
    wrss_null: float,
    k: int,
) -> float:
    """Compute the deviance-based BIC for a given set of moments."""
    
    penalty = np.log(n) * k

    wrss_diff = wrss - wrss_null
    
    return wrss_diff + penalty


def _fit_con_node(
    moments,
    k_con,
    alpha=1.0,
    use_correction=False,
    deviance_bic=False,
):
    int_con = moments.sum_wy / moments.sum_w if moments.sum_w != 0 else 0.0
    slope_con = 0.0
    wrss_con = (
        moments.sum_wy2 + int_con**2 * moments.sum_w - 2 * int_con * moments.sum_wy
    )
    if (
        wrss_con <= 1e-10
    ):  # check for perfect fit, returns before unstable BIC calculation
        return int_con, slope_con, wrss_con, float("-inf")
    if deviance_bic:
        bic_con = _get_deviance_BIC(moments.n, wrss_con, moments.sum_wy2, k_con)
    else:
        bic_con = _get_BIC(moments.n, wrss_con, k_con, alpha, use_correction=use_correction)
    return int_con, slope_con, wrss_con, bic_con


def _fit_lin_node(
    moments,
    k_lin,
    alpha=1.0,
    use_correction=False,
    deviance_bic=False,
):
    # compute slope and intercept for linear model with numerical stability
    denominator = moments.sum_w * moments.sum_wx2 - moments.sum_wx**2
    slope_lin = (
        (moments.sum_w * moments.sum_wxy - moments.sum_wx * moments.sum_wy)
        / denominator
        if abs(denominator) > MIN_DENOM_VALUE
        else 0.0
    )
    int_lin = (
        (moments.sum_wy - moments.sum_wx * slope_lin) / moments.sum_w
        if moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )

    # Clip coefficients
    slope_lin = _clip_coefficient(slope_lin)
    int_lin = _clip_coefficient(int_lin)

    # compute weighted residual sum of squares
    wrss_lin = (
        moments.sum_wy2
        + int_lin**2 * moments.sum_w
        + slope_lin**2 * moments.sum_wx2
        + 2 * int_lin * slope_lin * moments.sum_wx
        - 2 * int_lin * moments.sum_wy
        - 2 * slope_lin * moments.sum_wxy
    )
    if (
        wrss_lin <= 1e-10
    ):  # check for perfect fit, returns before unstable BIC calculation
        return int_lin, slope_lin, wrss_lin, float("-inf")
    # compute weighted BIC
    if deviance_bic:
        bic_lin = _get_deviance_BIC(moments.n, wrss_lin, moments.sum_wy2, k_lin)
    else:
        bic_lin = _get_BIC(moments.n, wrss_lin, k_lin, alpha, use_correction=use_correction)
    return int_lin, slope_lin, wrss_lin, bic_lin


def _fit_truncated_lin_node(
    moments,
    x,
    y,
    w,
    k_lin,
    alpha=1.0,
    y_min=None,
    y_max=None,
    use_correction=False,
    deviance_bic=False,
):
    # Ensure consistency with Moments: clip x and y
    # This prevents the "WRSS > CON" anomaly where parameters are fit on clipped data
    # but WRSS is evaluated on unclipped data.
    x = np.clip(x, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)
    y = np.clip(y, -MAX_COEFF_VALUE, MAX_COEFF_VALUE)

    # compute slope and intercept for linear model
    slope_lin = (
        (moments.sum_w * moments.sum_wxy - moments.sum_wx * moments.sum_wy)
        / (moments.sum_w * moments.sum_wx2 - moments.sum_wx**2)
        if (moments.sum_w * moments.sum_wx2 - moments.sum_wx**2) != 0
        else 0.0
    )
    int_lin = (
        (moments.sum_wy - moments.sum_wx * slope_lin) / moments.sum_w
        if moments.sum_w != 0
        else 0.0
    )

    if slope_lin == 0:
        x_min = float("-inf")
        x_max = float("inf")
        # If slope is 0, we effectively have a constant model.
        # However, if int_lin is outside [y_min, y_max], the entire line is clipped.
        if int_lin < y_min:
            mask_clip_min = np.ones_like(x, dtype=bool)
            mask_clip_max = np.zeros_like(x, dtype=bool)
        elif int_lin > y_max:
            mask_clip_min = np.zeros_like(x, dtype=bool)
            mask_clip_max = np.ones_like(x, dtype=bool)
        else:
            mask_clip_min = np.zeros_like(x, dtype=bool)
            mask_clip_max = np.zeros_like(x, dtype=bool)
    else:
        # Calculate intersection points
        x_at_ymin = (y_min - int_lin) / slope_lin
        x_at_ymax = (y_max - int_lin) / slope_lin

        if slope_lin > 0:
            x_min = x_at_ymin
            x_max = x_at_ymax
            # For positive slope: y < y_min when x < x_min
            mask_clip_min = x < x_min
            # For positive slope: y > y_max when x > x_max
            mask_clip_max = x > x_max
        else:
            # For negative slope:
            # y_max is hit at lower x (left bound), y_min is hit at higher x (right bound)
            x_min = x_at_ymax
            x_max = x_at_ymin
            # For negative slope: y > y_max when x < x_min (left side)
            mask_clip_max = x < x_min
            # For negative slope: y < y_min when x > x_max (right side)
            mask_clip_min = x > x_max

    wrss_clipped_min = np.sum(w[mask_clip_min] * (y[mask_clip_min] - y_min) ** 2)
    wrss_clipped_max = np.sum(w[mask_clip_max] * (y[mask_clip_max] - y_max) ** 2)

    mask_middle = ~(mask_clip_min | mask_clip_max)
    wrss_middle = np.sum(
        w[mask_middle] * (y[mask_middle] - (int_lin + slope_lin * x[mask_middle])) ** 2
    )

    # compute weighted residual sum of squares
    wrss_lin = wrss_clipped_min + wrss_clipped_max + wrss_middle
    if (
        wrss_lin <= 1e-10
    ):  # check for perfect fit, returns before unstable BIC calculation
        return int_lin, slope_lin, wrss_lin, float("-inf"), x_min, x_max
    # compute weighted BIC
    if deviance_bic:
        bic_lin = _get_deviance_BIC(moments.n, wrss_lin, moments.sum_wy2, k_lin)
    else:
        bic_lin = _get_BIC(moments.n, wrss_lin, k_lin, alpha, use_correction=use_correction)
    return int_lin, slope_lin, wrss_lin, bic_lin, x_min, x_max


def _fit_pcon_node(
    left_moments,
    right_moments,
    k_pcon,
    alpha=1.0,
    use_correction=False,
    deviance_bic=False,
):
    # compute intercepts with protection against division by zero
    int_left_pcon = (
        left_moments.sum_wy / left_moments.sum_w
        if left_moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )
    int_right_pcon = (
        right_moments.sum_wy / right_moments.sum_w
        if right_moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )

    # Clip intercepts to prevent extreme values
    int_left_pcon = _clip_coefficient(int_left_pcon)
    int_right_pcon = _clip_coefficient(int_right_pcon)
    # compute weighted residual sum of squares
    wrss_pcon = (
        left_moments.sum_wy2
        + int_left_pcon**2 * left_moments.sum_w
        - 2 * int_left_pcon * left_moments.sum_wy
        + right_moments.sum_wy2
        + int_right_pcon**2 * right_moments.sum_w
        - 2 * int_right_pcon * right_moments.sum_wy
    )
    if (
        wrss_pcon <= 1e-10
    ):  # check for perfect fit, returns before unstable BIC calculation
        return int_left_pcon, int_right_pcon, wrss_pcon, float("-inf")
    # compute weighted BIC
    if deviance_bic:
        bic_pcon = _get_deviance_BIC(
            left_moments.n + right_moments.n,
            wrss_pcon,
            left_moments.sum_wy2 + right_moments.sum_wy2,
            k_pcon,
        )
    else:
        bic_pcon = _get_BIC(
            left_moments.n + right_moments.n,
            wrss_pcon,
            k_pcon,
            alpha,
            use_correction=use_correction,
        )
    return int_left_pcon, int_right_pcon, wrss_pcon, bic_pcon


def _fit_plin_node(
    left_moments, right_moments, k_plin, alpha=1.0, use_correction=False, deviance_bic=False,
):
    # compute slope and intercept with numerical stability
    left_denominator = (
        left_moments.sum_w * left_moments.sum_wx2 - left_moments.sum_wx**2
    )
    right_denominator = (
        right_moments.sum_w * right_moments.sum_wx2 - right_moments.sum_wx**2
    )

    slope_left_plin = (
        (
            left_moments.sum_w * left_moments.sum_wxy
            - left_moments.sum_wx * left_moments.sum_wy
        )
        / left_denominator
        if abs(left_denominator) > MIN_DENOM_VALUE
        else 0.0
    )
    slope_right_plin = (
        (
            right_moments.sum_w * right_moments.sum_wxy
            - right_moments.sum_wx * right_moments.sum_wy
        )
        / right_denominator
        if abs(right_denominator) > MIN_DENOM_VALUE
        else 0.0
    )

    # Clip coefficients to prevent extreme values
    slope_left_plin = _clip_coefficient(slope_left_plin)
    slope_right_plin = _clip_coefficient(slope_right_plin)
    int_left_plin = (
        (left_moments.sum_wy - slope_left_plin * left_moments.sum_wx)
        / left_moments.sum_w
        if left_moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )
    int_right_plin = (
        (right_moments.sum_wy - slope_right_plin * right_moments.sum_wx)
        / right_moments.sum_w
        if right_moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )

    # Clip intercepts as well
    int_left_plin = _clip_coefficient(int_left_plin)
    int_right_plin = _clip_coefficient(int_right_plin)
    # compute weighted residual sum of squares
    wrss_plin = (
        left_moments.sum_wy2
        + int_left_plin**2 * left_moments.sum_w
        + slope_left_plin**2 * left_moments.sum_wx2
        + 2 * int_left_plin * slope_left_plin * left_moments.sum_wx
        - 2 * int_left_plin * left_moments.sum_wy
        - 2 * slope_left_plin * left_moments.sum_wxy
        + right_moments.sum_wy2
        + int_right_plin**2 * right_moments.sum_w
        + slope_right_plin**2 * right_moments.sum_wx2
        + 2 * int_right_plin * slope_right_plin * right_moments.sum_wx
        - 2 * int_right_plin * right_moments.sum_wy
        - 2 * slope_right_plin * right_moments.sum_wxy
    )
    if (
        wrss_plin <= 1e-10
    ):  # check for perfect fit, returns before unstable BIC calculation
        return int_left_plin, slope_left_plin, int_right_plin, slope_right_plin, wrss_plin, float("-inf")
    # compute weighted BIC
    if deviance_bic:
        bic_plin = _get_deviance_BIC(
            left_moments.n + right_moments.n,
            wrss_plin,
            left_moments.sum_wy2 + right_moments.sum_wy2,
            k_plin,
        )
    else:
        bic_plin = _get_BIC(
            left_moments.n + right_moments.n,
            wrss_plin,
            k_plin,
            alpha,
            use_correction=use_correction,
        )
    return int_left_plin, slope_left_plin, int_right_plin, slope_right_plin, wrss_plin, bic_plin


def _fit_pconc_node(
    left_moments,
    right_moments,
    k_pconc,
    alpha=1.0,
    use_correction=False,
    deviance_bic=False,
):
    # compute intercepts with numerical stability
    int_left_pconc = (
        left_moments.sum_wy / left_moments.sum_w
        if left_moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )
    int_right_pconc = (
        right_moments.sum_wy / right_moments.sum_w
        if right_moments.sum_w > MIN_DENOM_VALUE
        else 0.0
    )

    # Clip intercepts to prevent extreme values
    int_left_pconc = _clip_coefficient(int_left_pconc)
    int_right_pconc = _clip_coefficient(int_right_pconc)
    # compute weighted residual sum of squares
    wrss_pconc = (
        left_moments.sum_wy2
        + int_left_pconc**2 * left_moments.sum_w
        - 2 * int_left_pconc * left_moments.sum_wy
        + right_moments.sum_wy2
        + int_right_pconc**2 * right_moments.sum_w
        - 2 * int_right_pconc * right_moments.sum_wy
    )
    if (
        wrss_pconc <= 1e-10
    ):  # check for perfect fit, returns before unstable BIC calculation
        return int_left_pconc, int_right_pconc, wrss_pconc, float("-inf")
    # compute weighted BIC
    if deviance_bic:
        bic_pconc = _get_deviance_BIC(
            left_moments.n + right_moments.n,
            wrss_pconc,
            left_moments.sum_wy2 + right_moments.sum_wy2,
            k_pconc,
        )
    else:
        bic_pconc = _get_BIC(
            left_moments.n + right_moments.n,
            wrss_pconc,
            k_pconc,
            alpha,
            use_correction=use_correction,
        )
    return int_left_pconc, int_right_pconc, wrss_pconc, bic_pconc


class Pilot(BaseEstimator):

    def __init__(
        self,
        max_depth=12,
        max_model_depth=30,
        split_criterion="BIC",
        min_samples_split=10,
        min_samples_leaf=5,
        min_deviance_improvement: float = 1e-5,
        df_settings: dict[str, int] | None = None,
        regression_nodes: list[str] | None = None,
        min_unique_values_regression=5,
        precision_scale=1e-10,
        alpha: float = 1.0,
        store_diagnostics: bool = False,
        allow_con_at_root: bool = False,
        learning_rate: float = 1.0,
        warm_start: str | None = None,
        max_node_contribution: float | None = 10.0,
        deviance_check_splits: bool = True,
        use_corrected_bic: bool = True,
        use_effective_sample_size: bool = True,
        min_eff_sample_pconc: int = 0,
        class_weight: str | dict | None = None,
        init: str = "zero",
        max_backtrack: int = 10,
        deviance_bic_accept: bool = True,
        min_hessian_weight: float | None = None,
    ) -> None:

        # initialize class attributes
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.split_criterion = split_criterion
        self.regression_nodes = (
            REGRESSION_NODES if regression_nodes is None else regression_nodes
        )
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_unique_values_regression = min_unique_values_regression
        # Stored under both names: the fitting code reads min_relative_wrss_lin,
        # while sklearn's get_params/clone requires an attribute matching the
        # constructor argument. Without this, clone() and any *SearchCV raise
        # AttributeError.
        self.min_deviance_improvement = min_deviance_improvement
        self.min_relative_wrss_lin = min_deviance_improvement
        self.precision_scale = precision_scale
        self.alpha = alpha

        # Starting value of the additive predictor f_0 when warm_start is None.
        #   "zero"    -- f_0 = 0, i.e. p_hat = 0.5 for every observation (default).
        #   "logodds" -- f_0 = log(p_bar / (1 - p_bar)), the base rate.
        #
        # "zero" is statistically wrong under class imbalance: at 0.17% positives
        # it starts the model 6.3 log-odds from the base rate, inflating the
        # deviance ~52x, and because the IRLS weights are w = p(1-p) it pins every
        # weight at its maximum of 0.25, so the effective sample size 4*sum(w)
        # degenerates to n.
        #
        # It is nonetheless still the default, because correcting it is not free:
        # the inflated n_eff subsidises high-df split nodes, so df_settings as
        # currently tuned is calibrated to the wrong scale. Measured on a 30k
        # subsample, switching to "logodds" takes give_me_some_credit from 19
        # splits to 6 and costs 0.14 PR-AUC, while gaining on taiwan_default and
        # on ULB. Adopting "logodds" as the default requires re-running the
        # df_settings grid search underneath it -- see
        # docs/fraud_benchmark_setup.md section 5.5.
        if init not in ("zero", "logodds"):
            raise ValueError(f"init must be 'zero' or 'logodds', got {init}")
        self.init = init

        # Backtracking limit for the safeguarded Newton step (see
        # _safeguarded_step). 0 restores the unsafeguarded behaviour, where a
        # node whose full step does not reduce the deviance is discarded.
        if max_backtrack < 0:
            raise ValueError(f"max_backtrack must be >= 0, got {max_backtrack}")
        self.max_backtrack = max_backtrack

        # Re-test a damped node on the realised reduction in the true deviance
        # against a BIC penalty, instead of trusting the surrogate BIC that
        # selected it at full step. Only has an effect when backtracking is on.
        self.deviance_bic_accept = deviance_bic_accept

        # attributes used for fitting
        self.X = None
        self.y = None
        self.logodds = None
        self.f_0 = None
        self.sorted_X_indices = None
        self.n_features = None
        self.max_features_considered = None
        self.cat_ids = None
        self.model_tree = None
        self.tree_depth = 0
        self.model_depth = 0
        self.debug = False

        # order of preference for regression nodes
        # this cannot be changed as best split relies on this specific order
        self.regression_nodes = [
            node for node in NODE_PREFERENCE_ORDER if node in self.regression_nodes
        ]

        # degrees of freedom for each regression node
        self.df_settings = DEFAULT_DF_SETTINGS.copy()  # preset default values
        if df_settings is not None:  # if new settings are provided update dict
            self.df_settings.update(df_settings)

        # diagnostic storage (only populated if store_diagnostics=True)
        self.store_diagnostics = store_diagnostics
        self.diagnostics: dict[int, list[BICCandidate]] = {}  # node_depth -> candidates

        # CON node control: disabled at root by default to prevent zero-signal termination
        # on balanced datasets where pseudo-response has zero mean
        self.allow_con_at_root = allow_con_at_root

        # Learning rate: scales down node updates to prevent WRSS/deviance mismatch
        # Default 1.0 for backward compatibility; lower values (e.g., 0.1) help on
        # datasets where the Newton-Raphson quadratic approximation breaks down
        self.learning_rate = learning_rate
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")

        # Warm start: initialize f_0 from a pre-fit model instead of constant logodds
        # Options: None (default), "logreg" (logistic regression warm start)
        self.warm_start = warm_start
        if warm_start is not None and warm_start not in ["logreg"]:
            raise ValueError(f"warm_start must be None or 'logreg', got {warm_start}")

        # Per-node contribution clipping: prevents extreme node updates from cascading
        # during training (extreme f → tiny Hessian → exploding pseudo-response) and
        # bounds predictions. Default 10.0 (sigmoid(10) ≈ 0.99995). Set to None to disable.
        self.max_node_contribution = max_node_contribution

        # Deviance check for split nodes (pcon, plin, pconc): prevents splits that
        # improve WRSS but worsen actual classification deviance
        self.deviance_check_splits = deviance_check_splits

        # Robustness features
        self.use_corrected_bic = use_corrected_bic
        self.use_effective_sample_size = use_effective_sample_size
        self.min_eff_sample_pconc = min_eff_sample_pconc
        if min_hessian_weight is not None and not (
            np.isfinite(min_hessian_weight) and 0 < min_hessian_weight <= 0.25
        ):
            raise ValueError("min_hessian_weight must be in (0, 0.25] or None")
        self.min_hessian_weight = min_hessian_weight

        # Class weighting: handle imbalanced datasets by reweighting samples
        # Options: None (no weighting), 'balanced' (inversely proportional to class frequency),
        # or dict {0: weight_0, 1: weight_1} for manual weights
        # Weights are computed globally and applied consistently to maintain a coherent loss function
        # for gradient boosting optimization.
        self.class_weight = class_weight
        self.sample_weights_ = None  # computed in fit()

    def best_split(
        node: Tree,
        model: Pilot,
        truncate_lin: bool = False,
        debug: bool = False,
    ):
        deviance_bic = model.split_criterion == "dBIC"
        
        # set up split
        if deviance_bic:
            split = Split(bic=0.0)
        else:    
            split = Split()

        # diagnostic storage for this node
        candidates: list[BICCandidate] = [] if model.store_diagnostics else None

        # compute gradient and hessian with numerical stability
        # Clip f_node to prevent overflow in exp
        f_clipped = np.clip(node.f_node, -500, 500)
        sigmoid_f = 1 / (1 + np.exp(-f_clipped))
        g_full = sigmoid_f - model.y  # gradient
        h_full = sigmoid_f * (1 - sigmoid_f)  # hessian
        if model.min_hessian_weight is not None:
            h_full = np.maximum(h_full, model.min_hessian_weight)

        # setup transformed response variable and weights for binary classification
        # protect against division by zero in locations where h_full == 0
        pseudo_resp = np.zeros_like(node.f_node)
        nonzero_mask = h_full != 0
        pseudo_resp[nonzero_mask] = -g_full[nonzero_mask] / h_full[nonzero_mask]
        pseudo_resp[~nonzero_mask] = 0.0
        # Apply global class weights to hessian weights
        w = h_full * model.sample_weights_

        # CON
        # Skip CON at root if not allowed (prevents zero-signal termination on balanced data)
        is_root = node.model_depth is None or node.model_depth == 0
        evaluate_con = "con" in model.regression_nodes and (
            not is_root or model.allow_con_at_root
        )

        if is_root and "con" in model.regression_nodes and not model.allow_con_at_root:
            logger.debug("Skipping CON at root node (allow_con_at_root=False)")

        if evaluate_con:
            moments_con = Moments(
                x=np.zeros_like(pseudo_resp[node.obs_ids]),
                y=pseudo_resp[node.obs_ids],
                w=w[node.obs_ids],
            )

            # Apply effective sample size if enabled
            # n_eff = 4 * sum(w) is a proxy for the information-carrying sample size
            # in binary classification (where w_max = 0.25)
            if model.use_effective_sample_size:
                moments_con.n = max(1.0, 4 * np.sum(w[node.obs_ids]))

            int_con, slope_con, wrss_con, bic_con = _fit_con_node(
                moments_con,
                model.df_settings["con"],
                model.alpha,
                use_correction=model.use_corrected_bic,
                deviance_bic=deviance_bic,
            )

            split_con = Split(
                node="con",
                feature=-1,
                pivot=float("nan"),
                slope_left=slope_con,
                intercept_left=int_con,
                slope_right=0.0,
                intercept_right=0.0,
                interval=np.array([-np.inf, np.inf]),
                pivot_categorical=[0],
                wrss=wrss_con,
                bic=bic_con,
            )

            if model.store_diagnostics:
                candidates.append(
                    BICCandidate(
                        node_type="con",
                        feature_id=-1,
                        pivot=float("nan"),
                        bic=bic_con,
                        wrss=wrss_con,
                        intercept_left=int_con,
                        slope_left=slope_con,
                        intercept_right=0.0,
                        slope_right=0.0,
                        n_left=moments_con.n,
                        n_right=0,
                    )
                )

            if split_con.bic == float("-inf"):
                logger.debug("Perfect fit found for CON node.")
                return split_con

            if split.is_improved(split_con):
                logger.debug(f"Improved split found: CON node with BIC {bic_con:.6f}")
                split = split_con

        for feature_id in range(model.X.shape[1]):
            if (model.cat_ids is None) or (feature_id not in model.cat_ids):
                x_ids_sorted = model.sorted_X_indices[feature_id].reshape(
                    -1,
                )
                obs_ids_sorted = x_ids_sorted[np.isin(x_ids_sorted, node.obs_ids)]
                x_sorted = (
                    model.X[obs_ids_sorted, feature_id]
                    .copy()
                    .reshape(
                        -1,
                    )
                )
                pseudo_resp_sorted = pseudo_resp[obs_ids_sorted].copy()
                w_sorted = w[obs_ids_sorted].copy()

                right_moments = Moments(x=x_sorted, y=pseudo_resp_sorted, w=w_sorted)

                if model.use_effective_sample_size:
                    right_moments.n = max(1.0, 4 * np.sum(w_sorted))

                # LIN
                if "lin" in model.regression_nodes:
                    if truncate_lin:
                        int_lin, slope_lin, wrss_lin, bic_lin, x_min, x_max = (
                            _fit_truncated_lin_node(
                                right_moments,
                                x_sorted,
                                pseudo_resp_sorted,
                                w_sorted,
                                model.df_settings["lin"],
                                model.alpha,
                                y_min=min(pseudo_resp),
                                y_max=max(pseudo_resp),
                                use_correction=model.use_corrected_bic,
                                deviance_bic=deviance_bic,
                            )
                        )
                    else:
                        int_lin, slope_lin, wrss_lin, bic_lin = _fit_lin_node(
                            right_moments,
                            model.df_settings["lin"],
                            model.alpha,
                            use_correction=model.use_corrected_bic,
                            deviance_bic=deviance_bic,
                        )
                        x_min = x_sorted[0]
                        x_max = x_sorted[-1]

                    split_lin = Split(
                        node=f"{truncate_lin*"t"}lin",
                        feature=feature_id,
                        pivot=float("nan"),
                        slope_left=slope_lin,
                        intercept_left=int_lin,
                        slope_right=0.0,
                        intercept_right=0.0,
                        interval=np.array([x_min, x_max]),
                        pivot_categorical=[0],
                        wrss=wrss_lin,
                        bic=bic_lin,
                    )

                    if model.store_diagnostics:
                        candidates.append(
                            BICCandidate(
                                node_type=f"{truncate_lin*'t'}lin",
                                feature_id=feature_id,
                                pivot=float("nan"),
                                bic=bic_lin,
                                wrss=wrss_lin,
                                intercept_left=int_lin,
                                slope_left=slope_lin,
                                intercept_right=0.0,
                                slope_right=0.0,
                                n_left=right_moments.n,
                                n_right=0,
                            )
                        )

                    if split_lin.bic == float("-inf"):
                        logger.debug("Perfect fit found for LIN node.")
                        return split_lin

                    if split.is_improved(split_lin):
                        logger.debug(
                            f"Improved split found: {'TLIN' if truncate_lin else 'LIN'} node with BIC {bic_lin:.6f}"
                        )
                        split = split_lin

                # setup for piecewise models
                possible_pivots = np.unique(x_sorted)
                left_moments = Moments()

                for pivot_id in range(
                    len(possible_pivots) - 1
                ):  # exclude the max value, as then we get same as con or lin
                    pivot = possible_pivots[pivot_id]
                    index_pivot = np.where(x_sorted == pivot)
                    x_pivot = x_sorted[index_pivot].reshape(
                        -1,
                    )
                    w_pivot = w_sorted[index_pivot].reshape(
                        -1,
                    )
                    pseudo_resp_pivot = pseudo_resp_sorted[index_pivot].reshape(
                        -1,
                    )
                    left_moments.update_pivot(
                        x_pivot, pseudo_resp_pivot, w_pivot, moment_direction="left"
                    )
                    right_moments.update_pivot(
                        x_pivot, pseudo_resp_pivot, w_pivot, moment_direction="right"
                    )

                    if left_moments.n < model.min_samples_leaf:
                        continue  # not enough samples in left node, try next pivot
                    elif right_moments.n < model.min_samples_leaf:
                        break  # not enough samples in right node, stop searching for pivots

                    if model.use_effective_sample_size:
                        # For piecewise nodes, moments.n is re-calculated inside fit
                        # but we need to ensure the sum of n_eff is used for penalty
                        left_n_eff = max(1.0, 4 * left_moments.sum_w)
                        right_n_eff = max(1.0, 4 * right_moments.sum_w)
                        # We temporarily override n for the BIC calculation
                        original_left_n, original_right_n = (
                            left_moments.n,
                            right_moments.n,
                        )
                        left_moments.n, right_moments.n = left_n_eff, right_n_eff

                    # PCON
                    if "pcon" in model.regression_nodes:
                        int_left_pcon, int_right_pcon, wrss_pcon, bic_pcon = (
                            _fit_pcon_node(
                                left_moments,
                                right_moments,
                                model.df_settings["pcon"],
                                model.alpha,
                                use_correction=model.use_corrected_bic,
                                deviance_bic=deviance_bic,
                            )
                        )

                        split_pcon = Split(
                            node="pcon",
                            feature=feature_id,
                            pivot=pivot,
                            slope_left=0.0,
                            intercept_left=int_left_pcon,
                            slope_right=0.0,
                            intercept_right=int_right_pcon,
                            interval=np.array(
                                [possible_pivots[0], possible_pivots[-1]]
                            ),
                            pivot_categorical=[0],
                            wrss=wrss_pcon,
                            bic=bic_pcon,
                        )

                        if model.store_diagnostics:
                            candidates.append(
                                BICCandidate(
                                    node_type="pcon",
                                    feature_id=feature_id,
                                    pivot=pivot,
                                    bic=bic_pcon,
                                    wrss=wrss_pcon,
                                    intercept_left=int_left_pcon,
                                    slope_left=0.0,
                                    intercept_right=int_right_pcon,
                                    slope_right=0.0,
                                    n_left=left_moments.n,
                                    n_right=right_moments.n,
                                )
                            )

                        if split_pcon.bic == float("-inf"):
                            logger.debug("Perfect fit found for PCON node.")
                            if model.use_effective_sample_size:
                                left_moments.n, right_moments.n = (
                                    original_left_n,
                                    original_right_n,
                                )
                            return split_pcon

                        if split.is_improved(split_pcon):
                            logger.debug(
                                f"Improved split found: PCON node with BIC {bic_pcon:.6f}"
                            )
                            split = split_pcon

                    # PLIN
                    if pivot_id < model.min_unique_values_regression:
                        if model.use_effective_sample_size:
                            left_moments.n, right_moments.n = (
                                original_left_n,
                                original_right_n,
                            )
                        continue
                    if (
                        len(possible_pivots) - pivot_id
                        < model.min_unique_values_regression
                    ):
                        if model.use_effective_sample_size:
                            left_moments.n, right_moments.n = (
                                original_left_n,
                                original_right_n,
                            )
                        break
                    if "plin" in model.regression_nodes:
                        (
                            int_left_plin,
                            slope_left_plin,
                            int_right_plin,
                            slope_right_plin,
                            wrss_plin,
                            bic_plin,
                        ) = _fit_plin_node(
                            left_moments,
                            right_moments,
                            model.df_settings["plin"],
                            model.alpha,
                            use_correction=model.use_corrected_bic,
                            deviance_bic=deviance_bic,
                        )

                        split_plin = Split(
                            node="plin",
                            feature=feature_id,
                            pivot=pivot,
                            slope_left=slope_left_plin,
                            intercept_left=int_left_plin,
                            slope_right=slope_right_plin,
                            intercept_right=int_right_plin,
                            interval=np.array(
                                [possible_pivots[0], possible_pivots[-1]]
                            ),
                            pivot_categorical=[0],
                            wrss=wrss_plin,
                            bic=bic_plin,
                        )

                        if model.store_diagnostics:
                            candidates.append(
                                BICCandidate(
                                    node_type="plin",
                                    feature_id=feature_id,
                                    pivot=pivot,
                                    bic=bic_plin,
                                    wrss=wrss_plin,
                                    intercept_left=int_left_plin,
                                    slope_left=slope_left_plin,
                                    intercept_right=int_right_plin,
                                    slope_right=slope_right_plin,
                                    n_left=left_moments.n,
                                    n_right=right_moments.n,
                                )
                            )

                        if split_plin.bic == float("-inf"):
                            logger.debug("Perfect fit found for PLIN node.")
                            if model.use_effective_sample_size:
                                left_moments.n, right_moments.n = (
                                    original_left_n,
                                    original_right_n,
                                )
                            return split_plin

                        if split.is_improved(split_plin):
                            logger.debug(
                                f"Improved split found: PLIN node with BIC {bic_plin:.6f}"
                            )
                            split = split_plin

                    # Restore original n values if they were overridden
                    if model.use_effective_sample_size:
                        left_moments.n, right_moments.n = (
                            original_left_n,
                            original_right_n,
                        )

                continue  # end of numeric feature loop, go to next feature

            elif model.cat_ids is not None and feature_id in model.cat_ids:
                # For categorical features, we need to sort by feature values
                x_ids_sorted = model.sorted_X_indices[feature_id].reshape(
                    -1,
                )
                obs_ids_sorted = x_ids_sorted[np.isin(x_ids_sorted, node.obs_ids)]
                x_sorted = model.X[obs_ids_sorted, feature_id]
                pseudo_resp_sorted = pseudo_resp[obs_ids_sorted].copy()
                w_sorted = w[obs_ids_sorted].copy()

                # group response by category
                cats = np.unique(x_sorted)
                pseudo_resp_mean = []
                for cat in cats:
                    obs_cat = pseudo_resp_sorted[x_sorted == cat]
                    weights_cat = w_sorted[x_sorted == cat]
                    (
                        pseudo_resp_mean.append(
                            np.sum(weights_cat * obs_cat) / np.sum(weights_cat)
                        )
                        if np.sum(weights_cat) != 0
                        else pseudo_resp_mean.append(0.0)
                    )
                # sort categories by mean response
                cats_index_sorted = np.argsort(pseudo_resp_mean)
                cats_sorted = cats[cats_index_sorted]

                left_moments = Moments()
                right_moments = Moments(
                    np.zeros_like(pseudo_resp_sorted), pseudo_resp_sorted, w_sorted
                )

                # Skip PCONC for this node if its effective sample size is below threshold.
                # This prevents overfitting at small-n nodes where pconc=1.1 accepts spurious splits.
                # Only checked when min_eff_sample_pconc > 0 (disabled by default).
                if model.min_eff_sample_pconc > 0:
                    node_n_eff = 4 * np.sum(w_sorted)
                    if node_n_eff < model.min_eff_sample_pconc:
                        logger.debug(
                            f"Skipping PCONC for feature {feature_id}: "
                            f"n_eff={node_n_eff:.1f} < threshold={model.min_eff_sample_pconc}"
                        )
                        continue

                for cat in cats_sorted[:-1]:  # exclude last category as pivot
                    # get values for this category
                    x_cat, pr_cat, w_cat = (
                        np.zeros_like(pseudo_resp_sorted[x_sorted == cat]),
                        pseudo_resp_sorted[x_sorted == cat],
                        w_sorted[x_sorted == cat],
                    )
                    # update moments
                    left_moments.update_pivot(
                        x_cat, pr_cat, w_cat, moment_direction="left"
                    )
                    right_moments.update_pivot(
                        x_cat, pr_cat, w_cat, moment_direction="right"
                    )
                    # check minimum samples in leaf
                    if left_moments.n < model.min_samples_leaf:
                        continue
                    elif right_moments.n < model.min_samples_leaf:
                        break

                    if model.use_effective_sample_size:
                        left_n_eff = max(1.0, 4 * left_moments.sum_w)
                        right_n_eff = max(1.0, 4 * right_moments.sum_w)
                        original_left_n, original_right_n = (
                            left_moments.n,
                            right_moments.n,
                        )
                        left_moments.n, right_moments.n = left_n_eff, right_n_eff

                    int_left_pconc, int_right_pconc, wrss_pconc, bic_pconc = (
                        _fit_pconc_node(
                            left_moments,
                            right_moments,
                            model.df_settings["pconc"],
                            model.alpha,
                            use_correction=model.use_corrected_bic,
                            deviance_bic=deviance_bic,
                        )
                    )
                    # get pivot categories
                    idx_array = np.where(cats_sorted == cat)[
                        0
                    ]  # array of matching indices
                    idx = idx_array[0]  # first and only match
                    pivot_categorical = cats_sorted[
                        : idx + 1
                    ]  # all categories up to and including the pivot

                    split_pconc = Split(
                        node="pconc",
                        feature=feature_id,
                        pivot=cat,
                        slope_left=0.0,
                        intercept_left=int_left_pconc,
                        slope_right=0.0,
                        intercept_right=int_right_pconc,
                        interval=np.array(
                            [cats[0], cats[-1]]
                        ),  # use category range instead of possible_pivots
                        pivot_categorical=pivot_categorical,
                        wrss=wrss_pconc,
                        bic=bic_pconc,
                    )

                    if model.store_diagnostics:
                        candidates.append(
                            BICCandidate(
                                node_type="pconc",
                                feature_id=feature_id,
                                pivot=cat,
                                bic=bic_pconc,
                                wrss=wrss_pconc,
                                intercept_left=int_left_pconc,
                                slope_left=0.0,
                                intercept_right=int_right_pconc,
                                slope_right=0.0,
                                n_left=left_moments.n,
                                n_right=right_moments.n,
                            )
                        )

                    if split_pconc.bic == float("-inf"):
                        logger.debug("Perfect fit found for PCONC node.")
                        if model.use_effective_sample_size:
                            left_moments.n, right_moments.n = (
                                original_left_n,
                                original_right_n,
                            )
                        return split_pconc

                    if split.is_improved(split_pconc):
                        logger.debug(
                            f"Improved split found: PCONC node with BIC {bic_pconc:.6f}"
                        )
                        split = split_pconc

                    if model.use_effective_sample_size:
                        left_moments.n, right_moments.n = (
                            original_left_n,
                            original_right_n,
                        )

        # Store diagnostics for this node if enabled
        # Use model_depth instead of tree_depth since lin/tlin don't increment tree_depth
        if model.store_diagnostics and candidates:
            diag_depth = node.model_depth if node.model_depth is not None else 0
            if diag_depth not in model.diagnostics:
                model.diagnostics[diag_depth] = []
            model.diagnostics[diag_depth].extend(candidates)

        return split

    def max_depth_reached(self, tree_depth, model_depth):
        if tree_depth >= self.max_depth:
            return True
        if model_depth >= self.max_model_depth:
            return True
        return False

    def min_samples_reached(self, indices):
        if len(indices) < self.min_samples_split:
            return True
        return False

    def min_deviance_reached(self, f_parent, f_current):
        # Numerically stable deviance computation using log-sum-exp trick
        def stable_deviance(f, y):
            f_clipped = np.clip(f, -500, 500)
            log_term = np.where(
                f_clipped > 0,
                f_clipped + np.log1p(np.exp(-f_clipped)),
                np.log1p(np.exp(f_clipped)),
            )
            return 2 * np.sum(log_term - y * f_clipped)

        deviance_parent = stable_deviance(f_parent, self.y)
        deviance_current = stable_deviance(f_current, self.y)
        # The null model is wherever the fit actually started, not an assumed
        # base-rate intercept. Previously this was hardcoded to self.logodds
        # while f_0 was zero, so dev_explained was measured against a baseline
        # the model never occupied -- and under class imbalance the two differ
        # by a factor of ~50 in deviance.
        null_deviance = stable_deviance(self.f_0, self.y)

        logger.debug(
            f"Deviance check: parent={deviance_parent:.2f}, current={deviance_current:.2f}, "
            f"null={null_deviance:.2f}, f_current range=[{f_current.min():.2f}, {f_current.max():.2f}]"
        )

        # Compute the relative improvement
        rel_improvement = (
            (deviance_parent - deviance_current) / deviance_parent
            if deviance_parent != 0
            else 0.0
        )
        if rel_improvement < self.min_relative_wrss_lin:
            logger.debug(
                f"STOP: Relative deviance improvement ({rel_improvement:.6f}) below threshold ({self.min_relative_wrss_lin})"
            )
            return True

        # Compute the deviance explained
        dev_explained = (
            1 - (deviance_current / null_deviance) if null_deviance != 0 else 0.0
        )
        if dev_explained > 0.999:
            logger.debug(
                f"STOP: Deviance explained ({dev_explained:.6f}) exceeds 99.9%"
            )
            return True
        return False

    def _deviance(self, f):
        """Binomial deviance of the additive predictor f against self.y."""
        f_clipped = np.clip(f, -500, 500)
        log_term = np.where(
            f_clipped > 0,
            f_clipped + np.log1p(np.exp(-f_clipped)),
            np.log1p(np.exp(f_clipped)),
        )
        return 2 * np.sum(log_term - self.y * f_clipped)

    def _safeguarded_step(self, f_parent, build_f, df=1.0, obs_ids=None):
        """
        Take the largest damped Newton step that actually reduces the deviance.

        PILOT selects nodes on a second-order (IRLS) surrogate but accepts them
        on the true deviance. The two agree only while the quadratic
        approximation is good, and under extreme class imbalance it is not: at a
        0.17% base rate the working response is -1.0 for negatives and +546 for
        positives, and the resulting full step spans -90 to +245 log-odds,
        raising the deviance ~31-fold. The unsafeguarded code discarded such a
        node and terminated the branch, so the tree could end up empty on data
        a plain logistic regression fits at 0.98 AUC.

        Standard remedy: backtrack. Try the full learning_rate, and while the
        step fails to improve the deviance, halve it. This is a strict extension
        of the previous behaviour -- when the full step already improves the
        deviance the first trial succeeds and nothing changes, and setting
        max_backtrack=0 restores the old discard-on-failure semantics exactly.

        Damping alone is not sound, though. The node was selected because its
        *undamped* BIC beat the alternatives on the surrogate; a step taken at
        1/64 of that explains far less deviance than the criterion credited it
        with, so accepting it inserts a node the penalty never paid for. Once a
        damped step is found, therefore, re-test it on the quantity that
        actually matters -- the realised reduction in the true deviance -- against
        a BIC penalty at the node's effective sample size:

            accept iff  D(f_parent) - D(f_new)  >  alpha * df * log(n_eff)

        This is the "select on the surrogate, accept on the deviance" rule. It
        is what keeps backtracking from turning rejected nodes into weak ones:
        without it, enabling backtracking costs 0.10 PR-AUC on german_credit.

        `build_f(scale)` must return the candidate f for a given step scale.
        Returns (f, scale), or (None, 0.0) if the node should be discarded.
        """
        scale = self.learning_rate
        f, damped = None, False
        for attempt in range(self.max_backtrack + 1):
            candidate = build_f(scale)
            if not self.min_deviance_reached(f_parent, candidate):
                f, damped = candidate, attempt > 0
                if damped:
                    logger.debug(
                        f"Backtracked {attempt} time(s): step scaled to {scale:.5g}"
                    )
                break
            scale /= 2.0

        if f is None:
            logger.debug(f"STOP: no step down to {scale * 2:.5g} reduced the deviance")
            return None, 0.0

        # Only a *damped* step needs re-justifying. At the full step the node is
        # being applied exactly as the surrogate BIC evaluated it, so that
        # approval stands and behaviour is bit-identical to the unsafeguarded
        # code -- which is what makes max_backtrack=0 an exact fallback.
        if damped and self.deviance_bic_accept and obs_ids is not None:
            delta = self._deviance(f_parent) - self._deviance(f)
            p_hat = 1.0 / (1.0 + np.exp(-np.clip(f_parent[obs_ids], -500, 500)))
            n_eff = max(2.0, 4.0 * np.sum(p_hat * (1.0 - p_hat)))
            threshold = self.alpha * df * np.log(n_eff)
            if delta <= threshold:
                logger.debug(
                    f"STOP: realised deviance gain {delta:.4f} does not clear the "
                    f"BIC threshold {threshold:.4f} (df={df}, n_eff={n_eff:.1f})"
                )
                return None, 0.0

        return f, scale

    def build_tree(self, node: Tree = None) -> Tree:
        if node is None:
            # initial call to build the tree, this is the root node
            node = Tree(
                split=Split(),
                f_node=self.f_0,
                depth=0,
                model_depth=0,
                obs_ids=np.arange(self.X.shape[0]),
            )
            truncate_lin = True
            model_depth = 0
            tree_depth = 0
        else:
            truncate_lin = False

        tree_depth = node.depth + 1
        model_depth = node.model_depth + 1

        split = Pilot.best_split(
            node=node,
            model=self,
            truncate_lin=truncate_lin,
            debug=self.debug,
        )
        logger.debug(
            f"Depth {model_depth}: Selected {split.node.upper()} on feature {split.feature} "
            f"(BIC={split.bic:.2f}, WRSS={split.wrss:.4f})"
        )

        if split.node == "":
            logger.debug("STOP: No valid split found")
            return Tree(split=Split(node="END", wrss=split.wrss))
        elif split.node in ["con", "lin", "tlin"]:
            # do not include 'lin' and 'con' in the depth calculation
            tree_depth -= 1

        # stop building the tree if stop criterion is met
        if self.max_depth_reached(tree_depth=tree_depth, model_depth=model_depth):
            logger.debug(
                f"STOP: Max depth reached (tree_depth={tree_depth}, model_depth={model_depth})"
            )
            return Tree(split=Split(node="END", wrss=split.wrss))

        if self.min_samples_reached(indices=node.obs_ids):
            logger.debug(f"STOP: Insufficient samples (n={len(node.obs_ids)})")
            return Tree(split=Split(node="END", wrss=split.wrss))

        # Update recorded max depths only for nodes that will actually be used
        self.tree_depth = max(self.tree_depth, tree_depth)
        self.model_depth = max(self.model_depth, model_depth)

        node = Tree(
            split=split,
            f_node=node.f_node,
            depth=tree_depth,
            model_depth=model_depth,
            obs_ids=node.obs_ids,
        )

        if split.bic == float("-inf"):
            node.left = Tree(split=Split(node="END", wrss=split.wrss))
            node.right = Tree(split=Split(node="END", wrss=split.wrss))
            logger.debug("STOP: Perfect fit achieved (BIC=-inf)")
            return node
        
        if split.node == "":
            logger.debug("STOP: No improving split found")
            return Tree(split=Split(node="END"), bic=0)

        elif split.node == "con":
            # update only the entries corresponding to this node's observations
            f = node.f_node.copy()
            update = self.learning_rate * split.intercept_left
            f[node.obs_ids] += self._clip_update(update)
            # # stop recursion - CON is always terminal
            # logger.debug(
            #     f"STOP: CON node selected (intercept={split.intercept_left:.4f})"
            # )
            node.left = self.build_tree(
                node=Tree(
                    split=split,
                    f_node=f,
                    depth=tree_depth,
                    model_depth=model_depth,
                    obs_ids=node.obs_ids,
                )
            )

        elif split.node == "lin":
            # update only the entries corresponding to this node's observations
            def _build_lin(scale):
                f = node.f_node.copy()
                update = scale * (
                    split.intercept_left
                    + split.slope_left * self.X[node.obs_ids, split.feature]
                )
                f[node.obs_ids] += self._clip_update(update)
                return f

            f, _ = self._safeguarded_step(
                node.f_node, _build_lin,
                df=self.df_settings["lin"], obs_ids=node.obs_ids,
            )
            if f is None:
                # No damped step improved the deviance - discard this node
                return Tree(split=Split(node="END", wrss=split.wrss))
            # continue recursion
            node.left = self.build_tree(
                node=Tree(
                    split=split,
                    f_node=f,
                    depth=tree_depth,
                    model_depth=model_depth,
                    obs_ids=node.obs_ids,
                )
            )

        elif split.node == "tlin":
            # update only the entries corresponding to this node's observations
            mask_left = self.X[node.obs_ids, split.feature] < split.interval[0]
            mask_right = self.X[node.obs_ids, split.feature] > split.interval[1]
            middle_mask = ~(mask_left | mask_right)

            def _build_tlin(scale):
                f = node.f_node.copy()
                update_left = scale * (
                    split.intercept_left + split.slope_left * split.interval[0]
                )
                update_right = scale * (
                    split.intercept_left + split.slope_left * split.interval[1]
                )
                f[node.obs_ids[mask_left]] += self._clip_update(update_left)
                f[node.obs_ids[mask_right]] += self._clip_update(update_right)
                update_middle = scale * (
                    split.intercept_left
                    + split.slope_left
                    * self.X[node.obs_ids[middle_mask], split.feature]
                )
                f[node.obs_ids[middle_mask]] += self._clip_update(update_middle)
                return f

            f, _ = self._safeguarded_step(
                node.f_node, _build_tlin,
                df=self.df_settings["lin"], obs_ids=node.obs_ids,
            )
            if f is None:
                # No damped step improved the deviance - discard this node
                return Tree(split=Split(node="END", wrss=split.wrss))
            # continue recursion
            node.left = self.build_tree(
                node=Tree(
                    split=split,
                    f_node=f,
                    depth=tree_depth,
                    model_depth=model_depth,
                    obs_ids=node.obs_ids,
                )
            )

        else:
            if split.node == "pconc":
                cond = np.isin(
                    self.X[node.obs_ids, split.feature], split.pivot_categorical
                )
            else:
                cond = self.X[node.obs_ids, split.feature] <= split.pivot

            # get indices for left and right child nodes
            indices_left = node.obs_ids[cond].astype(int)
            indices_right = node.obs_ids[~cond].astype(int)

            # construct full-length f vectors for children, updating only their indices
            def _build_split(scale):
                f = node.f_node.copy()
                update_left = scale * (
                    split.intercept_left
                    + split.slope_left * self.X[indices_left, split.feature]
                )
                update_right = scale * (
                    split.intercept_right
                    + split.slope_right * self.X[indices_right, split.feature]
                )
                f[indices_left] += self._clip_update(update_left)
                f[indices_right] += self._clip_update(update_right)
                return f

            if self.deviance_check_splits:
                f, _ = self._safeguarded_step(
                    node.f_node, _build_split,
                    df=self.df_settings.get(split.node, 2.0),
                    obs_ids=node.obs_ids,
                )
                if f is None:
                    logger.debug(
                        f"STOP: Split node ({split.node}) deviance check failed "
                        "at every damped step - discarding split"
                    )
                    node.left = Tree(split=Split(node="END", wrss=split.wrss))
                    node.right = Tree(split=Split(node="END", wrss=split.wrss))
                    return node
            else:
                f = _build_split(self.learning_rate)

            f_left = node.f_node.copy()
            f_right = node.f_node.copy()
            f_left[indices_left] = f[indices_left]
            f_right[indices_right] = f[indices_right]

            if (
                split.wrss < self.precision_scale
            ):  # check if wrss is sufficiently small to stop
                logger.debug(
                    f"STOP: WRSS ({split.wrss:.6f}) below precision threshold ({self.precision_scale})"
                )
                node.left = Tree(split=Split(node="END", wrss=split.wrss))
                node.right = Tree(split=Split(node="END", wrss=split.wrss))
                return node
            # else continue recursion
            try:
                node.left = self.build_tree(
                    node=Tree(
                        split=split,
                        f_node=f_left,
                        depth=tree_depth,
                        model_depth=model_depth,
                        obs_ids=indices_left,
                    )
                )

                node.right = self.build_tree(
                    node=Tree(
                        split=split,
                        f_node=f_right,
                        depth=tree_depth,
                        model_depth=model_depth,
                        obs_ids=indices_right,
                    )
                )

            except RecursionError as re:
                raise Exception from re
        return node

    def fit(
        self,
        X,
        y,
        cat_ids: list[int] | None = None,
        debug=False,
        **kwargs,
    ):
        self.cat_ids = cat_ids
        # Clear diagnostics from previous fits
        self.diagnostics = {}
        # Reset depth tracking (prevents accumulation across multiple fit() calls)
        self.tree_depth = 0
        self.model_depth = 0

        logger.debug(f"Starting PILOT fit: {X.shape[0]} samples, {X.shape[1]} features")

        # check if amount of observations match
        assert X.shape[0] == y.shape[0]

        # switch pandas df to numpy
        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)

        if isinstance(y, pd.core.frame.DataFrame):
            y = np.array(y)
        elif (
            y.ndim == 1
        ):  # if format of array is wrong, reshape into array containing n elements
            y = y.reshape(
                -1,
            )

        self.n_features = X.shape[1]
        n_samples = X.shape[0]
        self.X = X
        self.debug = debug

        sorted_indices = np.array(
            [
                np.argsort(self.X[:, feature_id], axis=0).flatten()
                for feature_id in range(self.n_features)
            ]
        )
        # get sorted indices
        self.sorted_X_indices = np.arange(0, n_samples, dtype=int)[sorted_indices]

        self.y = y
        # compute logodds
        self.logodds = (
            np.log(y.mean() / (1.0 - y.mean())) if y.mean() not in [0, 1] else 0.0
        )

        # Compute class weights globally (applied consistently across all nodes)
        # This defines the effective loss function that gradient boosting minimizes
        if self.class_weight is None:
            self.sample_weights_ = np.ones(n_samples)
        elif self.class_weight == "balanced":
            # Compute balanced class weights: n_samples / (n_classes * np.bincount(y))
            class_counts = np.bincount(y.astype(int), minlength=2)
            n_classes = 2
            class_weights = n_samples / (n_classes * class_counts)
            self.sample_weights_ = class_weights[y.astype(int)]
            logger.debug(
                f"Global balanced class weights: class 0={class_weights[0]:.4f}, "
                f"class 1={class_weights[1]:.4f} (distribution: {class_counts})"
            )
        elif isinstance(self.class_weight, dict):
            # Manual class weights provided
            self.sample_weights_ = np.array([self.class_weight[int(yi)] for yi in y])
            logger.debug(f"Manual class weights applied: {self.class_weight}")
        else:
            raise ValueError(
                f"class_weight must be None, 'balanced', or a dict, got {self.class_weight}"
            )

        # Initialize f_0 based on warm_start option
        if self.warm_start == "logreg":
            # Warm start with logistic regression
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            logger.debug("Warm start: fitting logistic regression for initialization")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit LR with reasonable settings
            lr = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
            lr.fit(X_scaled, y)

            # Compute f_0 from LR predictions (log-odds scale)
            # Store scaler for potential use in prediction
            self._warm_start_scaler = scaler
            self._warm_start_lr = lr

            f_lr = X_scaled @ lr.coef_.T.flatten() + lr.intercept_[0]
            # Clip to prevent numerical issues
            self.f_0 = np.clip(f_lr, -10, 10)

            logger.debug(
                f"Warm start initialized: f_0 range=[{self.f_0.min():.2f}, {self.f_0.max():.2f}]"
            )
        elif self.init == "logodds":
            self.f_0 = np.ones(len(y)) * self.logodds
        else:
            self.f_0 = np.zeros(len(y))

        self.model_tree = self.build_tree()

        logger.debug(
            f"Fit complete: tree_depth={self.tree_depth}, model_depth={self.model_depth}"
        )
        return

    def _scale_and_clip(self, contribution):
        """Apply learning_rate scaling and optional contribution clipping (for prediction)."""
        contribution = self.learning_rate * contribution
        if self.max_node_contribution is not None:
            contribution = np.clip(
                contribution, -self.max_node_contribution, self.max_node_contribution
            )
        return contribution

    def _clip_update(self, update):
        """Apply optional contribution clipping to a training update (already scaled by learning_rate)."""
        if self.max_node_contribution is not None:
            update = np.clip(
                update, -self.max_node_contribution, self.max_node_contribution
            )
        return update

    def predict_proba(self, X, model=None, maxd=np.inf, **kwargs):
        y_hat = []
        if model is None:
            model_tree = self.model_tree
        else:
            model_tree = model

        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)

        # Compute baseline predictions based on warm_start option
        # For warm_start='logreg', use LogReg predictions as baseline (guarantees >= LogReg performance)
        # For default, use constant logodds
        if self.warm_start == "logreg" and hasattr(self, "_warm_start_lr"):
            X_scaled = self._warm_start_scaler.transform(X)
            baseline = (
                X_scaled @ self._warm_start_lr.coef_.T.flatten()
                + self._warm_start_lr.intercept_[0]
            )
            # Clip to match training behavior
            baseline = np.clip(baseline, -10, 10)
        elif getattr(self, "init", "zero") == "logodds":
            baseline = np.full(X.shape[0], self.logodds)
        else:
            baseline = np.full(X.shape[0], 0.0)

        if self.model_tree.node == "END":
            # No tree structure - just return baseline predictions
            baseline_clipped = np.clip(baseline, -500, 500)
            prob_1 = 1 / (1 + np.exp(-baseline_clipped))
            proba = np.zeros((X.shape[0], 2))
            proba[:, 1] = prob_1
            proba[:, 0] = 1 - prob_1
            return proba

        for obs in range(X.shape[0]):
            model = model_tree
            y_hat_obs = baseline[obs]  # start from baseline (LogReg or logodds)
            while model.node != "END" and model.depth < maxd:
                if model.node == "pconc":
                    if np.isin(X[obs, model.pivot[0]], model.pivot_categorical):
                        contribution = model.intercept_left
                        y_hat_obs += self._scale_and_clip(contribution)
                        model = model.left
                    else:
                        contribution = model.intercept_right
                        y_hat_obs += self._scale_and_clip(contribution)
                        model = model.right
                elif (
                    model.node in ["tlin", "lin", "con"]
                    or X[obs, model.pivot[0]] <= model.pivot[1]
                ):
                    # adjust x for truncation if necessary
                    x_truncated = np.max(
                        [
                            np.min([X[obs, model.pivot[0]], model.interval[1]]),
                            model.interval[0],
                        ]
                    )
                    contribution = model.slope_left * x_truncated + model.intercept_left
                    y_hat_obs += self._scale_and_clip(contribution)
                    model = model.left
                else:
                    # adjust x for truncation if necessary (here we only need to check the upper bound, as lower bound is already checked in the if condition)
                    x_truncated = np.min([X[obs, model.pivot[0]], model.interval[1]])
                    contribution = (
                        model.slope_right * x_truncated + model.intercept_right
                    )
                    y_hat_obs += self._scale_and_clip(contribution)
                    model = model.right
            y_hat.append(y_hat_obs)
        proba = np.zeros((len(y_hat), 2))
        # Clip y_hat to prevent overflow in exp
        y_hat_clipped = np.clip(np.array(y_hat), -500, 500)
        proba[:, 1] = 1 / (1 + np.exp(-y_hat_clipped))
        proba[:, 0] = 1 - proba[:, 1]
        return proba

    def predict(self, X, model=None, maxd=np.inf, **kwargs):
        proba = self.predict_proba(X, model=model, maxd=maxd, **kwargs)[:, 1]
        if proba.ndim == 1:
            return (proba >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def score(self, test_X, test_y):
        y_pred = self.predict(test_X)
        accuracy = np.mean(y_pred == test_y)
        return accuracy


def print_tree_inner_function(model_tree: Tree, level: int) -> None:
    if model_tree is not None:
        print_tree_inner_function(model_tree.left, level + 1)
        if model_tree.node == "lin":
            print(
                " " * 8 * level + "-->",
                model_tree.node,
                "feature:",
                model_tree.pivot[0],
                "wrss:",
                round(model_tree.wrss, 3),
                "left:",
                (round(model_tree.intercept_left, 3), round(model_tree.slope_left, 3)),
            )
        elif model_tree.node == "END":
            print(" " * 8 * level + "-->" + "END", "wrss:", round(model_tree.wrss, 3))
        elif model_tree.node == "pconc":
            print(
                " " * 8 * level + "-->",
                model_tree.node,
                "feature:",
                model_tree.pivot[0],
                "wrss:",
                round(model_tree.wrss, 3),
                "left:",
                round(model_tree.intercept_left, 3),
                "right:",
                round(model_tree.intercept_right, 3),
            )
        elif model_tree.node in ["con", "pcon"]:
            print(
                " " * 8 * level + "-->",
                model_tree.node,
                "feature:",
                model_tree.pivot[0],
                "wrss:",
                round(model_tree.wrss, 3),
                "left:",
                round(model_tree.intercept_left, 3),
                "right:",
                round(model_tree.intercept_right, 3),
                "pivot:",
                round(model_tree.pivot[1], 3),
            )
        else:
            print(
                " " * 8 * level + "-->",
                model_tree.node,
                "feature:",
                model_tree.pivot[0],
                "wrss:",
                round(model_tree.wrss, 3),
                "left:",
                (round(model_tree.intercept_left, 3), round(model_tree.slope_left, 3)),
                "right:",
                (
                    round(model_tree.intercept_right, 3),
                    round(model_tree.slope_right, 3),
                ),
                "pivot:",
                round(model_tree.pivot[1], 3),
            )
        print_tree_inner_function(model_tree.right, level + 1)


def plot_tree(
    model_tree: Tree, feature_names=None, figsize=(16, 10), title="PILOT Tree Structure"
):
    """
    Plot a visual representation of the PILOT tree using matplotlib.

    Parameters:
    -----------
    model_tree : tree
        The root node of the tree to plot
    feature_names : list or None
        List of feature names. If None, uses X0, X1, etc.
    figsize : tuple
        Figure size (width, height)
    title : str
        Title for the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    if model_tree is None or model_tree.node == "END":
        print("Tree is empty or has only END node.")
        return

    # Node colors for different node types
    node_colors = {
        "con": "#FFE5B4",  # Peach - constant
        "lin": "#B4E5FF",  # Light blue - linear
        "tlin": "#87CEFA",  # Light sky blue - truncated linear (special case of lin)
        "pcon": "#FFB4D9",  # Pink - piecewise constant
        "plin": "#D9B4FF",  # Lavender - piecewise linear
        "pconc": "#B4FFB4",  # Light green - piecewise constant categorical
        "END": "#E0E0E0",  # Gray - end node
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Calculate tree structure
    def get_tree_depth(node):
        """Calculate maximum depth of tree"""
        if node is None or node.node == "END":
            return 0
        return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))

    def count_leaves(node):
        """Count number of leaf nodes"""
        if node is None or node.node == "END":
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    max_depth = get_tree_depth(model_tree)
    leaf_count = count_leaves(model_tree)

    # Node positions storage
    node_positions = {}

    def calculate_positions(node, depth, x_min, x_max, parent_pos=None):
        """Recursively calculate positions for all nodes"""
        if node is None:
            return

        x = (x_min + x_max) / 2
        # Add padding: map depth to range [0.85, 0.05] instead of [1.0, 0.0]
        # This leaves space at top (0.15) and bottom (0.05)
        y = 0.85 - (depth / (max_depth + 1)) * 0.80

        node_positions[id(node)] = (x, y)

        if node.node != "END":
            # Calculate positions for children
            mid = (x_min + x_max) / 2
            if node.left is not None:
                calculate_positions(node.left, depth + 1, x_min, mid, (x, y))
            if node.right is not None:
                calculate_positions(node.right, depth + 1, mid, x_max, (x, y))

    # Calculate all positions
    calculate_positions(model_tree, 0, 0, 1)

    def format_node_text(node):
        """Format text to display in node"""
        if node.node == "END":
            return f"END\nRSS:{node.wrss:.2f}"

        # Feature name
        feat_idx = node.pivot[0] if node.pivot[0] != -1 else None
        feat_name = (
            f"X{feat_idx}"
            if feature_names is None or feat_idx is None
            else feature_names[feat_idx]
        )

        lines = [f"[{node.node.upper()}]"]
        (
            lines.append(f"samples:{len(node.obs_ids)}")
            if node.obs_ids is not None
            else None
        )

        if feat_idx is not None:
            lines.append(f"Feature {feat_name}")

        # Add model parameters (compact format)
        if node.node == "con":
            lines.append(f"int={node.intercept_left:.2f}")
        elif node.node in ["lin", "tlin"]:
            lines.append(f"int={node.intercept_left:.2f}")
            lines.append(f"sl={node.slope_left:.2f}")
        elif node.node in ["pcon", "pconc"]:
            # For piecewise constant, show L/R intercepts only
            lines.append(f"L={node.intercept_left:.2f}, R={node.intercept_right:.2f}")
        elif node.node == "plin":
            # For piecewise linear, most compact format
            lines.append(f"L: int={node.intercept_left:.2f}, sl={node.slope_left:.2f}")
            lines.append(
                f"R: int={node.intercept_right:.2f}, sl={node.slope_right:.2f}"
            )

        return "\n".join(lines)

    def draw_node(node, x, y):
        """Draw a single node"""
        color = node_colors.get(node.node, "#FFFFFF")

        # Node box - adjusted size based on node type
        if node.node == "plin":
            box_width = 0.09
            box_height = 0.06
        elif node.node in ["pcon", "pconc"]:
            box_width = 0.09
            box_height = 0.08
        else:
            box_width = 0.09
            box_height = 0.07

        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.005",
            edgecolor="black",
            facecolor=color,
            linewidth=1,
            zorder=1,
        )
        ax.add_patch(box)

        # Node text with adjusted font size
        text = format_node_text(node)
        # Use smaller font for nodes with more text
        fontsize = 6
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.0", facecolor="none", edgecolor="none"),
        )

    def draw_edge(parent_pos, child_pos, label=""):
        """Draw edge between parent and child"""
        x1, y1 = parent_pos
        x2, y2 = child_pos

        # Draw arrow
        arrow = FancyArrowPatch(
            (x1, y1 - 0.04),
            (x2, y2 + 0.04),
            arrowstyle="-",
            linewidth=1.5,
            color="black",
            zorder=1,
        )
        ax.add_patch(arrow)

        # Add label if provided
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"
                ),
            )

    def draw_tree(node):
        """Recursively draw the tree"""
        if node is None:
            return

        node_id = id(node)
        if node_id not in node_positions:
            return

        x, y = node_positions[node_id]

        # Draw current node
        draw_node(node, x, y)

        # Draw edges and children
        if node.node != "END":
            if node.left is not None:
                left_id = id(node.left)
                if left_id in node_positions:
                    left_pos = node_positions[left_id]
                    # Determine label based on node type
                    if node.node in ["pcon", "plin"]:
                        label = f"≤ {node.pivot[1]:.2f}"
                    elif node.node == "pconc":
                        label = "∈ cats"
                    else:
                        label = ""
                    draw_edge((x, y), left_pos, label)
                    draw_tree(node.left)

            if node.right is not None:
                right_id = id(node.right)
                if right_id in node_positions:
                    right_pos = node_positions[right_id]
                    # Determine label based on node type
                    if node.node in ["pcon", "plin"]:
                        label = f"> {node.pivot[1]:.2f}"
                    elif node.node == "pconc":
                        label = "∉ cats"
                    else:
                        label = ""
                    draw_edge((x, y), right_pos, label)
                    draw_tree(node.right)

    # Draw the entire tree
    draw_tree(model_tree)

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor=node_colors["con"], edgecolor="black", label="Constant (con)"
        ),
        mpatches.Patch(
            facecolor=node_colors["lin"], edgecolor="black", label="Linear (lin)"
        ),
        mpatches.Patch(
            facecolor=node_colors["tlin"],
            edgecolor="black",
            label="Truncated Linear (tlin)",
        ),
        mpatches.Patch(
            facecolor=node_colors["pcon"],
            edgecolor="black",
            label="Piecewise Constant (pcon)",
        ),
        mpatches.Patch(
            facecolor=node_colors["plin"],
            edgecolor="black",
            label="Piecewise Linear (plin)",
        ),
        mpatches.Patch(
            facecolor=node_colors["pconc"],
            edgecolor="black",
            label="Piecewise Constant Cat (pconc)",
        ),
        mpatches.Patch(facecolor=node_colors["END"], edgecolor="black", label="END"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1), fontsize=9
    )

    # Add title and info
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    info_text = f"Tree Depth: {max_depth} | Leaf Nodes: {leaf_count}"
    ax.text(
        0.5,
        -0.02,
        info_text,
        ha="center",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )
