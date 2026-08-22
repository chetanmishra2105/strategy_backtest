"""Market-regime detection via a Gaussian Hidden Markov Model (NEW module).

A genuine Markov model: the market is assumed to be in one of ``k`` hidden
states (e.g. calm-bull / choppy / high-vol-bear). Each state emits daily
observations (return, volatility) from its own Gaussian, and states transition
according to a Markov transition matrix. We fit the model with Baum-Welch (EM)
and decode the most-likely state path with Viterbi.

Why hand-rolled: ``hmmlearn``/``scikit-learn`` are not installed and this runs
behind a corporate proxy where pip may be blocked. This is a compact, dependency
-free 2-D diagonal-covariance Gaussian HMM (numpy + scipy only) — enough for
regime gating, not a general HMM library.

How it's used (see momentum.py): NOT a standalone trade signal. It labels each
date Bull / Neutral / Bear (by ranking states on mean return), and the momentum
engine goes to cash when the label is risk-off. Refit point-in-time by the
walk-forward harness so no future data leaks into a past regime label.

Observations per day: [ z-scored trailing return , z-scored trailing volatility ].
States are labelled AFTER fitting by sorting on mean return so labels are stable
across refits regardless of the arbitrary state indices EM lands on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The Gaussian log-likelihood is computed in pure numpy below, so this module
# has NO third-party dependency beyond numpy/pandas — important because
# hmmlearn/scikit-learn are not installed and pip may be blocked on this network.


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def build_features(close: pd.Series, ret_window: int = 5, vol_window: int = 20
                   ) -> pd.DataFrame:
    """Two causal features from a price series:
      * ret : trailing ``ret_window``-day log return (momentum sign/size),
      * vol : trailing ``vol_window``-day realised vol of daily returns.
    Standardised (z-scored) over the fitting sample by the caller.
    """
    c = close.astype(float)
    logret = np.log(c / c.shift(1))
    ret = logret.rolling(ret_window).sum()
    vol = logret.rolling(vol_window).std()
    feat = pd.DataFrame({"ret": ret, "vol": vol}).dropna()
    return feat


# --------------------------------------------------------------------------
# Gaussian HMM (diagonal covariance) — Baum-Welch + Viterbi
# --------------------------------------------------------------------------
class GaussianHMM:
    """Minimal diagonal-covariance Gaussian HMM.

    Parameters after fit:
      pi   : (k,) initial state probabilities
      A    : (k,k) row-stochastic transition matrix
      mu   : (k,d) per-state means
      var  : (k,d) per-state variances (diagonal covariance)
    """

    def __init__(self, n_states: int = 3, n_iter: int = 25, seed: int = 0,
                 tol: float = 1e-4):
        self.k = int(n_states)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.tol = float(tol)
        self.pi = self.A = self.mu = self.var = None

    # --- Emission log-likelihood (T,k) --------------------------------------
    def _log_emit(self, X: np.ndarray) -> np.ndarray:
        T, d = X.shape
        logB = np.empty((T, self.k))
        for j in range(self.k):
            var = np.maximum(self.var[j], 1e-6)
            diff = X - self.mu[j]
            # log N(x; mu, diag(var)) summed over dims.
            logB[:, j] = -0.5 * (np.sum(np.log(2 * np.pi * var))
                                 + np.sum((diff ** 2) / var, axis=1))
        return logB

    def _init_params(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        T, d = X.shape
        # k-means-lite init: sort by first feature, split into k chunks.
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, self.k)
        self.mu = np.array([X[c].mean(axis=0) for c in chunks])
        self.var = np.array([X[c].var(axis=0) + 1e-3 for c in chunks])
        self.pi = np.full(self.k, 1.0 / self.k)
        # Sticky transition prior (regimes persist).
        self.A = np.full((self.k, self.k), 0.1 / max(self.k - 1, 1))
        np.fill_diagonal(self.A, 0.9)

    def fit(self, X: np.ndarray, warm_start: "GaussianHMM | None" = None) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        if warm_start is not None and warm_start.mu is not None \
                and warm_start.mu.shape == (self.k, X.shape[1]):
            # Continue from a previously-fit model: on an expanding window the
            # regime structure barely moves, so a few EM iters re-converge — this
            # is what makes point-in-time refitting affordable.
            self.pi = warm_start.pi.copy()
            self.A = warm_start.A.copy()
            self.mu = warm_start.mu.copy()
            self.var = warm_start.var.copy()
        else:
            self._init_params(X)
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            logB = self._log_emit(X)
            log_alpha, ll = self._forward(logB)
            log_beta = self._backward(logB)
            # Posteriors.
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            # xi (T-1,k,k).
            xi = self._xi(log_alpha, log_beta, logB)
            # M-step.
            self.pi = gamma[0] + 1e-12
            self.pi /= self.pi.sum()
            A_num = xi.sum(axis=0) + 1e-12
            self.A = A_num / A_num.sum(axis=1, keepdims=True)
            Nk = gamma.sum(axis=0) + 1e-12
            self.mu = (gamma.T @ X) / Nk[:, None]
            for j in range(self.k):
                diff = X - self.mu[j]
                self.var[j] = (gamma[:, j] @ (diff ** 2)) / Nk[j] + 1e-4
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self._loglik = prev_ll
        return self

    def _forward(self, logB):
        T = logB.shape[0]
        log_alpha = np.empty((T, self.k))
        logA = np.log(self.A + 1e-12)
        log_alpha[0] = np.log(self.pi + 1e-12) + logB[0]
        for t in range(1, T):
            log_alpha[t] = logB[t] + _logsumexp(
                log_alpha[t - 1][:, None] + logA, axis=0)
        ll = _logsumexp(log_alpha[-1])
        return log_alpha, ll

    def _backward(self, logB):
        T = logB.shape[0]
        log_beta = np.zeros((T, self.k))
        logA = np.log(self.A + 1e-12)
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(
                logA + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
        return log_beta

    def _xi(self, log_alpha, log_beta, logB):
        T = logB.shape[0]
        logA = np.log(self.A + 1e-12)
        xi = np.zeros((T - 1, self.k, self.k))
        for t in range(T - 1):
            m = (log_alpha[t][:, None] + logA
                 + logB[t + 1][None, :] + log_beta[t + 1][None, :])
            m -= _logsumexp(m)
            xi[t] = np.exp(m)
        return xi

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most-likely hidden-state path (T,) of integer state indices."""
        X = np.asarray(X, dtype=float)
        logB = self._log_emit(X)
        T = logB.shape[0]
        logA = np.log(self.A + 1e-12)
        delta = np.empty((T, self.k))
        psi = np.zeros((T, self.k), dtype=int)
        delta[0] = np.log(self.pi + 1e-12) + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + np.max(scores, axis=0)
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def _logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


# --------------------------------------------------------------------------
# Regime labelling — the public API used by the app
# --------------------------------------------------------------------------
def _label_states(model: GaussianHMM, n_states: int) -> dict[int, str]:
    """Map raw HMM state indices -> human labels by ranking on mean return.

    2 states -> Bear/Bull. 3 -> Bear/Neutral/Bull. 4 -> add Strong-Bull at top.
    The first feature (index 0) is the trailing return, so we sort states by
    ``mu[:,0]`` ascending: lowest return = most bearish.
    """
    order = np.argsort(model.mu[:, 0])  # ascending mean return
    if n_states <= 2:
        names = ["Bear", "Bull"]
    elif n_states == 3:
        names = ["Bear", "Neutral", "Bull"]
    else:
        names = ["Bear", "Neutral", "Bull"] + ["Strong-Bull"] * (n_states - 3)
    return {int(state): names[i] for i, state in enumerate(order)}


def regime_series(
    close: pd.Series,
    n_states: int = 3,
    ret_window: int = 5,
    vol_window: int = 20,
    seed: int = 0,
) -> pd.Series:
    """Fit an HMM on the WHOLE ``close`` series and return a per-date regime
    label Series (Bull / Neutral / Bear ...).

    NOTE: this fits on the full sample (in-sample labelling), fine for the
    Backtest Lab's "current view". For a look-ahead-free backtest use
    ``regime_series_pit`` (expanding-window refit) via the walk-forward harness.
    """
    feat = build_features(close, ret_window, vol_window)
    if len(feat) < max(50, n_states * 20):
        # Fallback: simple trend/vol heuristic when there's too little data to
        # fit a stable HMM.
        return _heuristic_regime(close, ret_window, vol_window)
    Xz, _, _ = _zscore(feat.to_numpy())
    model = GaussianHMM(n_states=n_states, seed=seed).fit(Xz)
    path = model.viterbi(Xz)
    labels = _label_states(model, n_states)
    return pd.Series([labels[s] for s in path], index=feat.index, name="regime")


def regime_series_pit(
    close: pd.Series,
    n_states: int = 3,
    ret_window: int = 5,
    vol_window: int = 20,
    min_train: int = 252,
    refit_every: int = 63,
    seed: int = 0,
) -> pd.Series:
    """Point-in-time regime labels (no look-ahead).

    Walks forward: every ``refit_every`` bars, refit the HMM on data up to that
    date only, then label the bars until the next refit using that model. The
    z-scoring stats are also computed from the training window only. This is the
    version to use inside a backtest so a past regime label never 'sees' future
    prices.

    Performance: each refit **warm-starts** from the previous model (the regime
    structure barely moves on an expanding window), so only the first fit is
    cold and the rest converge in a couple of EM iterations. ``refit_every=63``
    (~quarterly) keeps the whole walk to a few seconds.
    """
    feat = build_features(close, ret_window, vol_window)
    if len(feat) < min_train + 5:
        return _heuristic_regime(close, ret_window, vol_window)
    X = feat.to_numpy()
    idx = feat.index
    out = np.empty(len(idx), dtype=object)
    out[:] = "Neutral"
    t = min_train
    model = None
    labels = None
    mu = sd = None
    while t < len(idx):
        Xtr = X[:t]
        Xz, mu, sd = _zscore(Xtr)
        try:
            # Cold first fit (25 iters); warm refits re-converge in ~6.
            n_iter = 25 if model is None else 6
            new_model = GaussianHMM(n_states=n_states, seed=seed, n_iter=n_iter).fit(
                Xz, warm_start=model)
            model = new_model
            labels = _label_states(model, n_states)
        except Exception:
            model = None
        block_end = min(t + refit_every, len(idx))
        if model is not None:
            Xblk = (X[t:block_end] - mu) / sd
            path = model.viterbi(Xblk)
            for i, s in enumerate(path):
                out[t + i] = labels[s]
        t = block_end
    return pd.Series(out, index=idx, name="regime")


def _zscore(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _heuristic_regime(close: pd.Series, ret_window: int, vol_window: int) -> pd.Series:
    """Dependency-free fallback: Bull when above 200-SMA & low vol, Bear when
    below & high vol, else Neutral. Used if scipy is missing or data is thin."""
    c = close.astype(float)
    sma = c.rolling(200).mean()
    logret = np.log(c / c.shift(1))
    vol = logret.rolling(vol_window).std()
    volmed = vol.rolling(252, min_periods=60).median()
    lab = pd.Series("Neutral", index=c.index)
    lab[(c > sma) & (vol <= volmed)] = "Bull"
    lab[(c < sma) & (vol > volmed)] = "Bear"
    return lab.reindex(close.index).dropna()


# States considered risk-off (momentum book goes to cash).
RISK_OFF = {"Bear"}
