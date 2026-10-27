"""
bs.py : solution analytique de Black-Scholes (option européenne, sans dividende).

Les paramètres de référence ci-dessous peuvent être importés dans les autres fichiers :
    from bs import S0, K, T, r, sigma, bs_analytic
"""
import numpy as np
from scipy.special import ndtr  # fonction de répartition de la loi normale centrée réduite

# --- Paramètres de référence ---
S0 = 100.0     # prix initial du sous-jacent
K = 100.0      # prix d'exercice (strike)
T = 1.0        # maturité (en années)
r = 0.05       # taux sans risque
sigma = 0.20   # volatilité


def _d1_d2(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):  # S = 0 : log(0) = -inf, sans avertissement
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S, d1, d1 - sigma * np.sqrt(T)


def bs_analytic(S, K, T, r, sigma, option="call"):
    """Prix de Black-Scholes (call ou put européen) à t = 0 ; S scalaire ou tableau."""
    S, d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    return K * np.exp(-r * T) * ndtr(-d2) - S * ndtr(-d1)


def call_bs(S, K, T, r, sigma):
    return bs_analytic(S, K, T, r, sigma, "call")


def put_bs(S, K, T, r, sigma):
    return bs_analytic(S, K, T, r, sigma, "put")


def delta(S, K, T, r, sigma, option="call"):
    """Delta : N(d1) pour le call, N(d1) - 1 pour le put."""
    _, d1, _ = _d1_d2(S, K, T, r, sigma)
    return ndtr(d1) if option == "call" else ndtr(d1) - 1.0


def gamma(S, K, T, r, sigma):
    """Gamma (identique pour call et put) : phi(d1) / (S sigma sqrt(T))."""
    S, d1, _ = _d1_d2(S, K, T, r, sigma)
    return np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) / (S * sigma * np.sqrt(T))


if __name__ == "__main__":
    c = bs_analytic(S0, K, T, r, sigma, "call")
    p = bs_analytic(S0, K, T, r, sigma, "put")
    print(f"Call européen : {c:.6f}")
    print(f"Put européen  : {p:.6f}")
    print(f"Delta du call : {delta(S0, K, T, r, sigma):.6f}")
    print(f"Gamma         : {gamma(S0, K, T, r, sigma):.6f}")
    print(f"Parité C - P = {c - p:.6f}  |  S0 - K e^(-rT) = {S0 - K * np.exp(-r * T):.6f}")