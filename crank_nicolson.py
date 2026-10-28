#crank_nicolson.py : schéma de Crank-Nicolson pour l'EDP de Black-Scholes.
#Référence : Crank & Nicolson (1947), Wilmott (2006) §ch. différences finies.

import numpy as np
import time
from bs import S0, K, T, r, sigma, bs_analytic  # paramètres + référence


# -------------------------------------------------------------------------
# Paramètres numériques
# -------------------------------------------------------------------------
N_S   = 200       # nombre de pas spatiaux
N_t   = 500       # nombre de pas temporels
#   CN est d'ordre 2 en temps → N_t bien plus faible qu'en explicite (10 000)
S_max = 3 * K

dS  = S_max / N_S
dt  = T / N_t

# Grilles
S_grid   = np.linspace(0, S_max, N_S + 1)   # S_0=0, …, S_{N_S}=S_max
tau_grid = np.linspace(0, T, N_t + 1)

# Condition initiale : payoff du call à maturité (tau = 0)
U = np.maximum(S_grid - K, 0.0)


# -------------------------------------------------------------------------
# Pré-calcul des diagonales (constantes en temps)
# -------------------------------------------------------------------------
j  = np.arange(1, N_S)        # indices intérieurs  j = 1 … N_S-1
Sj = S_grid[j]

# Coefficients "implicites" de base (= 2 × coeff CN côté implicite)
#   alpha_j = sigma²Sj²dt/(2dS²) - rSj*dt/(2dS)
#   beta_j  = 1 + sigma²Sj²dt/dS² + r*dt
#   gamma_j = sigma²Sj²dt/(2dS²) + rSj*dt/(2dS)
alpha = 0.5 * sigma**2 * Sj**2 * dt / dS**2 - 0.5 * r * Sj * dt / dS
beta  = 1.0 + sigma**2 * Sj**2 * dt / dS**2 + r * dt
gamma = 0.5 * sigma**2 * Sj**2 * dt / dS**2 + 0.5 * r * Sj * dt / dS

# Matrice M⁺  (côté n+1, implicite, coeff / 2)
#   diag. princ. Bp_j = 1 + (beta_j - 1) / 2
Ap = -alpha / 2                  # sous-diagonale de M⁺
Bp =  1.0 + (beta - 1.0) / 2    # diagonale principale de M⁺
Cp = -gamma / 2                  # sur-diagonale de M⁺

# Matrice M⁻  (côté n, explicite, coeff / 2)
#   diag. princ. Bm_j = 1 - (beta_j - 1) / 2
Am =  alpha / 2                  # sous-diagonale de M⁻
Bm =  1.0 - (beta - 1.0) / 2    # diagonale principale de M⁻
Cm =  gamma / 2                  # sur-diagonale de M⁻


# -------------------------------------------------------------------------
# Algorithme de Thomas  (élimination de Gauss tridiagonale, O(N))
# -------------------------------------------------------------------------
def thomas(a, b, c, d):
    """Résout le système tridiagonal Ax = d.

    Paramètres
    ----------
    a : sous-diagonale  (a[0] non utilisé)
    b : diagonale principale
    c : sur-diagonale   (c[-1] non utilisé)
    d : second membre
    Tous les tableaux ont la même longueur n.

    Retourne
    --------
    x : solution (tableau 1-D de longueur n)
    """
    n   = len(b)
    b_  = b.copy()
    d_  = d.copy()

    # Phase de descente (élimination vers l'avant)
    for i in range(1, n):
        w      = a[i] / b_[i - 1]
        b_[i] -= w * c[i - 1]
        d_[i] -= w * d_[i - 1]

    # Phase de remontée (substitution arrière)
    x = np.zeros(n)
    x[-1] = d_[-1] / b_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d_[i] - c[i] * x[i + 1]) / b_[i]

    return x


# -------------------------------------------------------------------------
# Boucle temporelle
# -------------------------------------------------------------------------
t1 = time.time()

for n in range(N_t):
    tau_n    = tau_grid[n]
    tau_next = tau_grid[n + 1]

    # Conditions aux limites
    U0_cur  = 0.0                                      # S = 0 : call sans valeur
    UN_cur  = S_max - K * np.exp(-r * tau_n)           # S = S_max au pas n
    U0_next = 0.0
    UN_next = S_max - K * np.exp(-r * tau_next)        # S = S_max au pas n+1

    # Valeurs intérieures au pas n
    Uint = U[1:N_S]

    # Calcul du second membre  rhs = M⁻ U^n  +  absorption CL de M⁺
    rhs = np.empty(N_S - 1)

    # Points intérieurs généraux  (indices locaux 1 … N_S-3)
    rhs[1:-1] = (Am[1:-1] * Uint[:-2]
                 + Bm[1:-1] * Uint[1:-1]
                 + Cm[1:-1] * Uint[2:])

    # Bord gauche (j=1) : voisin U_{j-1} = U_0
    rhs[0]  = Am[0] * U0_cur + Bm[0] * Uint[0] + Cm[0] * Uint[1]

    # Bord droit (j=N_S-1) : voisin U_{j+1} = U_{N_S}
    rhs[-1] = Am[-1] * Uint[-2] + Bm[-1] * Uint[-1] + Cm[-1] * UN_cur

    # Absorption des conditions aux limites dans M⁺
    rhs[0]  -= Ap[0]  * U0_next
    rhs[-1] -= Cp[-1] * UN_next

    # Résolution  M⁺ U^{n+1} = rhs  par Thomas
    U[1:N_S] = thomas(Ap, Bp, Cp, rhs)
    U[0]     = U0_next
    U[N_S]   = UN_next

temps_calcul = time.time() - t1


# -------------------------------------------------------------------------
# Résultats
# -------------------------------------------------------------------------
prix_cn    = np.interp(S0, S_grid, U)
prix_exact = bs_analytic(S0, K, T, r, sigma)

print(f"Prix du call — Crank-Nicolson     : {prix_cn:.6f}")
print(f"Prix du call — formule analytique : {prix_exact:.6f}")
print(f"Erreur absolue                    : {abs(prix_cn - prix_exact):.2e}")
print(f"Temps de calcul                   : {temps_calcul:.3f} s")
