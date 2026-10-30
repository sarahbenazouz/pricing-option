
import numpy as np
import time
from bs import S0, K, T, r, sigma, bs_analytic   # paramètres + référence analytique


# Paramètres numériques
N_S = 200
S_max = 3 * K
N_t = 10000

dS = S_max / N_S
dt = T / N_t

# Discrétisation du domaine
S_grid = np.linspace(0, S_max, N_S + 1)
tau_grid = np.linspace(0, T, N_t + 1)

# Condition initiale : payoff du call européen à maturité
U = np.maximum(S_grid - K, 0)

# Coefficients du schéma explicite
t1 = time.time()

for n in range(N_t):
    tau = tau_grid[n]

    U_new = U.copy()

    for j in range(1, N_S):
        a = 0.5 * sigma**2 * S_grid[j]**2 * dt / dS**2 \
            - 0.5 * r * S_grid[j] * dt / dS

        b = 1 - sigma**2 * S_grid[j]**2 * dt / dS**2 - r * dt

        c = 0.5 * sigma**2 * S_grid[j]**2 * dt / dS**2 \
            + 0.5 * r * S_grid[j] * dt / dS

        U_new[j] = a * U[j-1] + b * U[j] + c * U[j+1]

    # Conditions aux limites
    U_new[0] = 0
    U_new[N_S] = S_max - K * np.exp(-r * (tau + dt))

    U = U_new

temps_calcul = time.time() - t1

# Prix numérique pour S0
prix_explicite = np.interp(S0, S_grid, U)

print("Prix du call par le schéma explicite :", prix_explicite)
print("Temps de calcul :", temps_calcul, "secondes")
print("Prix du call par la formule analytique :", bs_analytic(S0, K, T, r, sigma))
