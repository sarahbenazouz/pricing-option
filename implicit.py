import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve

# Paramètres numériques
N_S = 200
S_max = 3 * K
N_t = 1000

dS = S_max / N_S
dt = T / N_t

# Discrétisation du domaine
S_grid = np.linspace(0, S_max, N_S + 1)
tau_grid = np.linspace(0, T, N_t + 1)

# Condition initiale : payoff du call européen à maturité
U = np.maximum(S_grid - K, 0)

# Construction de la matrice du schéma implicite
A = np.zeros((N_S + 1, N_S + 1))

# Condition à la limite en S = 0
A[0, 0] = 1

# Coefficients du schéma implicite
for j in range(1, N_S):
    Sj = S_grid[j]

    a = 0.5 * sigma**2 * Sj**2 / dS**2 \
        - r * Sj / (2 * dS)

    b = sigma**2 * Sj**2 / dS**2 + r

    c = 0.5 * sigma**2 * Sj**2 / dS**2 \
        + r * Sj / (2 * dS)

    A[j, j-1] = -dt * a
    A[j, j] = 1 + dt * b
    A[j, j+1] = -dt * c

# Condition à la limite en S = S_max
A[N_S, N_S] = 1

# Factorisation LU : calculée une seule fois
t1 = time.time()
LU, piv = lu_factor(A)

# Résolution aux différents pas de temps
for n in range(N_t):
    tau_next = tau_grid[n + 1]

    # Second membre
    b = U.copy()

    # Conditions aux limites au nouveau pas de temps
    b[0] = 0
    b[N_S] = S_max - K * np.exp(-r * tau_next)

    # Résolution du système linéaire
    U = lu_solve((LU, piv), b)

temps_calcul = time.time() - t1

# Prix numérique du call pour S = S0
prix_implicite = np.interp(S0, S_grid, U)

print("Prix du call par le schéma implicite :", prix_implicite)
print("Temps de calcul :", temps_calcul, "secondes")
```
