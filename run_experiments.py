"""Reproduit les résultats numériques de la partie 3 (tableaux et seuils).
Usage : python run_experiments.py   (depuis src/)"""
import numpy as np
from schemas_fd import solve, ex, S0, K, T, r, sigma, bs_analytic
from put_americain import put_amer, crr

print(f"Prix exact du call : {ex:.6f}\n")
print("== 3.1 Validation : (N, M) -> erreur absolue (explicite, implicite, CN)")
for N, M in [(50, 2000), (100, 2000), (200, 2000), (400, 8000), (800, 32000)]:
    errs = [abs(solve(s, N, M) - ex) for s in ("exp", "imp", "cn")]
    print(f"N={N:4d} M={M:6d}  " + "  ".join(f"{e:.2e}" for e in errs))

print("\n== 3.2 Ordre en temps (N=400, référence = même schéma, M très fin)")
N = 400
refs = {"imp": solve("imp", N, 12800), "cn": solve("cn", N, 6400)}
for s in ("imp", "cn"):
    prev = None
    for M in (10, 20, 40, 80, 160, 320):
        e = abs(solve(s, N, M) - refs[s])
        print(f"{s:3s} M={M:4d} err={e:.3e} ordre={'' if prev is None else round(np.log2(prev/e), 2)}")
        prev = e

print("\n== 3.3 Stabilité de l'explicite (N=200) : erreur max sur la grille")
N = 200
print(f"CFL théorique : M >= {T*sigma**2*300**2/(300/N)**2:.0f}")
for M in (1100, 1400, 1450, 1500, 1600, 2000):
    S, U = solve("exp", N, M, full=True)
    print(f"M={M:5d}  erreur max = {np.abs(U - bs_analytic(S, K, T, r, sigma)).max():.2e}")

print("\n== 3.4 Put américain (CN + PSOR) vs arbre CRR")
ref = crr(4000)
print(f"CRR américain = {ref:.5f}")
for N, M in [(100, 100), (200, 200), (400, 400)]:
    S, U, Ss, its = put_amer(N, M)
    p = np.interp(S0, S, U)
    print(f"N={N} M={M} prix={p:.4f} err={abs(p-ref):.2e} it.GS/pas={its/M:.1f} S*(0)={Ss[-1]}")
