"""Génère fig_convergence, fig_stabilite, fig_americain dans ../figures/ (PDF + PNG)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from schemas_fd import solve, ex, K, T, r, sigma, bs_analytic
from put_americain import put_amer

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
O = os.path.join(os.path.dirname(__file__), "..", "figures") + os.sep
os.makedirs(O, exist_ok=True)

# --- Convergence ---
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
Ns = np.array([25, 50, 100, 200, 400, 800])
for s, l, m in [("exp", "Explicite", "o"), ("imp", "Implicite", "s"), ("cn", "Crank-Nicolson", "^")]:
    e = [abs(solve(s, N, max(4000, int(1.1*T*(sigma**2*N**2*9+r)))) - ex) for N in Ns]
    ax[0].loglog(300/Ns, e, m+"-", label=l)
h = 300/Ns
ax[0].loglog(h, 0.0038*(h/1.5)**2, "k--", lw=.8, label="pente 2")
ax[0].set(xlabel=r"$\Delta S$", ylabel="erreur absolue", title=r"Convergence en espace"); ax[0].legend()
Ms = np.array([10, 20, 40, 80, 160, 320]); N = 400
ri, rc = solve("imp", N, 12800), solve("cn", N, 6400)
dts = T/Ms
ax[1].loglog(dts, [abs(solve("imp", N, M)-ri) for M in Ms], "s-", label="Implicite", color="C1")
ax[1].loglog(dts, [abs(solve("cn", N, M)-rc) for M in Ms], "^-", label="Crank-Nicolson", color="C2")
ax[1].loglog(dts, 0.1*(dts/dts[0]), "k:", lw=.8, label="pente 1")
ax[1].loglog(dts, 0.07*(dts/dts[0])**2, "k--", lw=.8, label="pente 2")
ax[1].set(xlabel=r"$\Delta\tau$", title=r"Convergence en temps ($N=400$)"); ax[1].legend()
plt.tight_layout(); plt.savefig(O+"fig_convergence.pdf"); plt.savefig(O+"fig_convergence.png"); plt.close()

# --- Stabilité ---
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, M, t in [(ax[0], 2000, r"$M=2000$ (stable)"), (ax[1], 1450, r"$M=1450$ (instable)")]:
    S, U = solve("exp", 200, M, full=True)
    a.plot(S, U, label="explicite"); a.plot(S, bs_analytic(S, K, T, r, sigma), "k--", lw=.8, label="Black-Scholes")
    a.set(title=t, xlabel="$S$"); a.legend()
ax[1].set_yscale("symlog", linthresh=100)
plt.tight_layout(); plt.savefig(O+"fig_stabilite.pdf"); plt.savefig(O+"fig_stabilite.png"); plt.close()

# --- Put américain ---
S, U, Ss, _ = put_amer(400, 400)
Se, Ue, _, _ = put_amer(400, 400, american=False)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5)); m = S <= 140
ax[0].plot(S[m], U[m], label="put américain"); ax[0].plot(S[m], Ue[m], "--", label="put européen")
ax[0].plot(S[m], np.maximum(K-S[m], 0), ":k", label="payoff"); ax[0].set(xlabel="$S$", title="Prix à $t=0$"); ax[0].legend()
tt = np.linspace(0, T, 401)
ax[1].plot(T-tt[1:], Ss[1:]); ax[1].set(xlabel="$t$ (temps réel)", ylabel="$S^*(t)$", title="Frontière d'exercice", ylim=(70, 102))
plt.tight_layout(); plt.savefig(O+"fig_americain.pdf"); plt.savefig(O+"fig_americain.png"); plt.close()
print("Figures écrites dans", os.path.abspath(O))
