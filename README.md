# Black-Scholes par différences finies

Mini-projet Polytech Nice Sophia / MAM5 IMAFA — *Méthodes numériques pour le pricing d'options*.
Auteures : Sarah Benazouz, Yasmine El-Hainous, Khaoula Hajlaoui.

Résolution de l'équation de Black-Scholes par trois schémas aux différences finies
(explicite, implicite, Crank-Nicolson), validation sur le call européen par la formule
analytique, puis extension au **put américain** (Crank-Nicolson + PSOR).

## Paramètres de référence
`S0 = K = 100`, `T = 1`, `sigma = 0.20`, `r = 0.05`, domaine `[0, 3K]` (définis dans `src/bs.py`).

## Structure
```
src/
  bs.py                 # formule analytique (call/put), delta, gamma, paramètres
  explicit.py           # schéma explicite (script autonome)
  implicit.py           # schéma implicite, factorisation LU (script autonome)
  crank_nicolson.py     # Crank-Nicolson, algorithme de Thomas (script autonome)
  schemas_fd.py         # les 3 schémas dans une fonction solve(schema, N, M) pour les études
  put_americain.py      # put américain CN + PSOR, frontière S*(t), arbre CRR de référence
  run_experiments.py    # reproduit les tableaux/seuils de la partie 3
  make_figures.py       # génère les figures du rapport
figures/                # fig_convergence, fig_stabilite, fig_americain (PDF + PNG)
results/
  partie3.tex           # section « Résultats et analyse » du rapport
  sortie_experiments.txt# sortie de run_experiments.py
```

## Installation et utilisation
```bash
pip install -r requirements.txt
cd src

python explicit.py          # prix du call, schéma explicite (N_t = 10000)
python implicit.py          # schéma implicite
python crank_nicolson.py    # Crank-Nicolson, erreur vs formule exacte

python run_experiments.py   # tableaux de la partie 3
python make_figures.py      # régénère ../figures
```


## Résultats principaux
| Étude | Résultat |
|---|---|
| Ordre en espace | 2 pour les 3 schémas |
| Ordre en temps | 1 (implicite), 2 (Crank-Nicolson, pour M ≥ 160) |
| Stabilité explicite (N=200) | stable pour M ≥ 1500, divergence pour M ≤ 1450 (CFL théorique : M ≥ 1600) |
| Put américain en S0 | 6.0909 (CN-PSOR, N=M=400) vs 6.0902 (arbre CRR, 4000 pas) |
| Prime d'exercice anticipé | ≈ 0.52 en S0 ; frontière S*(0) ≈ 81 |

## Notes
- `explicit.py` utilise des boucles Python pures (lent mais fidèle au schéma) ; `schemas_fd.py`
  est vectorisé et sert aux études de convergence.
- Le schéma explicite est conditionnellement stable : `dt <= dS² / (sigma² S_max²)`.
- La frontière `S*(t)` est connue à `dS` près (allure en escalier).
