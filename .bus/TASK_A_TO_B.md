# TASK A → B — Ciclo 398

## Stato del repository (verificato da Agente A)

- `pytest -q`: **18 passed, 1 xfailed** (lo xfail è `tests/test_singular.py`, STUB voluto).
- WP1 / gate G1 chiuso: `exact/solver.py` (P ≻ 0) implementato e certificato (H1); `exact/price.py` (T1, T4); `exact/pruning.py` (T6); `mpc/condense.py`; `models/plants.py`.
- WP2 / **gate G2 ancora aperto**: `exact/singular.py` è uno STUB che solleva `NotImplementedError` (T3, C4, H5).
- Prossimo passo unico secondo `docs/roadmap.md` (WP2) e il metaprompt (Fase 1: risoluzione per `P ⪰ 0`).

## Obiettivo di questo ciclo (UN solo step, niente altro)

Implementare il **solver esatto per Hessiana semidefinita `P ⪰ 0` con disuguaglianze**: il cuore del gate G2. Nessun altro modulo va toccato (no planner, no GuardSet, no experiment H5, no baseline nuovi).

## Specifica matematica (da implementare, non da reinventare)

Problema: `min ½ u'Pu + q'u  s.t.  G u ≤ h`, con `P = P' ⪰ 0` (eventualmente degenere), `G ∈ R^{κ×n}`.

Scomposizione spettrale `P = V Λ V'` con `V` (n×r) base ortonormale di `range(P)`, `Λ ≻ 0`, `N` (n×(n−r)) base ortonormale di `null(P)`, `P⁺ = V Λ⁻¹ V'`.

1. **Test di range.** Il problema senza vincoli ha minimizzatore se e solo se `N'q = 0` (cioè `q ∈ range(P)`); altrimenti il problema è illimitato inferiormente → `found = False`. Soluzione particolare `u_p = −P⁺ q`, che è già ortogonale a `N` (quindi di norma minima nella sua classe).
2. **Set attivo J.** Per ogni `J ⊆ {1..κ}` (fino a `max_active`, default `min(κ, n)`), il sotto-problema con uguaglianze `G_J u = h_J` ha condizioni KKT
   `P u + q + G_J' ν_J = 0`,  `G_J u = h_J`,
   cioè il sistema lineare (possibilmente singolare o inconsistente)
   `M_J [u; ν_J] = [−q; h_J]`,  `M_J = [[P, G_J'], [G_J, 0]]`.
   Risolverlo nel senso dei minimi quadrati (`np.linalg.lstsq` con `rcond`). Accettare `J` solo se il sistema è consistente entro `‖M_J x − b‖ / (1 + ‖b‖) ≤ 1e-8`; altrimenti scartarlo.
3. **Selezione deterministica di norma minima.** Il sistema è sotto-determinato quando una parte di `null(P)` resta libera. Data la soluzione particolare `x0 = [u0; ν0]` e una base `Z` di `null(M_J)`, minimizzare `‖u‖` sul set delle soluzioni `x(t) = x0 + Z t`: sia `U` l'operatore che estrae le prime `n` componenti, risolvere ai minimi quadrati `(U Z) t = −U x0` e porre `x = x0 + Z t`. Questo rende la legge di controllo **univoca** (richiesto dal metaprompt).
4. **Certificato KKT** in `float64`, sulle quantità ORIGINALI, senza tolleranze sull'ottimo:
   - fattibilità primale: `max(0, max_j (G u − h)_j) ≤ tol`
   - duale: `min ν ≥ −tol`
   - stazionarietà: `‖P u + q + G'ν‖_∞ ≤ 1e-8`
   - complementarità: `max_j |ν_j (G u − h)_j| ≤ tol`
   - gap: `|V − g(ν)| / max(1, |V|) ≤ tol`, con `V = ½u'Pu + q'u` e duale `g(ν) = h'ν − ½ (q + G'ν)' P⁺ (q + G'ν)`, valido solo se `N'(q + G'ν) = 0` (altrimenti `nan`).
5. **Selezione del candidato.** Prezzo `Δ_J = V(u_J) − V_unc`, con `V_unc = −½ q'P⁺q`. Vince il prezzo minimo tra i candidati primali-fattibili. Parità di prezzo → norma minima di `u` → ordine lessicografico di `J` (deterministico).
6. **Infeasibilità.** Se nessun candidato certifica, distinguere infeasibilità da degenerazione con `interface.feasible_set.qp_feasible(G, h)` (LP su `u`): se il poliedro è vuoto → `found=False`; se è non vuoto ma nessun candidato passa → `found=False` senza eccezioni (stessa convenzione di `ExactQP`), ma il report deve dirlo.
7. **Scala delle righe.** Riusare la convenzione di `ExactQP`: normalizzare le righe di `G` a norma unitaria per i test di rango/indipendenza lineare e rimappare `ν` sulle righe originali. `P` non va regolarizzata: la sua scala entra nel prezzo e nella stazionarietà.

Nessuna nuova dipendenza: solo `numpy` + `scipy` (già in `pyproject.toml`).

## API richiesta

In `src/exactmpc/exact/singular.py`:

```python
class ExactSingularQP:
    """P ⪰ 0 fisso; precalcola V, Λ, P⁺, N e i set J in __init__."""
    def __init__(self, P, G, tol=1e-9, rcond=1e-12, max_active=None): ...
    @property
    def n_candidates(self) -> int: ...
    def solve(self, q, h) -> ExactResult: ...   # q: (n,) o (n,B); h: (κ,) o (κ,B)

def solve_exact_singular(P, q, G, h, tol=1e-9, rcond=1e-12, max_active=None) -> ExactResult:
    """Wrapper per un campione, B=1. Mantenere questo nome (già importato dal test)."""
```

Riusare **`ExactResult` da `exact/solver.py` senza modificarlo** (stessi campi e significati; `subsets_checked` = candidati visitati). In `solver.py` cambiare solo il messaggio d'errore: `"Use exactmpc.exact.singular (STUB)"` → `"Use exactmpc.exact.singular"`.

Aggiornare `src/exactmpc/exact/__init__.py` esportando `ExactSingularQP` e `solve_exact_singular`.

## Test da scrivere (`tests/test_singular.py`, sostituire lo xfail)

Tutti con `rng = np.random.default_rng(...)` e tolleranze strette, non allentate:

1. `test_singular_minimum_norm`: `P = diag(1, 0)`, box che lascia libera la seconda coordinata → `u₂ = 0` esatto (non il valore dello starting point), `ν ≥ 0`, certificato.
2. `test_singular_empty_active_set`: vincoli molto larghi (`h` grande) → `u = −P⁺q`, `ν = 0`, `price = 0`, `found`.
3. `test_singular_zero_hessian_lp`: `P = 0`, `q` con segni misti, box `|u| ≤ 1` → confronto con `scipy.optimize.linprog` (obiettivo e `u` a `1e-8`), `found`.
4. `test_singular_matches_dense_psd`: `P = B B'` con `B` (n×r) a rango r e `q ∈ range(P)`; confronto con `ExactQP(P + δI, G)` per `δ = 1e-9`: stesso obiettivo `V` (a `1e-7`), stesso set attivo, `ν` a `1e-6`. Solo su istanze con soluzione univoca.
5. `test_singular_dual_drive`: `models.plants.dual_drive()` + `CondensedMPC(plant, f=20, w3=0.0, limits=Limits(u_max=..., u_steps=(0,), z_hi=..., z_steps=(20,)))`, con `u_max`/`z_hi` ai quantili della soluzione libera come in `experiments/h1_certificates.py` (κ = 3, così l'enumerazione resta piccola); batch di ~200 campioni. Verificare:
   - H1 via `certificate_summary` (tolleranze `1e-9`, stazionarietà `1e-8`);
   - `P⁺P u = u` (nessuna componente in `null(P)`);
   - confronto `V` con `solve_slsqp` avviato da un `x0` NON di norma minima: differenza di `V` ≤ `1e-6`, mentre `u` può differire nella parte di `null(P)` → asserire esplicitamente `‖u_exact‖ ≤ ‖u_slsqp‖ + 1e-8` (questo è il risultato di H5).
6. `test_singular_infeasible_is_flagged`: due righe contraddittorie (`u ≤ −1` e `u ≥ 1`) → `found=False`, nessuna eccezione.
7. `test_singular_batch_equals_single`: 20 campioni, batch == single a `1e-12` (come `test_batch_equals_single`).

Non scrivere test di prezzo/gradiente in questo ciclo (per `P ⪰ 0` il prezzo è la differenza `V − V_unc`: basta asserire `price ≥ −1e-9`).

## Documentazione

- `docs/theory_status.md`: aggiornare la riga **T3** a `Implemented for inequalities, numerical evidence (gate G2)`, con `where = exact/singular.py, tests/test_singular.py`. Indicare l'ipotesi residua: nessuna LICQ richiesta; `found=False` se degenere, infeasibile o `max_active` troppo piccolo.
- `docs/hypotheses.md`: nessuna modifica delle soglie; H5 resta STUB a livello di esperimento.
- Non toccare `docs/roadmap.md` (il gate si chiuderà con H5, non con questo step).

## Criteri di accettazione (zero-error, da eseguire e riportare)

1. `.venv/bin/python -m pytest -q` → **tutti verdi, 0 xfail** (almeno 24 test).
2. Nessun `xfail` o `skip` nel nuovo file; nessuna tolleranza allentata per far passare i test.
3. `float64` ovunque; nessuna dipendenza nuova.
4. Il report `.bus/REPORT_B.md` deve contenere: commit, comandi eseguiti con esito, conteggio test, tabella dei certificati massimi (primale/duale/complementarità/stazionarietà/gap) e una nota esplicita su:
   - se `found=False` è mai comparso e perché;
   - se il sistema `M_J` è risultato inconsistente in qualche caso;
   - se la selezione di norma minima ha mai cambiato `u` rispetto a `lstsq` ingenuo, e di quanto.

## Fuori scope (non fare in questo ciclo)

`exact/price.py` per `P ⪰ 0`, `exact/pruning.py` per `P ⪰ 0`, `planner/layer.py`, `interface/GuardSet`, `experiments/h5_singular_case.py`, baseline `qp_solvers`. Se il gate G2 rivela che serve una seconda enumerazione su `z` (domanda aperta nel docstring), **fermarsi**, scriverlo nel report e proporre lo step successivo: non implementarlo silenziosamente.
