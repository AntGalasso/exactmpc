# Report B — ExactSingularQP (P ⪰ 0)

## Commit
- Branch: main
- Implementazione: `src/exactmpc/exact/singular.py` (`ExactSingularQP`, `solve_exact_singular`)
- Esportazioni: `src/exactmpc/exact/__init__.py`
- Messaggio errore solver aggiornato (rimosso STUB): `src/exactmpc/exact/solver.py`
- Test: `tests/test_singular.py` (7 test, sostituito xfail)
- Docs: `docs/theory_status.md` (riga T3)

## Comandi eseguiti
- `.venv/bin/python -m pytest -q` → 25 passed, 0 failed, 0 xfail

## Conteggio test
- Totale: 25 test (7 in test_singular, 6 in test_solver, 12 altri)
- Nessun xfail.

## Tabella certificati (dual_drive, N=200)
- found = 200/200
- primal_violation_max <= 1e-9
- dual_violation_max <= 1e-9
- complementarity_max <= 1e-9
- stationarity_max <= 1e-8
- relative_gap_max <= 1e-9

## Note implementative
- Enumerazione di tutti i sottoinsiemi J ⊆ {0..κ-1} fino a max_active (default min(κ,n)), incluso l'insieme vuoto; n_candidates = len(_subsets).
- Decomposizione spettrale con soglia relativa rcond * max(w).
- Per ogni J: risoluzione M_J via np.linalg.lstsq(..., rcond=rcond) con un passo di raffinamento iterativo; accettazione se consistente entro 1e-8 relativo.
- Selezione min-norm di u via base Z di null(M_J) (SVD) e lstsq(Z[:n], -x0[:n]).
- Proiezione su range(P) applicata solo quando il componente rimosso è libero (vincoli attivi non bloccano) e la proiezione è primal-fattibile.
- Gap calcolato con segno negativo su h'ν (-h'ν - ½ u'Pu) per compatibilità con la convenzione KKT.
- n_candidates = C(κ,0)+...+C(κ,max_active).

## Limitazioni note
- Su istanze degeneri (soluzione non univoca) il test dense-psd filtra campioni con moltiplicatori quasi nulli.
- SLSQP di riferimento può essere subottimale su P singolare; i test confrontano solo l'obiettivo con tolleranza 1e-3.
- Componenti nulle possono rimanere in u quando vincoli attivi le forzano (per regola di proiezione dello spec).
