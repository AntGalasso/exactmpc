# TASK A → B — Sprint 1, singolo step: **T1b — chiudere e certificare il pruning batchato**

Data: 2026-11-20
Autore: Agente A (architetto)
Destinatario: Agente B (implementazione)
Riferimenti: `.bus/METAPROMPT.md`, `docs/theory_status.md`, `src/exactmpc/exact/{solver,pruning}.py`

> Questo file contiene **un solo step**. Non aprire sotto-task, non anticipare T2/T3/T4/T5.
> Se finito in anticipo, fermarsi e riportare.

---

## 0. Stato reale del repository (misurato da A, non stimato)

Ambiente: `.venv` presente e funzionante (`python3.13`, numpy 2.5.5).

| Verifica | Comando | Esito |
| --- | --- | --- |
| Suite corrente | `.venv/bin/python -m pytest -q` | `14 passed, 1 xfailed`, exit `0` |
| `solve_exact_pruned_batch` vs `ExactQP.solve`, 200 QP casuali (`n=8, k=5`) | script A | 200/200 confrontati, 0 mismatch (`1e-8`), 0 mancanti |
| batch vs ciclo su `solve_exact_pruned`, 40 campioni (`k=6`) | script A | 0 mismatch (`1e-10`) |
| righe `G` con scale `[1, 1e-6, 1]`, 100 istanze | script A | 0 mismatch single e batch vs `ExactQP` (`1e-7`) |
| efficienza osservata | stesso script | `visited/total` medio = **0.162** su `2^5-1 = 31` |

Quindi: la matematica del batch regge già su questi campioni. **Cosa manca davvero:**

1. `solve_exact_pruned_batch` **non è esportata** in `src/exactmpc/exact/__init__.py`.
2. **Non esiste alcun test** per il batch: `tests/test_pruning.py` (19 righe) copre solo il caso singolo.
3. Il test singolo non copre i casi richiesti da T1: dipendenza lineare, `visited < total` con limiti attivi, uguaglianza batch/ciclo.
4. Nessun numero di efficienza è registrato in modo riproducibile.
5. Nel batch il test dei sottoinsiemi "morti" è un doppio ciclo Python per campione
   (`any(D & ~Jmask == 0 for D in dead[b])`): corretto, ma `O(2^κ · B · |dead|)` con interpreti Python.
   È il punto che vanifica il vantaggio del batching proprio dove serve (B grande, κ grande).

Nota di contesto (non un bug da correggere qui): `ExactQP` normalizza le righe di `G`, il pruning no.
Il test di rango è invariante per scala (matrice di correlazione) e i campioni A non mostrano divergenze,
ma `tol` ha unità diverse nei due percorsi: va **verificato**, non assunto.

---

## 1. Obiettivo dello step

> Rendere `solve_exact_pruned_batch` un componente **esportato, testato e riproducibile** con le stesse
> garanzie di ottimalità dell'enumerazione completa, e rimuovere il collo di bottiglia Python nella
> propagazione dei sottoinsiemi morti.

Fuori perimetro: nessuna nuova API in `solver.py`, nessun `solve_ordered` (T2), nessuna modifica a
`singular.py`, `price.py`, piani, detector, plant, docs oltre a `docs/theory_status.md`.

---

## 2. Cosa implementare

### 2.1 Export e contratto

- Esportare `solve_exact_pruned_batch` in `src/exactmpc/exact/__init__.py` (aggiungere a `__all__`).
- **Mantenere** la firma attuale `(P, q, G, h, tol=1e-9, rcond=1e-12)` e il ritorno a 5 elementi
  `(u, nu, active, visited, total)`. Non introdurre `price` nel ritorno in questo step: il prezzo nei test
  si ricalcola da `nu` nella metrica del chiamante. La stabilità dell'interfaccia serve a T2.
- Aggiornare la docstring con una riga `STATUS:` esplicita (implementato + testato + da dove) e con la
  convenzione `NaN`: `u = nan` **sse** nessun candidato KKT trovato per quel campione.

### 2.2 Eliminare il ciclo Python sui sottoinsiemi morti

Sostituire la scansione per campione con una maschera vettoriale, senza cambiare la semantica:

- rappresentare i morti come matrice booleana `dead_masks` di forma `(κ, B)` (riga `j` = sottoinsieme
  singleton/set morto) **oppure** come maschera booleana `(2^κ, B)` se la memoria lo consente a `κ ≤ 12`;
  in alternativa una matrice `(κ, B)` di "profondità" non basta: la condizione è `D ⊆ J`, non `j ∈ J`.
  Scegliere la struttura e **dichiararla nel report** con il costo di memoria misurato.
- la condizione "`J` è morto" deve restare esattamente `∃ D morto con D ⊆ J`, con `D` anche di cardinalità
  maggiore di 1 (il codice attuale registra set di ogni dimensione).
- L'ordine di enumerazione per dimensione crescente e l'arresto anticipato non cambiano.

**Vincolo forte:** il risultato deve essere **identico** a quello della versione a ciclo (stessi `u`, `nu`,
`visited` per campione, bit a bit dove il calcolo è lo stesso). Se la vettorializzazione cambia anche un
solo `visited`, il test di equivalenza deve fallire e la colpa è dell'implementazione, non del test.

### 2.3 Casi limite da gestire senza eccezioni

- `q` o `h` monodimensionali (broadcast a batch, come già fa `ExactQP`);
- `B = 1`;
- campione con insieme attivo vuoto già ottimo (`r0 ≤ tol`): `found = True`, `visited` coerente;
- campione infeasible/degenere: `u = nan`, `nu = nan`, `found` implicito via `NaN` (contratto attuale);
- righe di `G` linearmente dipendenti: il ramo "dead" non deve perdere l'ottimo.

---

## 3. Test richiesti — `tests/test_pruning.py`

Aggiungere (senza toccare il test esistente, che resta com'è):

1. `test_batch_matches_full_enumeration_200`: 200 QP da `random_qp` (`n=8, k=5`), `u` del batch uguale a
   `ExactQP.solve` entro `1e-8`; dove `ExactQP` è `found`, il batch **non** può essere `NaN`.
   Il prezzo ricostruito `0.5 * r0' ν` (con `r0 = G(-P⁻¹q) - h`) deve coincidere con `res.price` entro `1e-8`
   **solo se** i due percorsi usano le stesse unità: in caso di discordanza documentare l'unità, non
   allargare la tolleranza.
2. `test_batch_equals_per_sample_loop`: stesso `(P, G)`, `B = 40`, `k = 6`; confronto con
   `solve_exact_pruned` campione per campione su `u` (`1e-10`) e `visited` (uguaglianza esatta).
3. `test_batch_visited_not_larger_than_total`: `visited <= total` sempre, e su un batch costruito con limiti
   attivi (`h` vicino al bordo) `visited.mean() < total.mean()`; riportare il rapporto medio nel report.
4. `test_batch_dependent_rows_keep_optimum`: `G` con una riga copia esatta di un'altra; il batch coincide con
   `ExactQP.solve` e non produce `NaN` dove `ExactQP` trova la soluzione.
5. `test_batch_row_scaling_parity`: stesse istanze riscalate come `test_solver.test_rows_of_very_different_scale_are_not_declared_dependent`
   (scale `[1, 1e-6, 1]`, `h` riscalato); batch vs `ExactQP` entro `1e-7`. Se fallisce, **correggere il
   pruning** (riusare la normalizzazione per riga già usata da `ExactQP`) e riportare la causa.
6. `test_batch_empty_active_set_and_nan_contract`: un campione con `J = ∅` ottimo e un QP infeasible
   (`G = [[1], [-1]]`, `h = [-1, -1]`): il primo `found`, il secondo tutto `NaN`, nessuna eccezione.

Non indebolire né rimuovere `test_pruned_equals_full_enumeration`. Non aggiungere `skip`/`xfail`.

---

## 4. Definition of Done

- `solve_exact_pruned_batch` esportata e documentata (`STATUS:` aggiornato).
- Tutti i test 1–6 verdi; suite completa verde (`14 passed, 1 xfailed` + nuovi).
- `docs/theory_status.md`: riga T6 aggiornata — soundness del pruning vale **anche in forma batchata**,
  con puntatore a `tests/test_pruning.py`; efficienza resta empirica (nessun claim H6).
- Report con: comando pytest, exit code, conteggi; rapporto medio `visited / (2^κ - 1)` misurato;
  costo di memoria della struttura di maschere scelta; eventuale divergenza di unità nel prezzo.

## 5. Accettazione (bloccante)

- `pytest` exit code `0`, nessun test allentato, nessun file di test esistente riscritto.
- 200 istanze: 0 discrepanze con l'enumerazione completa oltre `1e-8`, 0 `NaN` inattesi.
- Equivalenza bit a bit del `visited` tra batch vettorizzato e ciclo per campione.
- Nessuna modifica a `solver.py`, `price.py`, `singular.py`.

## 6. Formato del report (`.bus/REPORT_B.md`)

```
## T1b — Pruning batchato: export, test, vettorizzazione
Stato: PASS | FAIL | BLOCKED
Comandi: <comandi esatti>
Esito: <200 istanze, mismatch, visited/total medio, memoria maschere>
File: <path aggiunti/modificati>
Test: <n passed, n xfailed, exit code>
Rischi/Note: <unità di tol/prezzo, cosa resta per T2>
```

In coda, massimo 3 righe su cosa T2 eredita da questo step.
