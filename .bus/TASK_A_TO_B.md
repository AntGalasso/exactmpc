# TASK A → B — Sprint 1, singolo step: **T1b — pruning batchato: export, parità `visited`, vettorizzazione**

Data: 2026-11-20 (revisione)
Autore: Agente A (architetto)
Destinatario: Agente B (implementazione)
Riferimenti: `.bus/METAPROMPT.md`, `docs/theory_status.md`, `src/exactmpc/exact/{solver,pruning}.py`, `tests/conftest.py`

> **Questo file contiene un solo step.** Non aprire sotto-task, non anticipare T2/T3/T4/T5.
> Se finisci prima, fermati e riporta. Il `.bus/REPORT_B.md` attuale è un dump di terminale
> corrotto (screenshot, non un risultato): **non contiene alcuna evidenza** e va sovrascritto.

---

## 0. Stato reale misurato da A (non stimato, rng `default_rng(0)`)

Ambiente: `.venv/bin/python` = python3.13, numpy 2.5.5.

| Verifica | Comando | Esito |
| --- | --- | --- |
| Suite corrente | `.venv/bin/python -m pytest -q` | **19 test raccolti: `18 passed, 1 xfailed`**, exit `0` (`--collect-only`: test_condense 4, test_feasible_set 2, test_interface 1, test_price 4, test_pruning 1, test_singular 1, test_solver 6) |
| Export batch | `grep -n solve_exact_pruned_batch src/exactmpc/exact/__init__.py` | **assente** (import e `__all__` non lo contengono) |
| Test batch | `grep -c batch tests/test_pruning.py` | **assente**; il file copre solo il caso singolo |
| batch vs `ExactQP.solve`, 200 QP casuali `n=8, k=5` | probe A | 0 mismatch su `u` (`1e-8`), 0 `NaN` inattesi; `visited/(2^5-1)` medio = **0.3379** |
| batch vs ciclo su `solve_exact_pruned`, `B=40`, `k=6` | probe A | 0 mismatch su `u` (`1e-10`) ma **`visited` diverso in 4/40 campioni** |
| righe dipendenti (`G[3]=G[1]`, `h[3]=h[1]`), 20 istanze | probe A | 0 `NaN` dove `ExactQP` trova, 0 mismatch |
| scale righe `[1, 1e-6, 1, 1, 1]`, 100 istanze | probe A | max \|Δu\| = **5.0e-16** (nessuna divergenza) |
| `k > n` (`n=3, k=6`) | probe A | `total = 63`, ma l'enumerazione si ferma a `min(k,n)=3` → **41 sottoinsiemi raggiungibili** |
| QP infeasible (`G=[[1],[-1]]`, `h=[-1,-1]`) | probe A | `u=NaN`, `nu=NaN`, `visited=2`, `total=3`, nessuna eccezione → contratto ok |
| insieme attivo vuoto già ottimo (`P=I, q=0, h>0`) | probe A | `u` corretto ma **`visited=2`** vs **`visited=0`** del percorso singolo |

### Due difetti reali (non cosmetici)

1. **`visited` non è in parità col percorso singolo (4/40).** Causa individuata: `solve_exact_pruned`
   ritorna subito con `visited=0` quando `r0 <= tol`; il batch marca quei campioni `found`/`best=0`
   ma **continua a enumerare e a incrementare `visited`**. La riga di commento nel codice
   ("harmless: they just accumulate visited counts") è la fonte dell'incoerenza. Va corretta:
   il contratto di `visited` è "sottoinsiemi effettivamente valutati", e un campione già risolto
   dal test `r0 <= tol` non ne valuta nessuno.
2. **`total` non corrisponde all'enumerazione.** Entrambe le funzioni scrivono `2**kappa - 1`,
   ma i cicli si fermano a `min(kappa, n)`. Per `kappa > n` `total` è un limite superiore non
   raggiungibile e il rapporto di efficienza riportato è falsato. Il contratto corretto è
   `total = Σ_{s=1..min(kappa,n)} C(kappa, s)`.

Il collo di bottiglia Python resta quello descritto: il test dei sottoinsiemi morti è
`any(D & ~Jmask == 0 for D in dead[b])` dentro un ciclo su `B`, cioè `O(2^κ · B · |dead_b|)`
interpretato, proprio dove il batch dovrebbe servire.

Nota di unità (da **verificare**, non assumere): `ExactQP` normalizza le righe di `G`, il pruning
no. Il test di rango usa la matrice di correlazione (invariante), quindi sul campione A non c'è
divergenza, ma `tol` ha significato diverso nei due percorsi.

---

## 1. Obiettivo dello step

> `solve_exact_pruned_batch` diventa un componente **esportato, testato, in parità esatta** col
> percorso singolo, con il `total` che descrive davvero l'enumerazione, e senza il ciclo Python
> sui sottoinsiemi morti. Nessun claim di efficienza (H6 resta empirica).

Fuori perimetro: nessuna nuova API in `solver.py`, nessun `solve_ordered` (T2), nessuna modifica a
`singular.py`, `price.py`, piani, detector, plant. In `docs/` solo la riga T6 di
`docs/theory_status.md`.

---

## 2. Cosa implementare

### 2.1 Export e contratto d'interfaccia

- Esportare `solve_exact_pruned_batch` in `src/exactmpc/exact/__init__.py` e in `__all__`.
- **Mantenere** firma `(P, q, G, h, tol=1e-9, rcond=1e-12)` e ritorno a 5 elementi
  `(u, nu, active, visited, total)`. Niente `price` nel ritorno in questo step: T2 ne dipende,
  e la stabilità dell'interfaccia vale più della comodità. Il prezzo nei test si ricalcola da `nu`.
- Docstring: riga `STATUS:` esplicita (implementato + testato + file di test) e convenzione
  `NaN` dichiarata — `u` è `NaN` **sse** nessun candidato KKT è stato trovato per quel campione.

### 2.2 Parità di `visited` (difetto 1)

- I campioni con `r0 <= tol` devono essere **esclusi dall'enumerazione** (non solo marcati
  `found`), così `visited[b] == 0` come nel percorso singolo.
- Vale anche quando in un batch **solo alcuni** campioni sono in questa condizione: la maschera
  `alive` deve escluderli, senza alterare gli altri.
- `visited` resta la cardinalità dei sottoinsiemi valutati, con `visited <= total` per costruzione.

### 2.3 Contratto di `total` (difetto 2)

- Introdurre un unico helper (privato, in `pruning.py`) che calcoli
  `total = Σ_{s=1..min(kappa, n)} C(kappa, s)` e usarlo **in entrambe** le funzioni
  (`solve_exact_pruned` e `solve_exact_pruned_batch`), così che `total` descriva l'enumerazione
  reale anche per `kappa > n`.
- Documentarlo nella docstring: `total` è il numero di sottoinsiemi enumerabili, non `2^κ - 1`.

### 2.4 Eliminare il ciclo Python sui sottoinsiemi morti (vettorializzazione)

Struttura richiesta: `dead_rows: dict[int, np.ndarray]` che mappa **maschera di bit del
sottoinsieme morto** → vettore booleano di forma `(B,)` (`True` = quel campione ha quel morto).

- Il test "`J` è morto" diventa
  `alive = ~logical_or.reduce([row for D, row in dead_rows.items() if (D & Jmask) == D])`,
  cioè `D ⊆ J`, valutato con operazioni numpy su tutti i campioni insieme.
- La maschera è **condivisa** tra i campioni che hanno lo stesso morto: il costo è
  `O(num_maschere_distinte · B)` operazioni vettoriali per candidato, non `O(B · |dead_b|)` Python.
- Si possono eliminare da `dead_rows` le maschere di cardinalità `>= size` corrente (non possono
  essere sottoinsiemi di un candidato di cardinalità `size`); dichiararlo nel report.
- Ordine di enumerazione per dimensione crescente e arresto anticipato invariati.

**Vincolo forte:** a parità di `(P, q_col, G, h_col)` il risultato deve essere **identico** al
percorso singolo: stessi `u`, `nu`, `visited` (uguaglianza esatta degli interi), stessa `active`.
Se la vettorializzazione cambia anche un solo `visited`, il difetto è dell'implementazione.

### 2.5 Casi limite, senza eccezioni nuove

- `q` o `h` monodimensionali (broadcast a batch, comportamento attuale);
- `B = 1`; `kappa = 0` (nessun vincolo: `u = -P^{-1}q`, `visited = 0`, `total = 0`);
- campione già ottimo con `J = ∅` (`visited = 0`);
- campione infeasible/degenere: `u = NaN`, `nu = NaN`, `active` tutta `False`;
- righe di `G` linearmente dipendenti (copia esatta): il ramo "dead" non deve perdere l'ottimo;
- `kappa > n`: nessun accesso fuori range, `total` secondo 2.3.

---

## 3. Test richiesti — `tests/test_pruning.py`

Il test esistente `test_pruned_equals_full_enumeration` **resta intatto**. Aggiungere:

1. `test_batch_matches_full_enumeration_200`: 200 QP da `random_qp(n=8, k=5)`; `u` del batch
   uguale a `ExactQP.solve` entro `1e-8`; dove `ExactQP.found`, il batch non può essere `NaN`.
   Il prezzo ricostruito `0.5 * r0 @ nu` (`r0 = G(-P^{-1}q) - h`) deve coincidere con
   `res.price` entro `1e-8`. Se le unità divergono, **documentare l'unità nel report**; non
   allargare la tolleranza e non normalizzare dentro il test.
2. `test_batch_equals_per_sample_loop`: stesso `(P, G)`, `B = 40`, `k = 6`, `q`/`h` casuali per
   colonna; confronto con `solve_exact_pruned` su `u` (`1e-10`), `nu` (`1e-10`), `active`
   (uguaglianza esatta) e **`visited` (uguaglianza esatta)**. Questo test deve **fallire** sul
   codice attuale: è la regressione del difetto 1.
3. `test_batch_empty_active_set_parity`: batch misto in cui alcune colonne hanno `r0 <= tol`
   (es. `P=I, q=0`, `h` con componenti positive e negative); per quelle colonne `visited == 0`,
   per le altre `visited == ` valore del percorso singolo. Verifica esplicita del difetto 1.
4. `test_total_matches_enumerable_subsets`: per `kappa <= n` e per `kappa > n` (es. `n=3, k=6`)
   `total` vale `Σ_{s=1..min(k,n)} C(k,s)`, vale sia per il singolo sia per il batch, e
   `visited <= total`. Verifica esplicita del difetto 2.
5. `test_batch_visited_not_larger_than_total`: `visited <= total` sempre; su un batch con limiti
   attivi (`h` vicino al bordo, es. `h = G u‡` per un `u‡` ammissibile) `visited.mean() <
   total.mean()`. Riportare il rapporto medio misurato nel report.
6. `test_batch_dependent_rows_keep_optimum`: `G` con una riga copia esatta di un'altra (**e `h`
   coerente con la copia**); batch uguale a `ExactQP.solve`, nessun `NaN` dove `ExactQP` trova.
7. `test_batch_row_scaling_parity`: stesse istanze riscalate come in
   `test_solver.test_rows_of_very_different_scale_are_not_declared_dependent` (scale
   `[1, 1e-6, 1, ...]`, `h` riscalato coerentemente); batch vs `ExactQP` entro `1e-7`. Se fallisce,
   correggere il pruning riusando la normalizzazione per riga di `ExactQP` e riportare la causa.
8. `test_batch_nan_contract_infeasible`: `P=I, q=0, G=[[1],[-1]], h=[-1,-1]` → `u`, `nu` tutti
   `NaN`, `active` tutta `False`, nessuna eccezione, `visited <= total`.

Vietato: `skip`, `xfail`, tolleranze allargate per far passare un test, riscrittura dei test
esistenti.

---

## 4. Definition of Done

- `solve_exact_pruned_batch` esportata, con `STATUS:` e contratto `NaN`/`total` documentati.
- Difetto 1 chiuso: `visited` in parità esatta col percorso singolo (test 2 e 3 verdi).
- Difetto 2 chiuso: `total` = sottoinsiemi enumerabili, in entrambe le funzioni (test 4 verde).
- Dead-set test vettoriale via `dead_rows` (nessun ciclo Python su `B`), `visited` invariato.
- Test 1–8 verdi; **suite completa verde**: `18 passed, 1 xfailed` + i nuovi, exit `0`.
- `docs/theory_status.md`, riga **T6**: soundness del pruning vale anche in forma batchata, con
  puntatore a `tests/test_pruning.py`; efficienza resta empirica (nessun claim H6). Annotare la
  convenzione di `total`.
- Report con: comandi pytest esatti, exit code, conteggi; rapporto medio `visited / total`
  misurato; memoria della struttura `dead_rows` misurata (numero di maschere distinte e byte);
  eventuale divergenza di unità nel prezzo.

## 5. Accettazione (bloccante)

- `pytest` exit code `0`; nessun test allentato, nessun file di test esistente riscritto.
- 200 istanze: 0 discrepanze con l'enumerazione completa oltre `1e-8`, 0 `NaN` inattesi.
- `visited` bit a bit identico tra batch vettorizzato e ciclo per campione (inclusi i campioni
  con insieme attivo vuoto ottimo: `0`).
- `total` coerente con l'enumerazione anche per `kappa > n`, in entrambe le funzioni.
- Nessuna modifica a `solver.py`, `price.py`, `singular.py`.

## 6. Formato del report (`.bus/REPORT_B.md`, sovrascrivere integralmente)

```
## T1b — Pruning batchato: export, parità visited, vettorizzazione
Stato: PASS | FAIL | BLOCKED
Comandi: <comandi esatti>
Esito: <200 istanze: mismatch/NaN; parità visited (n colonne); visited/total medio; memoria dead_rows>
File: <path aggiunti/modificati>
Test: <n passed, n xfailed, exit code>
Rischi/Note: <unità tol/prezzo, total per kappa>n, cosa resta per T2>
```

In coda, **massimo 3 righe** su cosa T2 eredita da questo step (interfaccia a 5 elementi,
semantica di `visited`, contratto `total`).
