# PIANO OPERATIVO A → B — Sprint 1

Data: 2026-11-20
Autore: Agente A (architetto)
Destinatario: Agente B (implementazione)
Riferimenti: `.bus/METAPROMPT.md`, `README.md`, `docs/roadmap.md`, `docs/theory_status.md`, `docs/hypotheses.md`

---

## 0. Protocollo di lavoro (vincolante)

1. B legge questo file, esegue i task **in ordine di priorità** e scrive l'esito in `.bus/REPORT_B.md`
   (una sezione per task: `PASS` / `FAIL` / `BLOCKED`, comandi eseguiti, numeri misurati, file toccati,
   rischi residui). Solo a report completo crea/aggiorna `.bus/ready_for_a`.
2. **Nessun risultato inventato.** Ogni numero nel report deve venire da un comando riportato.
   I numeri ottenuti sul plant placeholder non sono risultati del progetto e vanno etichettati come tali.
3. **Zero-error:** `pytest` deve restare verde (incluso l'`xfail` stretto sul caso singolare) prima di
   chiudere ogni task. Vietato indebolire asserzioni, allargare tolleranze esistenti, marcare `skip`
   o aggiungere `xfail` per far passare un test. Se un test esistente è sbagliato, si argomenta nel report
   e lo si corregge solo con motivazione esplicita.
4. **float64 ovunque**: nessun cast a `float32`; i certificati KKT e i controlli di validità girano in
   `float64`. Ogni nuovo modulo inizia con una riga `STATUS:` (implementato / testato / stub), come da
   convenzione del repository.
5. **Niente refactor gratuiti.** Non rinominare il package, non spostare file, non cambiare API pubbliche
   esistenti senza necessità: i test e gli esperimenti attuali devono continuare a funzionare.
   Nuove dipendenze solo se aggiunte a `pyproject.toml` negli extra già previsti.
6. Se un task è bloccato (dipendenza mancante, decisione teorica aperta), B lo dichiara `BLOCKED` con la
   domanda precisa e passa al successivo. Non apre sotto-task non previsti.

---

## 1. Audit dello stato reale (Fase 0 — fatto da A)

### 1.1 Cosa è già implementato e testato

| Componente | File | Stato |
| --- | --- | --- |
| QP esatto `P ≻ 0`, enumerazione attiva, certificato KKT, batching su `q,h` | `src/exactmpc/exact/solver.py` | Implementato; coperto da `tests/test_solver.py` (SLSQP, soglie H1, batch=single, scala righe) |
| Prezzo `Δ`, gradiente di valore/prezzo, prezzo del margine (T4) | `src/exactmpc/exact/price.py` | Implementato; coperto da `tests/test_price.py` |
| Pruning sound **per singolo QP** (T6) | `src/exactmpc/exact/pruning.py` | Implementato; soundness testata, **non batched**, efficienza non stabilita |
| MPC condensato e limiti | `src/exactmpc/mpc/{condense,limits}.py` | Implementato; testato |
| Poliedro `A0` senza limiti, `F` e scala massima fattibile | `src/exactmpc/interface/{polyhedron,feasible_set}.py` | Implementato; testato |
| Metriche di certificato (H1) | `src/exactmpc/eval/metrics.py` | Implementato |
| Plant placeholder (feed drive, dual drive, doppio integratore) | `src/exactmpc/models/plants.py` | Implementato, **parametri placeholder** |
| Esperimento H1 | `experiments/h1_certificates.py` | Eseguibile sul placeholder |

### 1.2 Stub / mancante

- `exact/singular.py`: `P ⪰ 0` con disuguaglianze — stub, gate **G2**, T3, H5. Test `xfail` stretto.
- `interface/feasible_set.GuardSet`: stub (T2, H4); esistono solo `qp_feasible` e `max_feasible_scale`.
- `planner/{network,layer,loss}.py`: stub; `layer.py` richiede torch e la derivata esatta `du/dθ`.
- `baselines/{qp_solvers,optnet_layer,topp,hose_verified}.py`: stub; solo SLSQP di riferimento.
- `experiments/h2…h7`: stub con messaggio d'uscita.
- **Scalabilità oltre `κ = 20`**: non esiste né enumerazione efficiente né API di valutazione ordinata
  dei candidati (`J_{t-1}`, Top-K proposti). `mpc/limits.py` impone `κ ≤ 6` per design.
- **Zero-Trust Active-Set Detector**: assente; nessun contratto di interfaccia tra detector e solver.
- **Dati / split / Sobol / random walk**: assenti (`src/exactmpc/data` non esiste).
- **Launcher remoto** (`launch_adaption.py`, `train_cluster.py`, `ADAPTION_API_KEY`): assente.

### 1.3 Rischi e discrepanze da tenere presenti

1. **Layout**: il METAPROMPT parla di `src/exact_core/`, `src/plants/`, `src/models/`, `src/experiments/`;
   il repo usa `src/exactmpc/...` + `experiments/`. **Decisione A:** si estende il layout esistente,
   senza rinominare (mappa in §1.4). Motivo: non rompere import, test ed esperimenti.
2. **Target κ**: il target scientifico è `κ ≫ 20`, ma l'enumerazione attuale è `2^κ`. Lo Sprint 1 non
   risolve lo scaling: prepara il **contratto** (valutazione ordinata dei candidati + fallback) che lo
   rende possibile, senza ancora addestrare reti.
3. **Ambiente**: in questa macchina `python3` **non ha `numpy` né `pytest` installati** (verificato:
   `ModuleNotFoundError`). Il claim "18 passed, 1 xfail" del README è quindi **non verificato qui**:
   B deve installare e misurare (T0). Nessuna conclusione sullo stato dei test prima di T0.
4. **Plant placeholder**: nessun risultato di controllo (H1, H2, H5, H7) è pubblicabile finché non arriva
   il modello CNC del lab. Gli esperimenti dello Sprint 1 sono test di correttezza, non risultati.
5. **Tolleranze**: `tol=1e-9` è usato sia per fattibilità sia per segno dei moltiplicatori. Va
   documentato che non è una tolleranza sull'ottimo ma una soglia di certificato (H1).
6. **`singular.py` e disuguaglianze**: la domanda "la faccia ottima ammette ancora il bound `2^κ` o serve
   una seconda enumerazione su `z`?" è il gate G2. Non va decisa per tentativi: prima il caso con sole
   uguaglianze, poi un memo di decisione.

### 1.4 Mappa METAPROMPT → repository

| METAPROMPT | Dove va nel repo |
| --- | --- |
| `src/exact_core/` (motore esatto) | `src/exactmpc/exact/` + `src/exactmpc/interface/` |
| `src/plants/` | `src/exactmpc/models/plants.py` |
| `src/data/` | nuovo `src/exactmpc/data/` (Sprint 2) |
| `src/models/` (detector, planner) | `src/exactmpc/planner/` + nuovo `src/exactmpc/detector/` (Sprint 3) |
| `src/experiments/` | `experiments/` |
| `launch_adaption.py` / `train_cluster.py` | `scripts/` (Sprint 4) |

---

## 2. Obiettivo dello Sprint 1

> Rendere il motore esatto un componente **riutilizzabile, batchato e verificato** per `κ` arbitrario:
> pruning batchato, API di valutazione ordinata dei candidati con regola Zero-Trust e fallback,
> caso singolare `P ⪰ 0` con sole uguaglianze e decisione scritta sul gate G2, più l'ambiente
> riproducibile e la baseline dei test misurata.

Fuori perimetro (Sprint 2+): reti neurali/detector, training, planner, plant reale, dati, launcher,
esperimenti H2/H3/H7 come risultati. In questo sprint si toccano solo moduli esatti + test + docs.

---

## 3. Task dello Sprint 1

### T0 — Ambiente riproducibile e baseline misurata (P0, bloccante)

**Cosa**
- Creare un ambiente installabile e installare il progetto: `python3 -m pip install -e ".[dev]"`
  (se il sistema non lo consente, virtualenv locale `.venv`, che va in `.gitignore`).
- Eseguire `python3 -m pytest` e `make h1` e registrare l'output.

**Definition of Done**
- `pytest` eseguito con exit code riportato; conteggio esatto di passed/failed/xfailed.
- `experiments/h1_certificates.py` gira e il report riporta i numeri stampati (etichettati placeholder).
- Nessuna modifica al codice in questo task (solo eventuale `.gitignore` per `.venv`).

**Accettazione**
- Report con: versione Python, numpy, scipy, pytest, comando esatto, exit code, output sintetico.
- Se `pytest` non è verde, **ferrarsi e riportare `FAIL`**: non si prosegue su T1 prima di aver capito.

---

### T1 — Pruning batchato e confronto con l'enumerazione completa (P0)

**Cosa**
- Aggiungere in `src/exactmpc/exact/pruning.py` una variante batchata
  `solve_exact_pruned_batch(P, q, h, tol=1e-9, rcond=1e-12)` con `q: (n,B)`, `h: (κ,B)`, che riusi
  `T`, `u*`, `r0` precalcolati una volta per batch e restituisca per campione
  `(u, nu, active, visited, total)`; `visited` deve essere la media/array per campione.
- Riusare la logica di potatura già dimostrata (sottoinsiemi dipendenti morti, bound monotono `Δ_J`),
  **senza cambiare** le garanzie di soundness esistenti.
- Test nuovi in `tests/test_pruning.py`:
  - su 200 QP casuali (`random_qp`), `u` e `price` coincidono con `ExactQP.solve` entro `1e-8`;
  - `visited ≤ total` e `sum(visited) < sum(total)` su un batch con limiti attivi;
  - uguaglianza batch vs ciclo sui singoli campioni;
  - caso con righe linearmente dipendenti: il ramo "dead" non perde l'ottimo.

**Definition of Done**
- Funzione esportata in `exact/__init__.py`; docstring con `STATUS:` aggiornato.
- Tutti i test nuovi verdi; test esistenti intatti.
- Report con il rapporto medio `visited / (2^κ - 1)` misurato (dato di soundness/efficienza, non claim H6).

**Accettazione**
- Nessuna discrepanza con l'enumerazione completa oltre `1e-8` su 200 istanze.
- Nessuna modifica alle tolleranze dei test esistenti.

---

### T2 — API di valutazione ordinata dei candidati e regola Zero-Trust (P0)

Questo è il task **architetturalmente centrale**: definisce il contratto che il detector userà.

**Cosa**
- In `src/exactmpc/exact/solver.py` aggiungere a `ExactQP`:
  ```python
  solve_ordered(self, q, h, order, max_candidates=None) -> OrderedResult
  ```
  dove `order` è una sequenza di candidati per campione (lista di liste di indici, oppure array
  `(B, K, max_active)` con padding `-1`), valutati **nell'ordine dato** con la formula chiusa e il
  certificato KKT; ci si arresta al **primo** candidato valido (`nu ≥ -tol`, primale fattibile).
  - Candidato vuoto `J = ∅` sempre valutabile.
  - Se nessun candidato proposto è valido, si applica il **fallback** documentato: enumerazione completa
    (via `ExactQP.solve`) oppure, in alternativa, enumerazione per dimensione crescente; la scelta va
    motivata nel report.
  - Restituire: `u`, `nu`, `active`, `price`, `found`, i campi di certificato già presenti in
    `ExactResult`, più `candidates_evaluated` (per campione), `stopped_at` (indice nell'ordine o `-1`),
    `fallback_used` (bool per campione).
- Proprietà da testare in un nuovo `tests/test_candidates.py`:
  1. **0% subottimalità**: per ogni campione, se il risultato è `found`, il costo coincide con quello
     di `ExactQP.solve` entro `1e-9` (relativo), cioè il primo candidato valido è l'ottimo.
  2. **Ordine irrilevante per l'ottimo**: permutando l'ordine dei candidati, `u`, `price` restano
     invariati e `candidates_evaluated` cambia correttamente.
  3. **Early stop**: se `order[0]` è l'active set vero, `candidates_evaluated == 1` e
     `fallback_used == False`.
  4. **Fallback**: con un `order` di soli candidati non validi, `fallback_used == True` e il risultato
     coincide con l'enumerazione completa.
  5. **Robustezza**: `order` con indici duplicati, `J` non indipendente o oltre `max_active` non deve
     produrre eccezioni non gestite: il candidato va scartato e contato come valutato.
- Aggiungere in `docs/theory_status.md` una riga che dichiari: "Zero-Trust: il primo candidato KKT valido
  è l'ottimo; la rete può solo accelerare, mai degradare" con il puntatore al test.

**Definition of Done**
- API esportata, docstring `STATUS:` aggiornata, 5 test verdi, `pytest` completo verde.
- Report con la distribuzione di `candidates_evaluated` su un batch sintetico (ordine = active set
  dell'istanza precedente simulata + rumore): media, mediana, massimo, % fallback.

**Accettazione**
- Su 500 campioni casuali: 0 casi `found` con costo superiore all'ottimo oltre `1e-9`; nessun crash.
- Il fallback non è mai silenzioso: sempre segnalato per campione.

---

### T3 — Caso singolare `P ⪰ 0`: sole uguaglianze + memo G2 (P1)

**Cosa**
- Implementare in `src/exactmpc/exact/singular.py` il caso **con sole uguaglianze** `C u = d`:
  - soluzione particolare con pseudo-inversa `P⁺`;
  - test di range per la fattibilità a costo zero (capacità): `d - C u_p ∈ range(C V_N)`,
    `V_N` base ortonormale di `null(P)`;
  - selezione **deterministica di norma minima**;
  - prezzo oltre capacità secondo la formula già annotata nel docstring (`r_perp'(B Σ^{-2} B')⁺ r_perp`),
    da verificare numericamente contro una soluzione di riferimento (SLSQP) su istanze piccole.
- **Non** implementare ancora le disuguaglianze sulla faccia ottima.
- Scrivere `docs/g2_decision.md` (max 1 pagina): due opzioni (a) enumerazione `2^κ` sulla faccia,
  (b) secondo livello su `z`); costo, correttezza, esperimento minimo che discrimina; **raccomandazione**
  di A→B ma decisione finale rimandata all'esito dell'esperimento. Aggiornare `docs/theory_status.md` (T3).
- Test: sostituire l'`xfail` **solo** se l'implementazione delle uguaglianze è completa; in tal caso
  mantenere un test separato che continui a marcare le disuguaglianze come non implementate
  (es. `NotImplementedError` esplicito con messaggio G2). Nessuna rimozione silenziosa del test.

**Definition of Done**
- Uguaglianze: almeno 4 test (range soddisfatto, oltre capacità, min-norm unica, determinismo bit a bit
  su due chiamate).
- `docs/g2_decision.md` presente e coerente con il codice.
- Report: esito numerico su 20 istanze (errore vs SLSQP) e stato del gate G2 (`OPEN` / `CLOSED` con esito).

**Accettazione**
- Errore relativo sul costo `≤ 1e-8` dove SLSQP è affidabile (istanze ben condizionate), o spiegazione
  esplicita del perché il confronto non è conclusivo.
- Nessun claim su H5: la parte con disuguaglianze resta aperta.

---

### T4 — Gradiente esatto di `u` rispetto a `θ` (P1, abilitante per il layer)

**Cosa**
- In `src/exactmpc/exact/price.py` aggiungere `input_gradient(qp, res, Qth, Hth)` che restituisce
  `du/dθ` per campione usando la formula chiusa sull'active set:
  `du/dθ = -P⁻¹ (Qth + G_J' dν_J/dθ)`, `dν_J/dθ = -S_J⁻¹ (G_J P⁻¹ Qth + Hth_J)`,
  con `ν = 0` fuori da `J`.
- Verifica contro differenze finite centrali di `ExactQP.solve` **dentro** un intorno in cui l'active set
  non cambia (passo scelto in modo che `active` resti identico; il test deve asserire l'identità).
- Test in `tests/test_price.py`: 5 istanze, errore relativo `≤ 1e-6`; caso `J = ∅` (gradiente
  `-P⁻¹Qth`); caso con più righe attive.
- Aggiornare la docstring di `planner/layer.py` indicando che la derivata ora esiste e dove.

**Definition of Done**
- Funzione esportata; test verdi; nessun uso di torch in questo task.
- Report: errore massimo misurato e numero di campioni usati.

**Accettazione**
- Nessun test che passi solo grazie a un passo di differenze finite degenere: il test verifica anche
  che l'active set sia costante nell'intorno.

---

### T5 — Igiene float64 e determinismo (P2)

**Cosa**
- Audit di `src/exactmpc/**` per cast `float32`, `dtype=object`, ordinamenti non deterministici.
- Aggiungere `tests/test_determinism.py`: due esecuzioni identiche di `ExactQP.solve` su batch fisso
  producono bit a bit lo stesso `u`, `nu`, `price`; l'ordine dei sottoinsiemi generati è stabile.
- Documentare in `README.md` (sezione convenzioni) la politica NaN: `u = nan` sse `found = False`.

**Definition of Done**
- Nessun `float32` nel package; test di determinismo verde; README aggiornato di 2-3 righe.

---

## 4. Ordine di esecuzione e dipendenze

```
T0 ──> T1 ──> T2 ──> (T4)
                └──> T3 (indipendente da T2, dopo T0)
T5 dopo T1 (tocca gli stessi file; evitare conflitti)
```

- T1 prima di T2: il fallback di T2 riusa il percorso di enumerazione/pruning verificato.
- T3 non blocca T2 ma il suo memo G2 va consegnato anche se `BLOCKED` sul codice.
- Se il tempo è limitato: T0, T1, T2 sono obbligatori; T4 e T3 sono il minimo per lo Sprint 2.

---

## 5. Criteri globali di chiusura dello Sprint 1

1. `pytest` verde (nuovi test inclusi) con exit code e conteggi nel report.
2. `make h1` eseguito, output riportato e etichettato come placeholder.
3. API nuove documentate ed esportate; `STATUS:` aggiornati.
4. `docs/theory_status.md` aggiornato per T1, T3, Zero-Trust; `docs/g2_decision.md` presente.
5. `.bus/REPORT_B.md` completo, con per ogni task: comando, esito, numeri, file, rischi.
6. Nessun risultato scientifico dichiarato: lo Sprint 1 produce infrastruttura e certificati di
   correttezza, non evidenza per H2/H3/H5/H7.

## 6. Cosa A si aspetta di trovare nel REPORT_B (formato)

```
## T<n> — <titolo>
Stato: PASS | FAIL | BLOCKED
Comandi: <comandi esatti>
Esito: <numeri misurati, con unità>
File: <path aggiunti/modificati>
Test: <n passed, n xfailed, exit code>
Rischi/Note: <cosa resta aperto>
```

Con in testa una sezione `## Ambiente` (T0) e in coda `## Proposte per lo Sprint 2`
(max 5 righe, ordinate per impatto: detector, dati, plant reale, layer/training, launcher).
