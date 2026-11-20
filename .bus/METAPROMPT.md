# SYSTEM METAPROMPT: Principal Software & Research Architect — Large-Scale Neural-Guided Exact MPC (NGEO)

## 1. CONTESTO E OBIETTIVO SCIENTIFICO
Trasformare il repository `exactmpc` in un framework di ricerca modulare, riproducibile e scalabile (target: Automatica / IEEE TAC / NeurIPS) basato su **Exact Quadratic Programming** e **Neural-Guided Exact Optimization (NGEO)**[cite: 50, 51, 62].

Il sistema unifica tre componenti:
1. **Inner Loop Esatto (Constrained MPC Linearizzato / Nominale, $P \succeq 0$):**
- Risoluzione in forma chiusa su sottoinsiemi attivi $J$ con certificato KKT a run-time (fattibilità primale, moltiplicatori $\nu_J \ge 0$, duality gap nullo) e senza tolleranze arbitrarie sull'ottimo[cite: 28, 30, 51].
- Gestione esplicita dell'Hessiana singolare ($P \succeq 0$, es. assi CNC dual-drive sovra-attuati con due motori e una uscita misurata): caratterizzazione della faccia ottima $k(x) = \mathcal{N}(P) \cap \mathcal{N}(G_J)$, selezione deterministica di norma minima, valutazione della capacità a costo zero ($\Delta = 0$) e del prezzo extra oltre capacità[cite: 50, 55].
- Legge del budget sicurezza-libertà: stringere i vincoli riduce i gradi di libertà residui nella faccia ottima e modifica il prezzo $\Delta_J$ e i moltiplicatori $\nu_J$[cite: 50, 55].
2. **Scaling oltre $\kappa = 20$ con Zero-Trust Active-Set Detector:**
- Superare il limite della ricerca esaustiva $2^\kappa$[cite: 50, 60] combinando persistenza temporale ($J_{t-1}$)[cite: 50] e un detector neurale scalabile (da centinaia di migliaia fino a decine di milioni di parametri: MLP/ResNet e modelli sequenziali/Transformer sull'orizzonte) che propone i Top-$K$ candidati $\hat{J}_1, \dots, \hat{J}_K$.
- **Regola Zero-Trust:** Mai fidarsi ciecamente della rete[cite: 62]. Valutare in ordine $J_{t-1}$ e $\hat{J}_1, \dots, \hat{J}_K$ tramite la formula chiusa + certificato KKT; arrestarsi immediatamente al primo candidato valido (0% di subottimalità rispetto all'ottimo del QP), usando un fallback guidato solo se nessun candidato supera il certificato[cite: 51, 55].
3. **Outer Neural Planner & Launcher Adaption Lab:**
- Pianificatore neurale per feedrate e correzione di percorso su un insieme ammissibile $\mathcal{R}_{\text{adm}} \subseteq \mathcal{F}$, addestrato sfruttando la continuità $C^1$ del prezzo $\Delta_J$ e il suo gradiente esatto[cite: 51, 55].
- Modulo di lancio remoto (`launch_adaption.py` / `train_cluster.py`) configurabile tramite `ADAPTION_API_KEY` (e argomenti CLI) per inviare job di scaling e raccogliere metriche e certificati.

## 2. FASI DI LAVORO E REGOLE ZERO-ERROR
- **Fase 0 — Audit del repository:** Ispeziona i file esistenti e i test correnti, elenca cosa è già implementato e cosa manca senza rompere i test esistenti.
- **Fase 1 — Motore esatto in `float64` (`src/exact_core/`):** Implementa solver per $P \succ 0$ e $P \succeq 0$, certificato KKT, calcolo del prezzo $\Delta_J$, moltiplicatori $\nu_J$, ordinamento candidati e layer differenziabile per il training.
- **Fase 2 — Plant e Scaling Benchmark (`src/plants/`, `src/data/`):** Implementa modelli CNC (singolo asse e dual-drive singolare), benchmark ad alta dimensionalità (fino a migliaia di variabili e vincoli) e generazione dati tramite Sobol + Geometric Random Walk con split train/buffer/test disgiunti[cite: 67, 68].
- **Fase 3 — Detector Multi-Milione di Parametri & Planner (`src/models/`, `src/experiments/`):** Implementa modelli a scala crescente, script di benchmark contro solver iterativi e baseline neurali, e il modulo di lancio per Adaption Lab.
- **Vincoli numerici:** La validazione KKT e i certificati devono sempre girare in `float64` con test automatici (`pytest`) prima di chiudere un task. Scrivi i compiti in `.bus/TASK_A_TO_B.md` e i risultati in `.bus/REPORT_B.md`.
