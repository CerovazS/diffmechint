# IDEAS to Develop

## Asse 2 — Cross-model feature alignment & universalità (alta novelty)
Idea: stabilire quanto le rappresentazioni interne di DiT-REPA-E, DiT-SD, DiT-EQ siano "lo stesso modello con basi diverse" o davvero diverse computazioni.

Cosa fare:

Matching delle feature SAE tra i 3 modelli (Hungarian su correlazione delle attivazioni su un dataset di prompt fisso)
Calcolo della frazione di feature universali (matched cross-modello) vs tokenizer-specific
Probing su concept set fissato (oggetti COCO + stili WikiArt + attributi CelebA): quante concept directions sono linearmente decodificabili dalle SAE feature per ciascun tokenizer
Ipotesi: EQ-VAE dovrebbe portare a feature più condivise con SD VAE (stesso encoder family) ma con riallineamento spaziale per la sua equivarianza; REPA-E dovrebbe avere il maggior numero di feature universali ad alto livello semantico (perché ancorato a DINOv2) ma meno feature low-level texture.

Novelty: il filone "feature universality" è molto attivo sugli LLM (cross-model SAE matching, Anthropic/Bloom) ma inesistente sui diffusion. Saresti il primo a portarlo lì.

## Asse 5 — Causal interventions cross-tokenizer (alta novelty se funziona)
Idea: usare le feature SAE matched (dall'Asse 2) per fare patching cross-model — prendere il vettore di attivazione di una feature in DiT-REPA-E e iniettarlo in DiT-SD nello stesso layer.

Cosa misurare:

L'effetto generativo è coerente? Una "feature dog" di REPA-E aumenta la probabilità di cani anche in DiT-SD?
Quanti layer di profondità servono prima che il transfer fallisca? Questo è un proxy diretto della "universalità rappresentazionale"
Novelty: questo tipo di esperimento è frontiera anche sugli LLM. Sui diffusion è essenzialmente terra vergine. È rischioso (potrebbe non funzionare per via dei diversi residual stream basis) ma anche un risultato negativo è informativo.


## Asse 6 — Predittività SAE→diffusability metric (novelty media)
Idea: stabilire un mapping tra metriche geometriche del latente (Fisher rate di D1, three-property score di PAE D2) e metriche SAE-osservabili (Asse 1).

Se la correlazione è alta, ottieni un risultato pulito: "le proprietà geometriche del latente predicono la struttura SAE del denoiser". Se è bassa, ancora più interessante: "il denoiser ricodifica il latente in modi non determinati dalla sua sola geometria".

Novelty: linkbridge formale tra due sotto-letterature che oggi non si parlano.