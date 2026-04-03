# JMR Genetic Game of Neural Network Life

A bio-inspired neural network simulator where neurons are living cells with **15 genes** and **13 proteins** that grow, connect, learn, and die on a 28×28 grid. Networks self-assemble through genetic rules, learn via backpropagation with weights stored *inside* each cell's dendrites, and are sculpted by environmental pruning — combining Conway's Game of Life mechanics with gradient descent.

**100/100 on MNIST. 98/100 on Fashion-MNIST.**

**[▶ Run in your browser — no install](https://jmrothberg.github.io/3D-neuron-game-of-life-simulator/neurosim_web.html)**  
Press **M** → **D** → OK, then **M** (MNIST) or **F** (Fashion-MNIST) to load a bundled 500-sample demo, then **T** to train.

**Legacy Python desktop version:** see [`neurosim_python/README.md`](neurosim_python/README.md).

---

## Why This Is Different

Traditional neural networks are engineered: you choose the layer sizes, connectivity, and hyperparameters, then train a fixed weight matrix. **This simulator inverts that.** Cells are autonomous agents that:

1. **Grow their own connections** — each cell's dendrite size is encoded in its genes, not set by an architect
2. **Are born and die** — new cells emerge when conditions match their birth gene; cells die from overcrowding, isolation, or environmental pruning
3. **Carry heritable genes** — when two cells breed, offspring inherit genes via crossover and mutation
4. **Store weights locally** — weights live in each cell's dendrite array, not in a separate layer matrix
5. **Set their own learning rate** — each cell can evolve its own plasticity speed, gradient sensitivity, and activation curve
6. **Set their own pruning sensitivity** — each cell carries its own thresholds for weight-magnitude and contribution-score pruning (genes 12–13), allowing evolution to favor cells that resist or accept pruning pressure
7. **Are sculpted by the environment** — cells that don't contribute (low charge change, low gradient, near-zero weights, or low contribution score) are pruned, mimicking synaptic pruning during brain development

The result is a network that **self-organizes its own topology** through genetic evolution, then learns through backpropagation, then gets pruned by environmental pressure — a cycle of growth, learning, and selection that mirrors biological neural development.

---

## How This Is Different from a Traditional NN

Traditional NN: weights live in **layer-to-layer matrices** owned by the network. A global optimizer updates all weights at once.

This simulator: weights live **inside each cell's dendrites** — a flat 1D array (`weights[]`) the cell owns and updates independently. There is no global weight matrix. Each cell runs its own gradient descent with its own learning rate (gene 9), weight decay (gene 8), and activation curve (gene 11).

---

## The Cell: Genes and Proteins

Every cell carries two types of information, inspired by molecular biology:

- **Genes (15 values)** — inherited, stable parameters that define the cell's identity, structure, pruning sensitivity, immunity, and **how protein state is created and updated**. These are the cell's *genotype*.
- **Proteins (13 values)** — ALL dynamic cell state is protein state. This is the cell's *phenotype*. Proteins operate at three timescales:
  - **Fast signaling** (3): charge, error, gradient — overwritten each forward/backward step
  - **Long-term memory** (3): weights, bias, backprops_remaining — persist and accumulate across samples; weights and bias ARE the learned knowledge
  - **Integrative memory** (7): rolling statistics derived from charge and gradient trajectories — the cell watches its own protein activity over time and builds compressed history for pruning decisions (like post-translational modifications that mark how active a pathway has been)

### How genes specify proteins (expression, updates, and cutoffs)

In this model, **proteins are not arbitrary numbers** — they are **generated and governed by alleles** the same way real proteins are specified by the genome:

| Idea | Role of genes | What still **changes** at the protein level |
|------|----------------|----------------------------------------------|
| **Shape / count** | Gene **4** (dendrite size) fixes how many **weight** slots exist; gene **6** + **4** set He scaling for **initial weights**; gene **5** sets the **scale for initial bias**. | After init, **each weight and bias value** is free to move with learning. |
| **Dynamics** | Genes **9** (learning rate), **8** (decay), **11** (activation slope) set **how** charge is computed from inputs and **how fast** bias/weights move when gradients arrive. | **Charge, error, gradient** are recomputed every forward/backward step; **weights** and **bias** accumulate experience. |
| **Cutoffs / thresholds** | Genes **7**, **10**, **12**, **13** (or global config when **U** is off) hold **fixed comparison levels**: “enough activity,” “enough gradient,” “enough weight magnitude,” “enough contribution.” Those allele/config values **do not replace** the proteins — they are compared **against** quantities built from **changing** protein state (and from integrative memory derived from charge/gradient). | The **signals being tested** (charge swings, avg \|gradient\|, max \|weight\|, contribution score) **rise and fall** with training; the **bar** often stays on the gene side. |

So: **alleles supply architecture, initial scales, update rules, and threshold levels**; **proteins carry the evolving numbers** that learning and activity produce. Cutoff **values** may be **initialized from** or **stored as** gene (or config) alleles, but the **quantities being cut off** live in the phenotype and **can change** every sample.

### Memory: all memory is protein state

**Genes do not store experience.** ALL experience is carried in **proteins** — a useful mental model is **post-translational / activity-dependent modification**: the genome is fixed, but what happens to the cell **leaves marks in proteins at different timescales**.

**Fast signaling proteins** (overwritten each step — carry "what is happening now"):

| Protein | Role | Init | Runtime range |
|---------|------|------|---------------|
| **Charge** | Activation after forward pass — a snapshot, overwritten on each image. | 0 (hidden/output); random (input) | Hard-clipped [−10, 10]; typically −1 to 1 after leaky ReLU |
| **Error** | Backprop error signal for this step. | ε (1e-15) | Hard-clipped [−10, 10]; typically small |
| **Gradient** | Immediate learning direction for each weight slot. | 0 | Clipped to ±gradient_clip (default ±0.5) |

**Long-term memory proteins** (persist across samples — carry "what was learned"):

| Protein | What is remembered | Init range | Typical trained range |
|---------|-------------------|------------|----------------------|
| **Weights** (`weights[]`) | Learned connection strengths; accumulate across samples. Main **synaptic memory**. | He-scaled: randn × √(2/fan_in), clipped [−1, 1] | −2 to 2 (weight decay pulls toward 0) |
| **Bias** | Learned resting offset; persistent across samples. | uniform(−gene 5, +gene 5); default ±0.01 | −0.5 to 0.5 typically |
| **backprops_remaining** | Immune countdown — decremented each training cycle (forward pass). While > 0, cell is immune to all pruning. | gene 14 (10–100) | Counts down to 0, then stays at 0 |

Together, **weights** and **bias** are where the network's **learned mapping** actually lives. **backprops_remaining** protects newborn cells until they've had enough training cycles to integrate.

**Integrative memory proteins** (rolling statistics — carry "compressed history of recent activity"):

These proteins are built from the **trajectories of charge and gradient** over time. Think of them as activity-dependent modifications: the cell watches its own fast proteins and builds slow summaries. All pruning decisions read from them.

| Protein | What it tracks | How it's updated | Used by |
|---------|---------------|-----------------|---------|
| **max_charge_diff_forward** | Max charge swing across forward-pass samples | Each forward pass: push charge, track running max − min over last epoch-worth of samples | Activity pruning (P key), contribution score |
| **max_charge_diff_reverse** | Max charge swing across reverse-pass samples | Each reverse pass: push charge, track running max − min | Activity pruning (P key), contribution score |
| **avg_gradient_magnitude** | Rolling average of \|gradient\| over recent samples | Each backward pass: push \|gradient\| to history window, compute mean | Gradient pruning (O key), contribution score |
| **contributionScore** | Combined activity + learning signal | `max(charge_diff_fwd, charge_diff_rev) × avg_gradient_magnitude` | Contribution-score pruning (Z key), percentile pruning, 3D color mode |
| **significant_charge_change_forward** | Sticky flag: has forward charge ever exceeded gene 7 | Set `true` when `max_charge_diff_forward > threshold`, never cleared | Conway death protection only |
| **significant_charge_change_reverse** | Sticky flag: has reverse charge ever exceeded gene 7 | Set `true` when `max_charge_diff_reverse > threshold`, never cleared | Conway death protection only |
| **significant_gradient_change** | Sticky flag: has gradient ever exceeded gene 10 | Set `true` when `avg_gradient_magnitude > threshold`, never cleared | Conway death protection only |

**Summary:** Fast proteins carry what is happening now. Long-term proteins carry what was learned. Integrative proteins carry compressed history for pruning. ALL memory lives in proteins — there is no separate "memory field" concept.
---

### The 14 Genes

**Genes are fixed after a cell is created.** They do not change during training or backprop.

**Two exceptions (real biology has these too):**

- **Germline mutation:** When a new cell is born from two parents, gene 3 can trigger re-randomization of breeding or network genes (probability = **MR / 1000** per birth).
- **Somatic mutation:** While a cell is alive, the same gene 3 can very rarely trigger a full gene re-init (probability = **MR / 100,000** per evolution step). This is 100× rarer than germline, matching biology where somatic mutations are uncommon events.

**Gene 3 (mutation rate)** therefore controls **both** offspring mutation (germline) and rare in-life mutation (somatic), at different rates.

Genes 0–2 control **survival and reproduction** (Game of Life rules).
Genes 3–8 control **network structure and regularization**.
Genes 9–11 control **learning dynamics**.
Genes 12–13 control **pruning sensitivity**.

| Gene | Name | Sym | Allele range (autonomous) | Non-auto default | What the allele controls at runtime | Bio analogy |
|------|------|-----|---------------------------|------------------|-------------------------------------|-------------|
| **0** | Overcrowding | OT | integer 2–15 | 2–15 (random) | Cell dies if alive-neighbor count ≥ OT | Contact-inhibition apoptosis |
| **1** | Isolation | IT | integer 2–15 (≤ gene 0) | 2–15 (≤ gene 0) | Cell dies if alive-neighbor count ≤ IT | Trophic-factor starvation |
| **2** | Birth | BT | integer 2–15 | 2–15 (random) | Empty site gets a child when neighbor count = BT | Morphogen threshold for mitosis |
| **3** | Mutation Rate | MR | integer 0–99 | config (10) | Germline: MR/1000 per birth; somatic: MR/100 000 per step | DNA-repair fidelity |
| **4** | Dendrite Size | WG | 9, 25, or 49 | config (9) | Creates `weights[WG]` → sets the **number of synaptic weight proteins** | Dendritic arbor size |
| **5** | Bias Range | BR | 0.001 or 0.01 | config (0.01) | Bias protein initialized to uniform(−BR, +BR) | Resting membrane potential |
| **6** | Fan-In | AW | upstream cell count | config (5) | He scale = √(2/AW) → sets **initial weight magnitude** | Synaptic normalization |
| **7** | Charge Delta | CD | 10^uniform(−6,−2) → 1e-6 … 0.01 | config (0.01) | Pruning: cell survives only if max charge swing > CD. Also sets "significant charge" sticky flag. | Activity-dependent survival |
| **8** | Weight Decay | WD | 10^uniform(−6,−4) → 1e-6 … 1e-4 | config (1e-5) | Each update: `weight -= LR×grad + WD×weight` → **weight proteins shrink** toward 0 | Synaptic protein turnover |
| **9** | Learning Rate | LR | uniform(0.003, 0.05) | config (0.01) | Step size: `weight -= LR×grad` → controls **how fast weight and bias proteins change** | Hippocampal vs cortical plasticity |
| **10** | Gradient Thresh | GT | 10^uniform(−8,−4) → 1e-8 … 1e-4 | config (1e-4) | Pruning: cell dies if avg \|gradient protein\| ≤ GT | Neurotrophic receptor density |
| **11** | Activation Slope | AS | uniform(0.01, 0.3) | config (0.01) | Leaky ReLU: if pre-activation ≤ 0, charge = AS × pre-activation → **shapes the charge protein** | Ion-channel selectivity |
| **12** | Weight Prune Thresh | WPT | 10^uniform(−3,−1.5) → 0.001 … 0.032 | config (0.01) | Pruning: cell dies if max \|weight protein\| < WPT. Range capped below He-init scale so cells aren't born dead. | Synaptic maintenance threshold |
| **13** | Min Contribution | MCS | 10^uniform(−6,−2) → 1e-6 … 0.01 | config (0, off) | Pruning: cell dies if contributionScore < MCS (score = charge_diff × gradient) | Activity-dependent trophic need |
| **14** | Immune Period | IP | integer 10–100 | config (50) | Newborn cell is **protected from all pruning** for IP training cycles (forward passes). Protein `backprops_remaining` counts down each forward pass. | Neonatal immune period |

**How to read this table:** The "allele range" column shows the space of possible values a gene can take in autonomous mode. The "what the allele controls" column shows exactly which **protein or decision** the allele value feeds into at runtime — this is the gene→protein link.

#### The Cell Chromosome

Every cell carries a single chromosome of 15 genes, organized in five functional regions:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                CELL CHROMOSOME  (15 genes)                                        │
├───────────────┬──────────────────────────┬───────────────────┬──────────────────────┬─────────────┤
│  BREEDING     │  NETWORK STRUCTURE       │  LEARNING         │  PRUNING SENSITIVITY │  IMMUNITY   │
│  (survival)   │  (anatomy & wiring)      │  (plasticity)     │  (trophic thresholds)│  (neonatal) │
├───┬───┬───┬───┼───┬───┬───┬───┬───┬──────┼───┬───┬──────────┼───┬──────────────────┼─────────────┤
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │      │ 9 │10 │ 11       │12 │ 13               │ 14          │
│OT │IT │BT │MR │WG │BR │AW │CD │WD │      │LR │GT │ AS       │WPT│ MCS              │ IP          │
├───┴───┴───┴───┼───┴───┴───┴───┴───┴──────┼───┴───┴──────────┼───┴──────────────────┼─────────────┤
│ When does the │ How big are my dendrites │ How fast do I     │ How hard must I work │ How long am │
│ cell live,    │ and how are weights      │ learn, and what   │ to stay alive?       │ I protected │
│ die, or       │ seeded? How fast do      │ is my response    │ Thresholds compared  │ from pruning│
│ reproduce?    │ weights decay?           │ curve shape       │ against my proteins  │ after birth?│
│               │                          │ (slope=gene 11).  │ and proteins.        │             │
└───────────────┴──────────────────────────┴───────────────────┴──────────────────────┴─────────────┘
     Always           Non-autonomous:             Non-autonomous:       Non-autonomous:       Non-auto:
     per-cell         from global config          from global config    from global config    from config
                      Autonomous (U on):          Autonomous (U on):    Autonomous (U on):    Autonomous:
                      evolved per-cell             evolved per-cell      evolved per-cell     evolved
```

**Reading the chromosome:** Gene 0 (leftmost) through gene 14 (rightmost). Breeding genes are always per-cell. Genes 3–14 can be globally configured (U off) or independently evolved (U on). In autonomous mode, offspring inherit genes from two parents via crossover, with germline mutation controlled by gene 3.

#### Gene → Protein Map

Every gene either **creates** a protein, **controls** how a protein changes, sets a **threshold** compared against a protein, or is a **regulatory element** that makes no protein at all:

| Gene | Relationship | Protein(s) affected |
|------|-------------|---------------------|
| **0** OT | **Regulatory** — no protein | Compares neighbor count against allele → death decision |
| **1** IT | **Regulatory** — no protein | Compares neighbor count against allele → death decision |
| **2** BT | **Regulatory** — no protein | Compares neighbor count against allele → birth decision |
| **3** MR | **Regulatory** — no protein | Controls mutation probability (like DNA-repair fidelity) |
| **4** WG | **Creates** | `weights[]` — determines array size (number of synapses) |
| **5** BR | **Creates** | `bias` — sets initial magnitude range |
| **6** AW | **Controls** | `weights[]` — He initialization scaling |
| **7** CD | **Threshold** | Compared against `max_charge_diff_*` integrative proteins |
| **8** WD | **Controls** | `weights[]` — decay rate each update |
| **9** LR | **Controls** | `weights[]` + `bias` — learning step size |
| **10** GT | **Threshold** | Compared against `avg_gradient_magnitude` integrative protein |
| **11** AS | **Controls** | `charge` — leaky ReLU negative slope |
| **12** WPT | **Threshold** | Compared against max \|`weights`\| long-term protein |
| **13** MCS | **Threshold** | Compared against `contributionScore` integrative protein |
| **14** IP | **Creates** | `backprops_remaining` — sets initial countdown value |

**Why 15 genes but only 13 proteins:** Genes 0–3 are regulatory elements — they control discrete events (death, birth, mutation) based on environmental signals, not protein products. Meanwhile, genes 4, 6, 8, and 9 all share the `weights` protein (create it, scale it, decay it, update it). In biology, multiple genes routinely govern the same protein through different regulatory pathways.

#### Gene Groups Explained

**Genes 0–2 (Breeding):** Always per-cell, even in non-autonomous mode. They govern the Conway's Game of Life dynamics — when cells are born, when they die from overcrowding or isolation.

**Genes 3–8 (Network Structure):** Define the physical architecture of each cell — how many dendrites it has, how weights are initialized and decay, and the threshold for "significant" charge activity.

**Genes 9–11 (Learning Dynamics):** Control how each cell learns. Gene 9 (Learning Rate) is the single most impactful gene — in autonomous mode, evolution can discover that deep-layer cells should learn slowly while output cells learn fast. **Gene 11 (Activation Slope)** is the cell's response curve — it sets the leaky ReLU negative slope, controlling how much negative signal passes through. Low slope (0.01) = highly selective, suppresses negative signals; high slope (0.3) = permissive, passes more signal through. It is a gene (fixed at birth), not a protein, because it defines what *type* of neuron this is — analogous to which ion channels the cell expresses during development.

**Genes 12–13 (Pruning Sensitivity):** These genes put pruning thresholds *inside* the cell. In non-autonomous mode, all cells share the same global values from config. In autonomous mode, each cell evolves its own sensitivity — cells can become more or less resilient to pruning pressure through natural selection. This is analogous to cells expressing different levels of trophic factor receptors: a cell with a low MCS (gene 13) is "easy to satisfy" and survives with minimal contribution, while a cell with a high MCS is under stronger pressure to contribute or die.

**Gene 14 (Immune Period):** Every newborn cell gets an **immune period** — a number of training cycles (forward passes) during which it cannot be killed by any pruning strategy (P, O, Y, Z, or percentile). The gene sets the duration (10–100 training cycles), and the protein `backprops_remaining` counts down from that value. Once `backprops_remaining` reaches 0, the cell becomes eligible for pruning like any other. In autonomous mode, cells with longer immune periods have more time to learn and integrate into the network but also occupy resources longer if they turn out to be useless. Evolution balances this tradeoff. This is analogous to the neonatal period in biology where immature neurons are protected from apoptosis while they migrate and form synapses.

### Gene 4 (Dendrite Size) Detail

Gene 4 determines `reach = (√genes[4] − 1) / 2`:

| Gene 4 Value | Weight Matrix | Reach | Receptive Field |
|-------------|---------------|-------|-----------------|
| 9 | 3×3 | 1 | 1 cell in each direction |
| 25 | 5×5 | 2 | 2 cells in each direction |
| 49 | 7×7 | 3 | 3 cells in each direction |
| 81 | 9×9 | 4 | 4 cells in each direction |

### Why Log-Uniform Distributions Matter

Genes 7, 8, 10, 12, and 13 span multiple orders of magnitude. If you use a uniform random distribution over (say) 1e-8 to 1e-4, 99.99% of values land near the top. **Log-uniform sampling** (`10^uniform(a, b)`) ensures equal probability across each order of magnitude, producing genuine diversity in survival sensitivity, decay rate, and activity thresholds — exactly what evolution needs.

---

### The 13 Proteins

These are ALL the cell's dynamic state — its *expressed behavior*. Organized by timescale:

#### Fast Signaling Proteins (3) — overwritten each forward/backward step

| Protein | What It Is | How It Changes | Range | Biological Analogy |
|---------|-----------|----------------|-------|--------------------|
| **Charge** | Activation signal | Forward: `leaky_ReLU(bias + Σ(upstream_charge × weight))` | [−10, 10] | Membrane potential / firing rate |
| **Error** | Backprop error signal | Backward: accumulated from downstream errors × reversed weights | [−10, 10] | Retrograde signaling molecules |
| **Gradient** | Most recent learning signal | `error × upstream_charge`, clipped to [−clip, +clip] | [−clip, clip] | Calcium/CaMKII activity level |

#### Long-term Memory Proteins (3) — persist and accumulate across samples

| Protein | What It Is | How It Changes | Range | Biological Analogy |
|---------|-----------|----------------|-------|--------------------|
| **Weights** | Synaptic connection strengths (1D array, size = gene 4) | He-init: `randn × √(2/fan_in)`. Update: `w -= lr × gradient + decay × w` | Unconstrained | Synaptic receptor density |
| **Bias** | Baseline offset before activation | `bias -= lr × error` each backward step | Initialized near 0 | Resting membrane potential |
| **backprops_remaining** | Immune countdown (gene 14) | Init from gene 14; decrements each training cycle (forward pass). While > 0, cell is immune to all pruning. | 0 to gene 14 value | Neonatal immune period |

Weights and bias ARE the learned knowledge — the network's memory of what it has been taught.

#### Integrative Memory Proteins (7) — rolling statistics derived from fast protein trajectories

These are NOT separate from proteins. They ARE proteins — built by the cell watching its own charge and gradient over time. Think of them as post-translational modifications that accumulate with activity. All pruning decisions read from them.

| Protein | What It Tracks | How It's Updated | Used By |
|---------|---------------|-----------------|---------|
| **max_charge_diff_forward** | Max charge swing across forward-pass samples | Each forward pass: push charge, track running max − min over last epoch-worth of samples | Activity pruning (P key), contribution score |
| **max_charge_diff_reverse** | Max charge swing across reverse-pass samples | Each reverse pass: push charge, track running max − min | Activity pruning (P key), contribution score |
| **avg_gradient_magnitude** | Rolling average of \|gradient\| over recent samples | Each backward pass: push \|gradient\| to history window (size = training data count), compute mean | Gradient pruning (O key), contribution score |
| **contributionScore** | Combined activity + learning signal | `max(max_charge_diff_fwd, max_charge_diff_rev) × avg_gradient_magnitude` | Contribution-score pruning (Z key), percentile pruning, 3D color mode |
| **significant_charge_change_forward** | Sticky flag: has forward charge ever exceeded gene 7 | Set to `true` when `max_charge_diff_forward > threshold`, never cleared (until explicit reset) | Conway death protection only (shouldDieGenetic) |
| **significant_charge_change_reverse** | Sticky flag: has reverse charge ever exceeded gene 7 | Set to `true` when `max_charge_diff_reverse > threshold`, never cleared | Conway death protection only (shouldDieGenetic) |
| **significant_gradient_change** | Sticky flag: has gradient ever exceeded gene 10 | Set to `true` when `avg_gradient_magnitude > threshold`, never cleared | Conway death protection only (shouldDieGenetic) |

**Key design principle:** Rolling metrics (`max_charge_diff_*`, `avg_gradient_magnitude`, `contributionScore`) are *live* — they reflect recent training and are used for pruning. Sticky flags (`significant_*`) are only for Conway-style genetic death protection: once a cell has ever been significantly active, it's protected from overcrowding/isolation rules.

**Pruning uses rolling windows, not single-sample snapshots.** The rolling window for `max_charge_diff_*` and `avg_gradient_magnitude` is sized to the training set (1 epoch worth of samples). A cell must be consistently inactive or non-learning across many images before it can be pruned.

---

### How Genes and Proteins Interact

Genes and proteins create a two-timescale system:

| Timescale | What Changes | Mechanism |
|-----------|-------------|-----------|
| **Per-sample** (fast) | Proteins: charge, error, weights, bias, gradient | Forward/backward pass, gradient descent |
| **Per-sample** (fast) | Integrative memory proteins: rolling charge diffs, gradient history, contribution score | Updated inside `updateCharge()` and `updateGradientImportance()` |
| **Per-generation** (slow) | Genes 0–13 | Crossover, mutation, natural selection |

- Gene 4 determines *how many* weights a cell has → Proteins (weights) fill that array and are trained
- Gene 7 determines the *threshold* for significant activity → Integrative protein (charge diffs) is measured against it
- Gene 8 determines *how fast* weights decay → Protein (weights) shrink by that factor each update
- Gene 9 determines *how fast* the cell learns → Protein (weights) update at that rate
- Gene 10 determines the *gradient survival threshold* → Integrative protein (avg_gradient_magnitude) is compared against it
- Gene 11 determines the *response curve* → Protein (charge) passes through that activation function
- Gene 12 determines *weight-magnitude pruning sensitivity* → Protein (max \|weight\|) is compared against it
- Gene 13 determines *contribution-score pruning sensitivity* → Integrative protein (contributionScore) is compared against it
- Genes 0–2 determine *who lives and dies* → The population of cells is shaped by these rules

---

## Cell Autonomy

The `U` key toggles `autonomous_network_genes`:

- **Off (default):** All cells share the same network gene values from global config. Pruning decisions for charge delta, gradient threshold, weight prune threshold, and min contribution score read from **config** at decision time — so changing values via **E** or **C** takes effect immediately on all cells without rewriting each cell's gene snapshot. This is like training a traditional network — uniform architecture and hyperparameters.
- **On:** Each cell has its own random gene values, subject to evolution. Pruning reads from the per-cell gene values. This is the bio-inspired mode — cells evolve independently, producing a heterogeneous network.

| Gene | Autonomous Off (U off) | Autonomous On (U on) |
|------|------------------------|----------------------|
| 0–2 (breeding) | Always per-cell (random at birth) | Always per-cell (random at birth) |
| 3 (mutation rate) | From config; same for all cells | Random per cell: 0–99 |
| 4 (dendrite size) | From config; same for all cells | Random: 9, 25, or 49 weights |
| 5 (bias range) | From config; same for all cells | Random: 0.001 or 0.01 |
| 6 (fan-in) | From config; same for all cells | Measured per cell from actual connections |
| 7 (charge delta) | **Config value at runtime** (E/C changes apply immediately to all cells) | Per-cell gene: 10^uniform(−6,−2) |
| 8 (weight decay) | **Config value at runtime** (E changes apply immediately) | Per-cell gene: 10^uniform(−6,−4) |
| 9 (learning rate) | **Config value at runtime** (E/I changes apply immediately) | Per-cell gene: uniform(0.003, 0.05) |
| 10 (gradient threshold) | **Config value at runtime** (E/C changes apply immediately) | Per-cell gene: 10^uniform(−8,−4) |
| 11 (activation slope) | **Config value at runtime** (E changes apply immediately) | Per-cell gene: uniform(0.01, 0.3) |
| 12 (weight prune threshold) | **Config value at runtime** (C changes apply immediately) | Per-cell gene: 10^uniform(−3,−1.5) → 0.001–0.032 |
| 13 (min contribution score) | **Config value at runtime** (C changes apply immediately) | Per-cell gene: 10^uniform(−6,−2) |
| 14 (immune period) | **Config value at runtime** (E changes apply immediately) | Per-cell gene: integer 10–100 |

**Key difference:** With U off, changing a parameter via **E**, **I**, or **C** takes effect **immediately** on every cell — the config value is read fresh each forward/backward pass and each pruning check. With U on, each cell uses its own gene allele; the only way to change it is through **evolution** (birth with new genes) or **X** (re-init from config).

---

## Conway's Game of Life vs This Version

**Original Conway's rules (fixed for all cells):**
- Overcrowding (OT=3): >3 neighbors → die.
- Isolation (IT=2): <2 neighbors → die.
- Birth (BT=3): exactly 3 neighbors → new cell.
- No genes, no mutation, no evolution. Rules are static.

**This version makes OT, IT, BT into per-cell GENES that are inherited:**
- Gene 0 (OT): neighbors ≤ OT to survive (dies if more)
- Gene 1 (IT): neighbors ≥ IT to survive (dies if fewer)
- Gene 2 (BT): neighbors == BT for birth
- Survival band: IT ≤ neighbors ≤ OT (OT > IT enforced)

Offspring inherit genes via crossover from two parents + mutation. This creates evolving populations with diverse survival strategies.

---

## Pruning: Five Complementary Strategies

Pruning removes cells that don't contribute, mimicking synaptic pruning during brain development. The simulator implements five strategies that can be combined:

### Strategy 1: Charge-Delta Pruning — key P

Cells whose charge doesn't change significantly across training samples are killed. Uses integrative proteins (`max_charge_diff_forward`, `max_charge_diff_reverse`) compared against gene 7 (Charge Delta) or config value.

- **AND logic (`=` key):** Cell must show significant change in *both* forward and reverse passes to survive. Strict — requires bidirectional contribution.
- **OR logic (`=` key):** Cell survives if it shows significant change in *either* direction. More lenient.

### Strategy 2: Gradient Pruning — key O

Cells with average gradient magnitude below their survival threshold are killed. Uses integrative protein (`avg_gradient_magnitude`) compared against gene 10 (Gradient Threshold) or config value.

### Strategy 3: Weight-Magnitude Pruning — key Y

Cells whose maximum absolute weight falls below their weight prune threshold are killed. Uses protein data (max \|weight\|) compared against gene 12 (Weight Prune Threshold) or config value. Biologically: a synapse that has decayed to near zero carries no signal.

### Strategy 4: Contribution-Score Pruning — key Z

Cells whose contribution score (`max(charge_diff_fwd, charge_diff_rev) × avg_gradient_magnitude`) falls below their minimum contribution score are killed. Uses integrative protein (`contributionScore`) compared against gene 13 (Min Contribution Score) or config value. Biologically: combines "is this cell active?" with "is it learning?" into a single survival test.

### Strategy 5: Percentile Pruning (automatic at epoch boundary)

At the end of each epoch, the bottom N% of cells (ranked by contribution score) are killed. Configured via `prune_percentile` (0 = off, set via **C** key). Unlike strategies 1–4 which use per-cell thresholds, this is a relative/competitive mechanism: cells must outperform their peers to survive.

### Pruning Summary Table

| Strategy | Key | What's Compared | Threshold Source | When Checked | Biological Analogy |
|----------|-----|----------------|-----------------|--------------|-------------------|
| Charge-delta | **P** | `max_charge_diff_*` | Gene 7 / config | Every evolution step | Activity-dependent survival |
| Gradient | **O** | `avg_gradient_magnitude` | Gene 10 / config | Every evolution step | Neurotrophic factor requirement |
| Weight-magnitude | **Y** | `max(\|weight\|)` | Gene 12 / config | Every evolution step | Synaptic maintenance threshold |
| Contribution score | **Z** | `contributionScore` | Gene 13 / config | Every evolution step | Combined activity + learning |
| Percentile | auto | Rank of `contributionScore` | Config `prune_percentile` | Once per epoch boundary | Competitive survival pressure |

**Note:** "Every evolution step" means pruning conditions are checked continuously. However, the metrics being checked (`max_charge_diff`, `avg_gradient_magnitude`) are rolling averages over an epoch-worth of samples, so a cell must be consistently inactive over many samples to fail.

---

## The Life Cycle of a Network

### Phase 1: Growth (Andromida Mode)

Starting from a sparse grid, cells reproduce according to their birth genes. A cell is born at an empty location if its parent-derived gene 2 matches the local neighbor count. Offspring inherit all 14 genes from two parents via crossover, with germline mutation controlled by gene 3 (probability = MR/1000).

### Phase 2: Learning (Training Mode)

Input layer cells are loaded with MNIST pixel data. Charge propagates forward through dendritic weights. Error propagates backward through reversed weight indices. Each cell updates its own weights and bias using its own learning rate (gene 9).

### Phase 3: Pruning (Environmental Selection)

Cells that don't contribute to the network are removed through five mechanisms (see Pruning section above). In autonomous mode, each cell's tolerance for pruning pressure is encoded in its genes (7, 10, 12, 13) — evolution discovers which sensitivity levels lead to useful networks.

### Phase 4: Regrowth

After pruning, evolution can restart. New cells fill gaps, potentially with mutated genes that produce different dendrite sizes, learning rates, activation slopes, or pruning sensitivities. Somatic mutation (gene 3, probability = MR/100,000) can also rarely re-randomize an existing cell's genes. The cycle repeats: **grow → learn → prune → regrow**.

---

## The Grid

```
         Layer 0       Layer 1       Layer 2      ...   Layer N-2     Layer N-1
        (Input)       (Hidden)      (Hidden)           (Output)      (Desired)
       ┌────────┐    ┌────────┐    ┌────────┐         ┌────────┐    ┌────────┐
       │ 28×28  │    │ 28×28  │    │ 28×28  │         │ 28×28  │    │ 28×28  │
       │ pixels │───>│ cells  │───>│ cells  │───>...─>│ cells  │    │ labels │
       │(MNIST) │    │        │    │        │         │        │    │        │
       └────────┘    └────────┘    └────────┘         └────────┘    └────────┘
                     ◄──── charge flows forward (F key) ────►
                     ◄──── error flows backward (B key) ────►
```

Layer 0 is loaded with MNIST pixel data. Layer N-1 holds desired output labels.
Layers 1 through N-2 are the hidden layers where cells live, learn, and die.

---

## How the Forward Pass Works (one cell's perspective)

Each cell looks at the layer **above** it (closer to input), gathers charges from
nearby cells within its dendrite reach, and computes its own charge.

```
    LAYER ABOVE (layer - 1)                    THIS CELL (layer)
    ┌─────────────────────────┐
    │  .  .  .  .  .  .  .   │
    │  .  .  .  .  .  .  .   │
    │  .  . [A] [B] [C] .  . │                ┌───────────┐
    │  .  . [D] [E] [F] .  . │ ──charges──>   │  Cell X    │
    │  .  . [G] [H] [I] .  . │                │  at (x,y)  │
    │  .  .  .  .  .  .  .   │                └───────────┘
    │  .  .  .  .  .  .  .   │
    └─────────────────────────┘
          3×3 dendrite reach
          (gene 4 = 9 weights)
```

**Step 1: Gather upstream cells.** Cell X checks every position within its `reach`
(determined by gene 4) in the layer above. With gene 4 = 9, reach = 1, so it
checks a 3×3 area centered on its own (x,y) position. It collects the charges
from cells A through I (skipping any empty positions).

**Step 2: Compute weighted sum + bias.**

```
    charge = bias + (A.charge × w[0]) + (B.charge × w[1]) + (C.charge × w[2])
                  + (D.charge × w[3]) + (E.charge × w[4]) + (F.charge × w[5])
                  + (G.charge × w[6]) + (H.charge × w[7]) + (I.charge × w[8])
```

The bias is the cell's resting potential — a baseline signal present even with
no input. It is learned via gradient descent, just like the weights.

**Step 3: Apply leaky ReLU activation** (gene 11 = **activation_slope**; set via **E** or per-cell when **U** is on — see **Key E — what you will enter**).

```
    if charge > 0:  charge = charge           (pass through)
    if charge ≤ 0:  charge = slope × charge   (slope = activation_slope)
```

Low slope (0.01) = selective neuron, suppresses negative signals.
High slope (0.3) = permissive neuron, passes more signal through.

**Step 4: Clip to [-10, 10] and store as the cell's new charge.**

**Step 5: Update integrative memory proteins.** The new charge is pushed to the cell's charge history array. Rolling `max_charge_diff` is recomputed (max − min of recent charges). `contributionScore` is recomputed as `max(charge_diff_fwd, charge_diff_rev) × avg_gradient_magnitude`.

---

### Weight Indexing

Weights are stored in a flat 1D array. The 2D offset (dx, dy) from the cell
to each upstream neighbor maps to an array index:

```
    weight_index = (dx + reach) × matrix_width + (dy + reach)

    For a 3×3 dendrite (reach=1, matrix_width=3):

    Upstream position:    (-1,-1) (-1, 0) (-1,+1)      Weight index:  0  1  2
                          ( 0,-1) ( 0, 0) ( 0,+1)  →                  3  4  5
                          (+1,-1) (+1, 0) (+1,+1)                      6  7  8
```

For a 5×5 dendrite (gene 4 = 25, reach = 2), the same formula produces
indices 0–24 over a 5×5 grid of upstream positions.

---

## How the Backward Pass Works (one cell's perspective)

Backpropagation has two jobs: (1) compute this cell's error signal, and
(2) update this cell's weights. It works from the output layer back toward
the input layer.

### Job 1: Compute Error Signal

**Output layer cells** (layer N-2): error = how wrong we are.

```
    error = (my_charge − desired_output) × leaky_ReLU_derivative(my_charge)
```

**Hidden layer cells**: error = share of blame from cells below.

```
    THIS CELL (layer)                          LAYER BELOW (layer + 1)
    ┌───────────┐                              ┌─────────────────────────┐
    │  Cell X    │                              │  .  .  .  .  .  .  .   │
    │  error = ? │ <──errors + weights────      │  .  . [1] [2] [3] .  . │
    └───────────┘                               │  .  . [4] [5] [6] .  . │
                                                │  .  . [7] [8] [9] .  . │
                                                │  .  .  .  .  .  .  .   │
                                                └─────────────────────────┘
                                                 These cells BELOW have
                                                 already computed their
                                                 errors (we work backward)
```

Cell X accumulates error from each cell below that connects to it:

```
    error = Σ (cell_below.error × cell_below.weights[reversed_index])
            × leaky_ReLU_derivative(my_charge)
```

### Why `reversed_index`?

In the forward pass, Cell 5 (below) uses Cell X's charge via `weight_index` to compute Cell 5's charge. In the backward pass, we need the reverse: Cell X needs Cell 5's error via the **same connection**, but looked up from Cell 5's weight array in the opposite direction:

```
    reversed_index = len(cell_below.weights) − 1 − weight_index

    Forward (Cell 5 looks UP at Cell X):         Backward (Cell X looks DOWN at Cell 5):
    weight_index = (dx + reach) × m + (dy + reach)
                                                  reversed_index = len(w) - 1 - weight_index

    For 3×3:  forward index 0 ↔ reversed index 8     (-1,-1) ↔ (+1,+1)
              forward index 1 ↔ reversed index 7     (-1, 0) ↔ (+1, 0)
              forward index 4 ↔ reversed index 4     ( 0, 0) ↔ ( 0, 0)  (center stays)
              forward index 8 ↔ reversed index 0     (+1,+1) ↔ (-1,-1)

    This maps (dx, dy) → (−dx, −dy), which is equivalent to transposing
    the weight matrix — exactly what standard backprop does.
```

### Job 2: Update Weights

Once Cell X knows its error, it updates each of its own dendritic weights
using the upstream cell charges (same cells used in the forward pass):

```
    For each upstream cell in the layer above:
        gradient = my_error × upstream_cell.charge
        gradient = clip(gradient, −clip_range, +clip_range)

        weight[index] −= learning_rate × gradient + weight_decay × weight[index]
                         ╰──────── learn ────────╯   ╰──── regularize ────────╯

    bias −= learning_rate × my_error
```

In autonomous mode, `learning_rate` comes from gene 9 and `weight_decay`
from gene 8 — each cell runs its own gradient descent at its own speed.

### Job 3: Update Integrative Memory Proteins

After weight updates, `updateGradientImportance()` pushes |gradient| to the rolling history, recomputes `avg_gradient_magnitude`, and recomputes `contributionScore`.

---

### Complete Forward + Backward Example

```
    Layer 0 (Input)    Layer 1 (Hidden)    Layer 2 (Output)    Layer 3 (Desired)
    ┌──────────┐       ┌──────────┐        ┌──────────┐        ┌──────────┐
    │          │       │          │        │          │        │          │
    │  MNIST   │──F──> │  Cells   │──F───> │  Cells   │        │  Labels  │
    │  pixels  │       │  learn   │        │  predict │        │  (0-9)   │
    │          │       │          │ <─B─── │          │ <─B─── │          │
    └──────────┘       └──────────┘        └──────────┘        └──────────┘

    F = forward pass: charge flows left to right, layer by layer
    B = backward pass: error flows right to left, layer by layer

    1. Load MNIST image into Layer 0 (pixel values become cell charges)
    2. Load one-hot label into Layer 3 (desired output)
    3. FORWARD: Layer 1 cells compute charges from Layer 0
                Layer 2 cells compute charges from Layer 1
    4. BACKWARD: Layer 2 error = (Layer 2 charge − Layer 3 desired)
                 Layer 2 updates its weights using Layer 1 charges
                 Layer 1 error = accumulated from Layer 2 errors
                 Layer 1 updates its weights using Layer 0 charges
    5. Integrative memory proteins updated: charge diffs, gradient history, contribution scores
    6. Repeat for next training sample
```

---

## 3D Visualization Color Modes

Press **G** in 3D view to cycle through five color modes:

| Mode | What It Shows | Color Mapping |
|------|--------------|---------------|
| **Charge** | Cell activation level | Dark (low) → Bright (high charge) |
| **Error** | Backprop error signal | Blue (negative) → Dark (zero) → Red (positive) |
| **Gradient** | Learning signal strength | Dark (low \|gradient\|) → Green (high \|gradient\|) |
| **Weight Strength** | Average absolute weight | Dark (near-zero weights) → Yellow (strong weights) |
| **Contribution** | Combined activity × learning score | Dark (low) → Cyan (medium) → White (high contribution) |

Key **4** activates the backprop learning view: switches to 3D, sets color mode to **Error** so you see the full network update in real time as each training sample is processed. Press **G** to cycle to other modes while in this view.

---

## V Key: Statistics Views

Press **V** to cycle through three statistics screens:

### Screen 0: Current Settings + Pruning Readiness
Shows all current config values alongside network-measured suggestions (`← net: X`). Includes:
- Key learning parameters (LR, bias range, weight decay, etc.)
- Pruning thresholds with `[gene 12]` and `[gene 13]` labels
- Network measurements (cells, avg fan-in, median gradient/weight/charge/error)
- **Pruning Readiness:** "Would die" counts for each pruning strategy at current settings
- Contribution score percentile distribution (p10, p25, p50, p75, p90, max)

### Screen 1: Per-Layer Averages
Detailed per-layer statistics: average charge, error, gradient, weight, fan-in, silent neuron count, weight utilization.

### Screen 2: Cell Types
Cell type census and per-digit accuracy breakdown.

All screens include footnotes explaining when metrics are measured (snapshot vs rolling average) and how pruning differs from display metrics.

---

## Training Features

| Feature | Details |
|---------|---------|
| **Full neural-network training** | Forward pass, backpropagation, weight/bias updates, cross-entropy loss |
| **15 heritable genes** | All 15 genes (breeding, network, learning, pruning, immunity) with autonomous/non-autonomous modes |
| **5 pruning strategies + immunity** | Charge-delta (**P**), gradient (**O**), weight-magnitude (**Y**), contribution-score (**Z**), and percentile (auto) — thresholds via **C** key. Gene 14 gives newborn cells a grace period. |
| **Epoch-based training loop** | One epoch = one full pass over all loaded samples; epochs are counted and displayed in the status bar |
| **Minibatch gradient accumulation** | Configurable batch size **K** (key **K**); gradients accumulate over K samples before one weight update |
| **Epoch shuffling** | Training samples are Fisher-Yates shuffled at the start of each epoch |
| **Dual loss plots** | Separate scrolling plots for per-minibatch loss (blue) and per-epoch loss (green), each with an independent Y-axis scale slider (0.1–20) |
| **He initialization** | Weight reset (**X** key) measures average effective fan-in and scales as `randn × √(2/fan_in)` |
| **Smart defaults with suggestions** | `suggestParams()` scans the network and suggests LR, bias range, weight decay, pruning thresholds. Shown in **E**, **I**, **C** dialogs and **V** screen |
| **Save / Load** | **S** exports full network state as JSON; **L** imports it; auto-downloads a snapshot when accuracy hits 100% |
| **Save filename convention** | `saved_N_980_1000_L8W9_250402_045.json` — N/F = dataset (N = number-MNIST, F = Fashion), maxCorrect\_epochSize, L = layers, W = weights, YYMMDD, 3-digit milliseconds |

---

## Typical Workflow

1. Open `neurosim_web.html` (browser) or the GitHub Pages link
2. Load a saved network (**L**) or draw cells manually (click/drag on the grid)
3. Load training data (**M**) — choose **D** for demo, **J** for your own MNIST JSON
4. Set forward direction (**F**), enable backprop (**B**), start training (**T**)
5. Check **V** screen for pruning readiness before enabling pruning
6. Toggle pruning (**P** charge, **O** gradient, **Y** weight-magnitude, **Z** contribution) to remove dead cells
7. Use **C** to tune pruning thresholds (suggestions shown from network analysis)
8. Disable display (**D**) for faster training — plots and status bar keep updating
9. Watch accuracy climb — save good networks (**S**)

---

## Key **E** — what you will enter (prompt order)

Press **E** (2D view only). A series of modal dialogs appears **in this order**. Type a number (or keep the default) and click **OK** for each. Lines marked **← net** show an optional suggestion from the live network; you can copy that value or ignore it.

| Step | Prompt label | What it means |
|------|----------------|---------------|
| 1 | `num_layers (4-16):` | How many layers in the stack (input + hidden + label layer). |
| 2 | `dendrite length:` | 1–4. Sets dendrite footprint: matrix side = `2×length+1` (e.g. 1 → 3×3 = 9 weights per cell). |
| 3 | `mutation_rate:` | Gene 3 scale for germline/somatic mutation rates (see genes section). |
| 4 | `lower_allele:` | Lower bound for random Conway genes (OT/IT/BT). |
| 5 | `upper_allele:` | Upper bound for random Conway genes. |
| 6 | `weight_change_threshold:` | Threshold for detecting “large” weight changes in some diagnostics. |
| 7 | `avg_weights_cell (current: …) [net suggests: …]:` | Target average fan-in for He init / scaling; often auto-updated after dendrite change. |
| 8 | `weight_decay (current: …) [net suggests: …]:` | L2-style shrink on weights each update (gene 8). |
| 9 | `bias_range (current: …) [net suggests: …]:` | Scale for **initial** bias randomization (not the same as activation slope). |
| 10 | `learning_rate (current: …) [net suggests: …]:` | Step size for weight/bias updates (gene 9). |
| 11 | `charge_delta (current: …) [net suggests: …]:` | Activity threshold for pruning / “significant” charge change (gene 7 / config). |
| 12 | `gradient_threshold (current: …) [net suggests: …]:` | Survival threshold for gradient-based pruning (gene 10 / config). |
| 13 | `activation_slope (current: …) [net suggests: …]:` | **Leaky ReLU negative slope** (gene 11) — explained below. |

After the last prompt, hidden cells may **remap weights** if dendrite size changed, and **avg_weights_cell** may be auto-set from measured fan-in.

### What is **activation slope**? When is it used?

It is a **single positive number** (typical range about **0.01** to **0.3**; web default **0.01**).

**When it runs:** On **every forward pass**, for **every hidden/output cell**, **after** the cell computes the weighted sum of upstream charges plus bias (the “pre-activation” value). That value is passed through a **leaky ReLU**:

- If **pre-activation > 0** → the cell’s **charge** becomes that value (unchanged).
- If **pre-activation ≤ 0** → the cell’s **charge** becomes **`activation_slope × pre-activation`**.

So the slope only affects **negative or zero** linear outputs. It does **not** set the bias initial range (that’s **bias_range** earlier in the **E** wizard).

**Intuition:**

- **Small slope (e.g. 0.01)** — strong suppression of negative side: neuron acts like a **sharp** rectifier; weak inputs die out quickly.
- **Larger slope (e.g. 0.2–0.3)** — more signal leaks through when the linear part is negative; neuron is **more permissive**.

**Backprop:** The derivative of that activation uses the same slope on the negative side, so it also affects **how error flows backward** through that cell.

With **U** (autonomous) on, each cell can carry its **own** gene 11 value; with **U** off, all cells use the global **activation_slope** you set here (and pruning still uses global thresholds where applicable).

---

## Keyboard Controls

### Training & Learning

| Key | Action |
|-----|--------|
| **T** | Toggle training mode (forward pass + optional backprop) |
| **B** | Toggle backpropagation (error signal + weight updates) |
| **F / R** | Forward / Reverse charge flow direction |
| **M** | Load training data — **J** (file), **D** then **M** or **F** (hosted MNIST / Fashion demo), or **M** (synthetic) |
| **K** | Set gradient minibatch size (accumulate gradients over K samples before applying) |
| **I** | Change learning rate (with suggestion based on fan-in) |

### Evolution & Pruning

| Key | Action |
|-----|--------|
| **Space** | Toggle running mode (enables evolution loop: Andromida birth/death + pruning) |
| **A** | Toggle Andromida mode (genetic birth/death using cell genes 0–2) |
| **P** | Toggle charge-delta pruning (low charge swing — gene 7) |
| **O** | Toggle gradient pruning (low avg gradient — gene 10) |
| **Y** | Toggle weight-magnitude pruning (max \|weight\| too small — gene 12) |
| **Z** | Toggle contribution-score pruning (charge_diff × gradient too low — gene 13) |
| **=** | Toggle prune logic between AND / OR (for charge-based pruning) |
| **C** | Change pruning parameters: charge delta, gradient threshold, contribution score, percentile |
| **U** | Toggle autonomous cell genes (per-cell evolved vs global config for genes 3–13) |

### Network Editing & Reset

| Key | Action |
|-----|--------|
| **E** | Edit all 13 parameters — see **Key E — what you will enter** |
| **X** | Reset network weights/biases (He initialization, auto-computes scaling) |
| **W** | Reset gradient tracking for all cells |
| **N** | Nuke all hidden-layer cells |
| **S / L** | Save / Load network state (JSON) |
| **Mouse** | Left-click: place/remove cells. Right-click / Ctrl+click: inspect cell. Drag: paint cells |

### Visualization & Display

| Key | Action |
|-----|--------|
| **G** | 2D: switch genes / proteins display. 3D: cycle color mode (Charge → Error → Gradient → Weight → Contribution) |
| **V** | Cycle statistics views (Settings + Pruning / Averages / Cell Types) |
| **D** | Toggle display updating (training + plots keep running; only cell rendering is paused) |
| **3** | Toggle 3D Three.js view |
| **4** | Toggle 3D backprop visualization (full network, Error color mode) |
| **Q** | Dump per-layer telemetry report to stats pane |
| **H** | Jump to next **##** heading in the README help panel (2D only) |
| **?** | Scroll help panel to top |

---

## Files and Build

| File | Role |
|------|------|
| `README.md` | **Canonical manual** — rendered as the in-app help preview (Markdown → HTML at build time) |
| `neurosim_web.js` | All simulation, training, visualization, and UI logic |
| `build_neurosim_web.py` | Reads `README.md` + `neurosim_web.js`, writes `neurosim_web.html` (requires `pip install markdown`) |
| `get_help_defs.py` | Legacy duplicate of early help strings; **not** used by the build — kept for reference only |
| `mnist_to_neurosim_web_json.py` | Downloads MNIST/Fashion-MNIST and writes compact JSON training files |
| `mnist_demo_500.json` | Bundled MNIST demo (500 samples); **M** → **D** → **M** |
| `fashion-mnist_demo_500.json` | Bundled Fashion-MNIST demo (500 samples); **M** → **D** → **F** |
| `neurosim_web.html` | Built output — single-file app (HTML + CSS + JS inline) |
| `neurosim_python/` | Legacy Python/pygame desktop version (see its own README) |

### Building the standalone HTML

```bash
python3 -m pip install markdown   # once — converts README.md for the help panel
python3 build_neurosim_web.py
```

Re-run after editing `README.md`, `neurosim_web.js`, or styles inside `build_neurosim_web.py`.

### Running locally

**Option A — direct open:** Double-click `neurosim_web.html`. Works for most features, but Chromium may block some `file://` loads.

**Option B — local HTTP server (recommended):**

```bash
python3 -m http.server 8765
# then open http://localhost:8765/neurosim_web.html
```

This avoids all `file://` security restrictions and enables the **M** → **D** demo-fetch path.

---

## Loading MNIST Training Data

**Quick demo (500 samples):** Press **M** → **D** → OK, then choose **M** for `mnist_demo_500.json` or **F** for `fashion-mnist_demo_500.json` (both ship in the repo for GitHub Pages).

**Full dataset (up to 5000+ samples):** Generate a local file with the conversion script:

```bash
python3 mnist_to_neurosim_web_json.py --dataset mnist --count 5000 -o mnist_training_web.json
# Fashion-MNIST: --dataset fashion
# SSL issues: add --no-verify-ssl
# Custom slice: --offset 1000 --count 2000
```

Then press **M** → **J** → select the file.

The compact format (default) stores only charge values and labels — ~22 MB for 5000 samples vs ~1.9 GB verbose. Use `--verbose-cells` for the old format if needed.

---

## Troubleshooting

**`file://` security errors:** Chromium may log *"Unsafe attempt to load URL … 'file:' URLs are treated as unique security origins"*. Each `file://` path is treated as its own origin, so subframe loads and fetch calls can fail. **Fix:** serve the repo over HTTP (`python3 -m http.server 8765`) and open via `http://localhost:8765/neurosim_web.html`.

**Large JSON files crashing the browser:** If you generated verbose-format JSON for thousands of samples, the file can be 1+ GB. Use the compact format (default) or reduce `--count`.

**Loading a network during training:** Safe — the epoch counter resets automatically so training resumes cleanly from epoch 0.

---

## Results

- **MNIST digits:** 100/100 correct on 6-layer network with 25 weights/cell
- **Fashion-MNIST:** 98/100 on similar architecture
- Networks survive save/reload and can continue training or evolution

---

## Author

Jonathan Marc Rothberg

## License

MIT
