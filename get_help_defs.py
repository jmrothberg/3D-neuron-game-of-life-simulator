"""
Legacy help strings (flat text). The web app help panel is built from README.md via
build_neurosim_web.py (Markdown → HTML). This module is kept for reference or offline
copy/paste only — it is not imported by the build script.
"""


def get_defs():
     
    jmr_defs = """
    THE 14 GENES  (inherited, mostly stable — the cell's genotype)

    ── Breeding Genes (always per-cell) ──────────────────────────────────────
    Gene 0  Overcrowding Tolerance (OT)   Max neighbors before death. OT=3 → dies if >3 neighbors.
    Gene 1  Isolation Tolerance (IT)       Min neighbors to survive.  IT=2 → dies if <2 neighbors.
    Gene 2  Birth Threshold (BT)           Exact neighbors for reproduction. BT=3 → born if 3 neighbors.
            Offspring inherit genes by crossover from two parents. Germline mutation can give
            a BT value neither parent had — allowing colonization of new neighborhoods.

    ── Network Structure Genes (per-cell when U=on, else global config) ──────
    Gene 3  Mutation Rate (MR)             Chance of gene mutation.  Birth: MR/1000.  Life: MR/100000.
    Gene 4  Dendrite Size (WG)             Weight count: 9 (3×3), 25 (5×5), 49 (7×7), or 81 (9×9).
            Determines reach = (√WG − 1) / 2. Larger = wider receptive field.
    Gene 5  Bias Range (BR)                Initial bias magnitude: 0.001 or 0.01.
    Gene 6  Fan-In (AW)                    He initialization scaling: √(2/AW).  Measured from actual connections.
    Gene 7  Charge Delta (CD)              Activity threshold for pruning. Log-uniform 1e-6 to 1e-2.
    Gene 8  Weight Decay (WD)              L2 regularization strength. Log-uniform 1e-6 to 1e-4.

    ── Learning Dynamics Genes ───────────────────────────────────────────────
    Gene 9  Learning Rate (LR)             Synaptic plasticity speed.  Range 0.003–0.05.
    Gene 10 Gradient Threshold (GT)        Gradient-pruning survival threshold. Log-uniform 1e-8 to 1e-4.
    Gene 11 Activation Slope (AS)          Leaky ReLU negative slope: 0.01 (selective) to 0.3 (permissive).

    ── Pruning Sensitivity Genes ─────────────────────────────────────────────
    Gene 12 Weight Prune Threshold (WPT)   Min max-|weight| to survive. Log-uniform 1e-3 to 1e-1.
    Gene 13 Min Contribution Score (MCS)   Min contribution score to survive. Log-uniform 1e-6 to 1e-2.
            Contribution score = max(charge_diff_fwd, charge_diff_rev) × avg |gradient|.
    """

    jmr_defs2 = """
    THE 5 PROTEINS  (dynamic, change every training step — the cell's phenotype)

    Charge     Activation signal.   Forward: leaky_ReLU(bias + Σ(upstream_charge × weight)).  Range [−10, 10].
    Error      Backprop signal.     Backward: accumulated from downstream errors × reversed weights.
    Bias       Baseline offset.     Updated each step: bias -= lr × error.  Initialized near 0.
    Weights    Synaptic strengths.  He-init: randn × √(2/fan_in).  Update: w -= lr × grad + decay × w.
    Gradient   Learning signal.     gradient = error × upstream_charge, clipped.

    THE 6 CELL MEMORY FIELDS  (rolling statistics, stored in the cell, used for pruning)

    max_charge_diff_forward    Max charge swing (max−min) across forward-pass samples.
    max_charge_diff_reverse    Max charge swing across reverse-pass samples.
    avg_gradient_magnitude     Rolling mean of |gradient| over recent samples (window = training data size).
    contributionScore          max(charge_diff_fwd, charge_diff_rev) × avg_gradient_magnitude.
    significant_charge_change  Sticky flag: has charge ever exceeded gene 7. Used only for Conway death protection.
    significant_gradient_change Sticky flag: has gradient ever exceeded gene 10. Conway death protection only.

    ── How They Interact ─────────────────────────────────────────────────────
    Gene 4  → how many weights        Gene 7  → activity threshold     Gene 9  → learning speed
    Gene 8  → weight decay rate        Gene 10 → gradient survival     Gene 11 → activation selectivity
    Gene 12 → weight-prune sensitivity Gene 13 → contribution-prune sensitivity

    3×3 Cell Display:  OT      IT      BT
                       MR      Charge  Error
                       Bias    Weight  Gradient

    U key toggles autonomous genes: OFF = all cells share global config values.
                                    ON  = each cell evolves its own gene values.
    """

    conways_defs = """
    CONWAY'S GAME OF LIFE vs JMR'S VERSION

    Original Conway's rules (fixed for all cells):
      Overcrowding (OT=3): >3 neighbors → die.
      Isolation (IT=2):    <2 neighbors → die.
      Birth (BT=3):        exactly 3 neighbors → new cell.
      No genes, no mutation, no evolution. Rules are static.

    JMR's version makes OT, IT, BT into per-cell GENES that are inherited:
      Gene 0 (OT): neighbors <= OT to survive (dies if more)
      Gene 1 (IT): neighbors >= IT to survive (dies if fewer)
      Gene 2 (BT): neighbors == BT for birth
      Survival band: IT <= neighbors <= OT  (OT > IT enforced)

    Offspring inherit genes via crossover from two parents + mutation.
    This creates evolving populations with diverse survival strategies.
    """

    how_network_works = """
    HOW THE NETWORK IS DIFFERENT FROM A TRADITIONAL NN

    Traditional NN: weights in layer-to-layer matrices owned by the network.
    This simulator: weights inside each cell's dendrites — a flat 1D array the cell owns and updates.

    The Grid:
      Layer 0 (Input)   →   Layers 1..N-2 (Hidden)   →   Layer N-1 (Desired)
      MNIST pixels           cells learn & evolve          labels (0-9)

    Each cell has its own learning rate (gene 9), weight decay (gene 8), activation curve (gene 11),
    and pruning sensitivity (genes 12-13).  No global optimizer — each cell runs its own gradient descent.

    5 PRUNING STRATEGIES  (environmental pressure on cells)

    1. Activity Pruning (P key)     charge_diff < gene 7  → die.   AND/OR logic for fwd/rev (= key).
    2. Gradient Pruning (O key)     avg |gradient| < gene 10  → die.
    3. Weight-Magnitude (with P)    max |weight| < gene 12  → die.   Near-zero weights = no signal.
    4. Contribution Score (with P)  contributionScore < gene 13  → die.  Combines activity + learning.
    5. Percentile (end of epoch)    Bottom N% by contribution score killed.  Set via C key (0=off).

    In autonomous mode, thresholds live inside the cell (genes 7, 10, 12, 13) — cells evolve their
    own pruning resilience.  In non-autonomous mode, all cells use the global config values.
    """

    forward_pass = """
    FORWARD PASS (one cell's view)

    Cell X looks at the layer ABOVE (closer to input), gathers charges within dendrite reach:

    Layer above:            This cell:
     [A][B][C]
     [D][E][F] ────────>    Cell X at (x,y)
     [G][H][I]
     3×3 reach (gene 4 = 9 weights)

    Step 1: Gather upstream charges (skip empty positions)
    Step 2: charge = bias + A.charge×w[0] + B.charge×w[1] + … + I.charge×w[8]
    Step 3: Leaky ReLU:  if charge > 0 → keep.  if ≤ 0 → charge × slope (gene 11).
    Step 4: Clip to [−10, 10], store as new charge.
    Step 5: Update cell memory: push charge to history, recompute max_charge_diff, contributionScore.

    Weight index:  idx = (dx + reach) × matrix_width + (dy + reach)
      (-1,-1)→0  (-1,0)→1  (-1,+1)→2
      ( 0,-1)→3  ( 0,0)→4  ( 0,+1)→5
      (+1,-1)→6  (+1,0)→7  (+1,+1)→8
    """

    how_backprop_works = """
    BACKWARD PASS: 2 JOBS PER CELL

    Job 1: COMPUTE ERROR SIGNAL
      Output layer (N-2):  error = (my_charge − desired) × ReLU_derivative(charge)
      Hidden layers:       error = Σ(cell_below.error × cell_below.weights[rev_idx]) × ReLU_derivative

    Job 2: UPDATE MY WEIGHTS
      For each upstream cell:
        gradient = my_error × upstream.charge
        w[idx] -= lr × gradient + decay × w
        bias   -= lr × my_error
      lr = gene 9, decay = gene 8 (per-cell in autonomous mode)

    Job 3: UPDATE CELL MEMORY
      Push |gradient| to history → recompute avg_gradient_magnitude → recompute contributionScore.
    """

    how_backprop_works2 = """
    WHY reversed_index WORKS

    Forward:  Cell 5 (below) looks UP at Cell X through weight_index:
      idx = (dx + reach) × matrix_width + (dy + reach)

    Backward: Cell X looks DOWN at Cell 5 through reversed_index:
      rev = len(weights) − 1 − idx

    For 3×3:  fwd 0 ↔ rev 8  (-1,-1) ↔ (+1,+1)
              fwd 1 ↔ rev 7  (-1, 0) ↔ (+1, 0)
              fwd 4 ↔ rev 4  ( 0, 0) ↔ ( 0, 0)   center stays
              fwd 8 ↔ rev 0  (+1,+1) ↔ (-1,-1)

    Maps (dx,dy) → (−dx,−dy) = transpose.  Standard backprop from a flat 1D array.

    FULL PICTURE:
    Layer above    This cell    Layer below
     [A][B][C]                   [1][2][3]
     [D][E][F]      Cell X       [4][5][6]
     [G][H][I]     error=?       [7][8][9]
        ↓                            ↓
    Update X's      Compute X's error from
    weights using   below cells' errors + reversed weights
    A–I charges
    """

    controls = """
    ── KEYBOARD CONTROLS ─────────────────────────────────────────────────────

    Mouse:  Left-click/Drag = place cells.  Right-click/Ctrl+click = inspect cell.

    ── Evolution ──
    Space   Toggle Running (evolution loop)       A   Toggle Andromida (genetic birth/death)
    U       Toggle Autonomous Genes (per-cell)    =   Toggle Prune Logic AND/OR

    ── Training ──
    T   Toggle Training         B   Toggle Backprop          F/R   Forward / Reverse direction
    I   Set Learning Rate       K   Set Minibatch Size (gradient accumulation over K samples)

    ── Pruning ──
    P   Toggle Activity Prune (charge + weight-mag + contribution score)
    O   Toggle Gradient Prune (avg |gradient| < gene 10)
    C   Set all pruning params: charge delta, gradient threshold, min contribution score, prune percentile

    ── View ──
    G   2D: genes/proteins toggle.  3D: cycle color mode (Charge → Error → Gradient → Weight → Contribution)
    V   Cycle stats views: Settings+Pruning / Averages / Cell Types
    Q   Dump per-layer telemetry          D   Toggle display (training still runs)
    H   Cycle help screens                ?   Scroll help to top
    3   Toggle 3D view                    4   Toggle 3D backprop highlight

    ── Data / IO ──
    M   Load training data (J=file, D=demo, M=synthetic)     E   Edit all parameters (see README: Key E)
    X   Reset network genes/proteins (He init)                N   Nuke hidden cells
    S   Save network (JSON)              L   Load network     W   Reset gradient tracking

    ── Key E — thirteen prompts in order ──
    num_layers, dendrite length, mutation_rate, lower_allele, upper_allele,
    weight_change_threshold, avg_weights_cell, weight_decay, bias_range, learning_rate,
    charge_delta, gradient_threshold, activation_slope.
    activation_slope (gene 11): leaky ReLU NEGATIVE slope, used every FORWARD pass after weighted sum+bias.
    If that sum is <= 0, charge = slope * sum; if > 0, charge = sum. Typical 0.01 (sharp) to 0.3 (leaky).
    Not the same as bias_range (only initial bias randomization scale).
    """

    return jmr_defs, jmr_defs2, conways_defs, how_network_works, forward_pass, how_backprop_works, how_backprop_works2, controls
