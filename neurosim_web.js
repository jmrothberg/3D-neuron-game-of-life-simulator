/**
 * Browser port of neurosim/ (pygame + OpenGL) — expects global THREE and README_HTML (from build_neurosim_web.py).
 * JSON save/load; synthetic training; optional training JSON import.
 */
(function () {
  'use strict';

  const COLORS = [[13,0,184],[255,165,0],[255,192,203],[255,255,0],[0,255,0],[255,0,0],[179,0,255],[0,0,255]];
  const WINDOW_WIDTH = 1008, WINDOW_HEIGHT = 1008, WINDOW_EXTENSION = 100;
  const HELP_PANEL_WIDTH = 500, MAIN_SURFACE_WIDTH = 1508 - HELP_PANEL_WIDTH;
  const CELL_SIZE = 9;
  const WIDTH = (WINDOW_WIDTH / CELL_SIZE / 4) | 0;
  const HEIGHT = (WINDOW_HEIGHT / CELL_SIZE / 4) | 0;
  const ARRAY_LAYERS = 16;
  /* Y-axis max defaults; sliders let user zoom in/out on the loss scale. */
  const PLOT_YMAX_MIN = 0.1;
  const PLOT_YMAX_MAX = 30;
  const FASHION_LABELS = '0 T-shirt/top | 1 Trouser | 2 Pullover | 3 Dress | 4 Coat | 5 Sandal | 6 Shirt | 7 Sneaker | 8 Bag | 9 Ankle boot';
  const WEIGHT_DRAW_THRESHOLD = 0.01;
  const EPS = 1e-8;

  /* Expand compact training sample {c:[784 charges], l:label} into {layer0, layerLast} cell-JSON grids.
     Compact format written by mnist_to_neurosim_web_json.py v2 — ~100x smaller than full cell objects. */
  const _COMPACT_GENES = [15,2,3,10, 9,0.01,5,0.001,1e-6, 0.01,1e-7,0.1, 0.01,0, 50];
  function _compactCell(layer, x, y, charge) {
    return {x:x, y:y, layer:layer, genes:_COMPACT_GENES, weights:[0,0,0,0,0,0,0,0,0],
      bias:0, charge:charge, error:EPS, gradient:0, reach:1,
      forward_charges:[], reverse_charges:[], max_charge_diff_forward:0, max_charge_diff_reverse:0,
      significant_charge_change_forward:false, significant_charge_change_reverse:false,
      gradient_history:[], avg_gradient_magnitude:0, significant_gradient_change:false, contributionScore:0};
  }
  function expandCompactSample(sample, layerLast) {
    const charges = sample.c, label = sample.l;
    const layer0 = [], layerL = [];
    for (let x = 0; x < GRID_W; x++) {
      const row0 = [], rowL = [];
      for (let y = 0; y < GRID_W; y++) {
        row0.push(_compactCell(0, x, y, charges[x * GRID_W + y]));
        rowL.push(null);
      }
      layer0.push(row0); layerL.push(rowL);
    }
    for (let d = 0; d < 10; d++) layerL[9+d][14] = _compactCell(layerLast, 9+d, 14, d === label ? 1.0 : 0.0);
    return {layer0: layer0, layerLast: layerL};
  }
  const GRID_W = 28;

  /** Prepended in #helpScroll — keep in sync with README / mnist_to_neurosim_web_json.py */
  const QUICK_START = [
    'QUICK START — read this if training will not run',
    '',
    'WHERE ARE MNIST FILES?',
    '  • Raw downloads go to a folder .mnist_cache/ next to mnist_to_neurosim_web_json.py',
    '    (your project clone). Open that folder in Finder if you want to see the .gz files.',
    '  • The JSON you load in the browser is ONLY the file you pass with -o — for example:',
    '      cd same-folder-as-the-script',
    '      python3 mnist_to_neurosim_web_json.py -o ./mnist_training_web.json',
    '    creates ./mnist_training_web.json in THAT terminal current directory (check pwd).',
    '    In the file picker, navigate to that same folder and pick the .json.',
    '',
    '1) Click the big white grid once (keyboard focus). Press ? = jump help to top; H = next README ## section.',
    '   Full manual is the README below. Key E = 13 parameter prompts; activation_slope = leaky ReLU slope (see README “Key E”).',
    '',
    '2) Paint hidden neurons (layers 1 … N-2): LEFT-CLICK or CLICK-DRAG on the 16 small',
    '   quadrant squares. Layer 0 (input) and last layer (targets) come from training data.',
    '   You need some cells in hidden layers or the network has nothing to train.',
    '',
    '3) Load training batches — key M (not in 3D view):',
    '   • Dialog 1: type J then OK = load real MNIST/Fashion JSON.',
    '   •          type D then OK = fetch demo: pick M (MNIST) or F (Fashion), 500 samples each.',
    '   •          type M then OK = quick fake data (no file).',
    '   • Dialog 2–3: batch size and start index (for J, count must be ≤ samples in file).',
    '   • Then your browser opens a file picker — choose e.g. mnist_training_web.json',
    '     (create it on your computer:  python3 mnist_to_neurosim_web_json.py -o mnist_training_web.json)',
    '',
    '4) Start training — key T. Press T again to stop.',
    '   Tip: key B toggles backprop (learning); leave B ON to update weights.',
    '   Key K — gradient minibatch size (1=SGD per image; larger=average grad over K images then one update; typical MNIST 16–64).',
    '   Plot strip: sliders = X history length before scroll (green=epoch mean, blue=gradient-minibatch mean).',
    '',
    '5) Full manual: README.md is rendered below (tables, diagrams). Key H jumps between each ## section.',
    '',
    '6) If keys do nothing: click the canvas again. If 3D (key 3) is on, M/H/V etc.',
    '   only work after you press 3 again to return to 2D.',
    '',
    '7) Offline: you still need the internet once so Three.js loads (or 3D tab stays blank).',
    '   Saving JSON (S) and loading (L) works offline after that.',
    '',
  ].join('\n');

  function clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function randInt(a, b) { return a + Math.floor(Math.random() * (b - a + 1)); }
  function choice(arr) { return arr[(Math.random() * arr.length) | 0]; }
  function uniform(a, b) { return a + Math.random() * (b - a); }
  function randn() {
    let u = 1 - Math.random(), v = 1 - Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function log10(x) { return Math.log(x) / Math.LN10; }
  function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
  function sumAbsWeights(cells, nl) {
    let s = 0;
    for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
      for (let z = 1; z < nl - 1; z++) {
        const c = cells[x][y][z];
        if (c) for (let i = 0; i < c.weights.length; i++) s += Math.abs(c.weights[i]);
      }
    }
    return s;
  }

  function setDefaultValues() {
    const c = {
      num_layers: 8, length_of_dendrite: 1, weight_matrix: 3, number_of_weights: 9,
      mutation_rate: 10, lower_allele_range: 2, upper_allele_range: 15,
      autonomous_network_genes: false,
      learning_rate: 0.01, bias_range: 0.01, avg_weights_cell: 5, weight_decay: 1e-5,
      charge_delta: 0.01, gradient_threshold: 1e-4, gradient_clip_range: 0.5,
      weight_change_threshold: 0.005, activation_slope: 0.01,
      how_much_training_data: 20, start_index: 0, epsilon: EPS,
      gradient_minibatch_size: 1,
      shuffle_epoch: true,
      /* Prune cells where max |weight| < this threshold (weight-magnitude pruning via O key combo) */
      weight_prune_threshold: 0.01,
      /* Contribution-score threshold: cells with score < this die (0 = off). Score = charge_diff × gradient. */
      min_contribution_score: 0,
      /* Percentile pruning: kill bottom N% of cells by contribution score each epoch (0 = off, e.g. 10 = bottom 10%) */
      prune_percentile: 0,
      /* Immune period: number of training cycles (forward passes) a newborn cell survives before becoming prunable (gene 14) */
      immune_period: 50,
    };
    c.updateDerived = function () {
      this.weight_matrix = 2 * this.length_of_dendrite + 1;
      this.number_of_weights = this.weight_matrix * this.weight_matrix;
    };
    c.updateDerived();
    return c;
  }

  /* Fisher-Yates shuffle of an integer index array. */
  function shuffleIndices(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  function makeIndexArray(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; }

  /* Compute average effective fan-in across all hidden cells (real neighbor count, not weight-array size).
     Used to auto-set avg_weights_cell for proper He initialization scaling. */
  function computeAvgFanIn(state, config) {
    const nl = config.num_layers, reach = config.length_of_dendrite;
    let totalFanIn = 0, count = 0;
    for (let z = 1; z < nl - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (!c) continue;
        let fan = 0;
        const r = config.autonomous_network_genes ? c.reach : reach;
        for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT && state.cells[nx][ny][z - 1]) fan++;
        }
        totalFanIn += fan; count++;
      }
    return count > 0 ? totalFanIn / count : config.number_of_weights;
  }

  /* Scan the live network to suggest sensible defaults for key parameters.
     Returns an object with suggested values and the measurements they came from. */
  function suggestParams(state, config) {
    const nl = config.num_layers;
    let nCells = 0, totalFanIn = 0;
    const gradMags = [], chargeMags = [], errorMags = [], weightMags = [];
    const biasAbs = [];
    for (let z = 1; z < nl - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (!c) continue;
        nCells++;
        gradMags.push(Math.abs(c.gradient));
        chargeMags.push(Math.abs(c.charge));
        errorMags.push(Math.abs(c.error));
        biasAbs.push(Math.abs(c.bias));
        let wSum = 0, fan = 0;
        const r = config.autonomous_network_genes ? c.reach : config.length_of_dendrite;
        for (let i = 0; i < c.weights.length; i++) wSum += Math.abs(c.weights[i]);
        weightMags.push(c.weights.length > 0 ? wSum / c.weights.length : 0);
        for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT && state.cells[nx][ny][z - 1]) fan++;
        }
        totalFanIn += fan;
      }
    if (nCells === 0) return null;
    gradMags.sort((a, b) => a - b);
    chargeMags.sort((a, b) => a - b);
    errorMags.sort((a, b) => a - b);
    weightMags.sort((a, b) => a - b);
    biasAbs.sort((a, b) => a - b);
    const pct = (arr, p) => arr[Math.min(((p * arr.length) | 0), arr.length - 1)];
    const med = (arr) => pct(arr, 0.5);

    const avgFanIn = totalFanIn / nCells;
    const medGrad = med(gradMags);
    const p25Grad = pct(gradMags, 0.25);
    const medWeight = med(weightMags);
    const medCharge = med(chargeMags);
    const medError = med(errorMags);
    const medBias = med(biasAbs);

    /* Learning rate: ~1/fan_in is a safe starting point; clamp to reasonable range */
    const sugLR = clip(1.0 / Math.max(avgFanIn, 1), 0.001, 0.05);
    /* Charge delta: set at ~10th percentile of charge magnitudes so only truly silent cells get pruned */
    const sugCD = Math.max(1e-6, pct(chargeMags, 0.10));
    /* Gradient threshold: set at ~10th percentile of gradient magnitudes */
    const sugGT = Math.max(1e-8, pct(gradMags, 0.10));
    /* Weight decay: smaller when weights are small, larger when weights are big */
    const sugWD = clip(medWeight > 0 ? 0.01 * medWeight : 1e-5, 1e-6, 1e-3);
    /* Bias range for He init: based on current network bias magnitudes */
    const sugBR = clip(medBias > 0 ? medBias * 2 : 0.01, 0.001, 0.1);
    /* Weight prune threshold: below 10th percentile of avg weight magnitudes */
    const sugWPT = Math.max(1e-4, pct(weightMags, 0.10));
    /* Activation slope: keep current unless evidence suggests otherwise */
    const sugAS = config.activation_slope;
    /* Gradient clip: ~2× the 95th percentile gradient */
    const p95Grad = pct(gradMags, 0.95);
    const sugGCR = clip(p95Grad > 0 ? p95Grad * 2 : 0.5, 0.1, 5.0);

    return {
      nCells, avgFanIn,
      medGrad, p25Grad, medWeight, medCharge, medError, medBias,
      lr: +sugLR.toPrecision(3),
      charge_delta: +sugCD.toPrecision(2),
      gradient_threshold: +sugGT.toPrecision(2),
      weight_decay: +sugWD.toPrecision(2),
      bias_range: +sugBR.toPrecision(2),
      weight_prune_threshold: +sugWPT.toPrecision(2),
      activation_slope: sugAS,
      gradient_clip_range: +sugGCR.toPrecision(2),
      avg_weights_cell: Math.max(1, Math.round(avgFanIn)),
    };
  }

  function makeCellsGrid() {
    const cells = new Array(WIDTH);
    for (let x = 0; x < WIDTH; x++) {
      cells[x] = new Array(HEIGHT);
      for (let y = 0; y < HEIGHT; y++) {
        cells[x][y] = new Array(ARRAY_LAYERS).fill(null);
      }
    }
    return cells;
  }

  function SimState() {
    return {
      cells: makeCellsGrid(),
      running: false, prune: false, gradient_prune: false, weight_mag_prune: false, contrib_score_prune: false, training_mode: false,
      andromida_mode: false, charge_change_protection: true, back_prop: false,
      training_data_loaded: false, display_updating: true, simulating: true, not_saved_yet: true,
      prune_logic: 'OR', display: 'proteins', direction_of_charge_flow: '+++++>>>>>',
      show_3d_view: false, show_backprop_view: false, show_training_stats: false, display_set: 0, _3d_color_mode: 0,
      bingo_count: 0, max_bingo_count: 0, total_cells: 0, total_loss: 0, total_predictions: 0,
      running_avg_loss: 0, last_step_loss: 0, training_cycles: 0, total_weights: 0,
      total_weights_list: new Float64Array(1000),
      minibatchLossPoints: [], epochLossPoints: [], epochs: 1, batch_size: 1,
      _batch_loss_sum: 0, _batch_sample_count: 0,
      _mini_loss_sum: 0, _mini_n: 0,
      plotEpochYmax: 25, plotMinibatchYmax: 25,
      training_data_layer_0: [], training_data_num_layer_minus_1: [],
      /* M=MNIST F=Fashion S=synthetic U=unknown file X=no training data loaded — used in save filenames */
      training_dataset_code: 'X',
      rotation_x: 0, rotation_y: 0, rotation_angle: 0, zoom: -15,
      mouse_up: true, current_index: 0, side_panel_text: [], training_stats_buffer: {},
      stats_update_frequency: 1, timing: false,
      /* Per-frame index when training with display on (one sample per animation frame so the grid can repaint). */
      _training_sample_i: null,
      _shuffle_order: null,
      _max_reach_per_layer: Object.create(null), _3d_dirty: true,
      invalidateNeighborCache() {
        for (const k in this._max_reach_per_layer) delete this._max_reach_per_layer[k];
        this._3d_dirty = true;
      },
      getMaxReachForLayer(layer, cells) {
        if (this._max_reach_per_layer[layer] == null) {
          let maxR = 0;
          for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
            const c = cells[x][y][layer];
            if (c) maxR = Math.max(maxR, c.reach);
          }
          this._max_reach_per_layer[layer] = maxR;
        }
        return this._max_reach_per_layer[layer];
      },
      reset_training_metrics() {
        this.bingo_count = 0; this.max_bingo_count = 0; this.total_loss = 0; this.total_predictions = 0;
        this.running_avg_loss = 0; this.last_step_loss = 0; this.training_cycles = 0;
        this.minibatchLossPoints = []; this.epochLossPoints = []; this._batch_loss_sum = 0; this._batch_sample_count = 0;
        this._mini_loss_sum = 0; this._mini_n = 0;
      },
    };
  }

  let CellConfig = null;

  class Cell {
    static setConfig(cfg) { CellConfig = cfg; }

    constructor(layer, x, y, wpc, br, aw, cd, wd, mr, genes) {
      if (!(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)) {
        this.x = clip(x | 0, 0, WIDTH - 1); this.y = clip(y | 0, 0, HEIGHT - 1);
      } else { this.x = x; this.y = y; }
      this.layer = layer;
      if (genes == null) {
        this.initalizeAllGenes();
        this.initalizeBreedingGenes();
        this.initalizeNetworkGenes(wpc, br, aw, cd, wd, mr, null);
      } else this.genes = genes.slice();
      this.colorGenes();
      this.initializeNetworkProteins();
      this.initalizeCellMemory();
      this.colorProteins();
    }

    initalizeCellMemory() {
      this.forward_charges = []; this.reverse_charges = [];
      this.max_charge_diff_forward = 0; this.max_charge_diff_reverse = 0;
      this.significant_charge_change_forward = false; this.significant_charge_change_reverse = false;
      this.number_of_upper_layer_cells = 0; this.number_of_lower_layer_cells = 0;
      this.gradient_history = []; this.avg_gradient_magnitude = 0; this.significant_gradient_change = false;
      this.contributionScore = 0;
    }
    initalizeAllGenes() {
      this.genes = new Array(15).fill(0);
      this.colors = new Array(15).fill(0);
      this.protein_colors = new Array(15).fill(0);
    }
    initalizeBreedingGenes() {
      const cfg = CellConfig;
      const lo = cfg ? cfg.lower_allele_range : 2, hi = cfg ? cfg.upper_allele_range : 15;
      this.genes[0] = randInt(lo, hi); this.genes[1] = randInt(lo, hi); this.genes[2] = randInt(lo, hi);
      if (this.genes[0] < this.genes[1]) { const t = this.genes[0]; this.genes[0] = this.genes[1]; this.genes[1] = t; }
      this.colorGenes();
    }
    initalizeNetworkGenes(wpc, br, aw, cd, wd, mr, cells_array) {
      const cfg = CellConfig, aut = cfg && cfg.autonomous_network_genes;
      if (!aut) {
        this.genes[3] = mr; this.genes[4] = wpc | 0; this.genes[5] = br; this.genes[6] = aw;
        this.genes[7] = cd; this.genes[8] = wd;
        this.genes[9] = cfg ? cfg.learning_rate : 0.01;
        this.genes[10] = cfg ? cfg.gradient_threshold : 1e-4;
        this.genes[11] = cfg ? cfg.activation_slope : 0.01;
        this.genes[12] = cfg ? cfg.weight_prune_threshold : 0.01;
        this.genes[13] = cfg ? cfg.min_contribution_score : 0;
        this.genes[14] = cfg ? cfg.immune_period : 50;
      } else {
        this.genes[3] = randInt(0, 99);
        this.genes[4] = Math.pow(randInt(1, 3) * 2 + 1, 2);
        this.genes[5] = choice([0.001, 0.01]);
        this.reach = ((Math.sqrt(this.genes[4]) | 0) - 1) >> 1;
        if (cells_array) {
          const u = this.getUpperLayerCells(cells_array, this.reach);
          this.genes[6] = u.length > 0 ? u.length : aw;
        } else this.genes[6] = aw;
        this.genes[7] = Math.pow(10, uniform(-6, -2));
        this.genes[8] = Math.pow(10, uniform(-6, -4));
        this.genes[9] = uniform(0.003, 0.05);
        this.genes[10] = Math.pow(10, uniform(-8, -4));
        this.genes[11] = uniform(0.01, 0.3);
        this.genes[12] = Math.pow(10, uniform(-3, -1.5));
        this.genes[13] = Math.pow(10, uniform(-6, -2));
        this.genes[14] = randInt(10, 100);
      }
      this.colorGenes();
    }
    initializeNetworkProteins() {
      this.charge = 0; this.gradient = 0; this.error = EPS;
      this.backprops_remaining = this.genes[14] | 0;
      this.reach = ((Math.sqrt(this.genes[4]) | 0) - 1) >> 1;
      const fan = Math.max(this.genes[6], 1), n = this.genes[4] | 0;
      const sc = Math.sqrt(2 / (fan + 1e-8));
      this.weights = new Float64Array(n);
      for (let i = 0; i < n; i++) this.weights[i] = clip(randn() * sc, -1, 1);
      this.bias = uniform(-this.genes[5], this.genes[5]);
    }
    colorGenes() {
      const g4 = [];
      for (let i = 0; i < 4; i++) {
        const g = this.genes[i], col = COLORS[i];
        g4.push([
          clip(((col[0] * g / 15) | 0), 0, 255),
          clip(((col[1] * g / 15) | 0), 0, 255),
          clip(((col[2] * g / 15) | 0), 0, 255),
        ]);
      }
      const s4 = clip((((this.genes[4] - 9) / 72) * 255) | 0, 0, 255);
      const s5 = clip((this.genes[5] * 255) | 0, 0, 255);
      const s6 = clip((((this.genes[6] - 5) / 25) * 255) | 0, 0, 255);
      const g7v = clip(this.genes[7], 1e-6, 1e-2);
      const s7 = clip((((log10(g7v) + 6) / 4) * 255) | 0, 0, 255);
      const g8v = clip(this.genes[8], 1e-6, 1e-4);
      const s8 = clip((((log10(g8v) + 6) / 2) * 255) | 0, 0, 255);
      const colors = g4.concat([[s4, 0, s4], [0, s5, s5], [s6, 0, 0], [0, s7, 0], [0, 0, s8]]);
      const s9 = clip((((this.genes[9] - 0.003) / 0.047) * 255) | 0, 0, 255);
      const g10v = clip(this.genes[10], 1e-8, 1e-4);
      const s10 = clip((((log10(g10v) + 8) / 4) * 255) | 0, 0, 255);
      const s11 = clip((((this.genes[11] - 0.01) / 0.29) * 255) | 0, 0, 255);
      /* gene 12: weight prune threshold (log scale ~1e-3..~0.032) */
      const g12v = clip(this.genes[12], 1e-3, 0.032);
      const s12 = clip((((log10(g12v) + 3) / 1.5) * 255) | 0, 0, 255);
      /* gene 13: min contribution score (log scale ~1e-6..1e-2) */
      const g13v = clip(this.genes[13], 1e-6, 1e-2);
      const s13 = clip((((log10(g13v) + 6) / 4) * 255) | 0, 0, 255);
      /* gene 14: immune period (integer 10..100) */
      const s14 = clip((((this.genes[14] - 10) / 90) * 255) | 0, 0, 255);
      colors.push([0, s9, s9], [s10, 0, s10], [s11, s11, 0], [s12, s12, 0], [0, s13, s13], [s14, 0, s14]);
      this.colors = colors;
    }
    colorProteins() {
      this.protein_colors = new Array(12).fill(0);
      const br = this.genes[5];
      this.protein_colors[0] = [clip((Math.abs(this.bias) / (br + EPS) * 255) | 0, 0, 255), 0, clip((Math.abs(this.bias) / (br + EPS) * 255) | 0, 0, 255)];
      this.protein_colors[1] = [clip((this.max_charge_diff_forward * 255) | 0, 0, 255), 0, 0];
      this.protein_colors[2] = [0, 0, clip((this.max_charge_diff_reverse * 255) | 0, 0, 255)];
      this.protein_colors[4] = [clip((this.charge * 255) | 0, 0, 255), 0, 0];
      const em = Math.abs(this.error) + EPS;
      this.protein_colors[6] = [0, 0, clip((255 * (log10(em) + 3) / 2) | 0, 0, 255)];
      let mw = 0; for (let i = 0; i < this.weights.length; i++) mw += Math.abs(this.weights[i]);
      mw /= Math.max(1, this.weights.length);
      this.protein_colors[7] = [0, clip((mw * 255) | 0, 0, 255), 0];
      const gr = this.charge * this.error;
      this.protein_colors[8] = [0, 0, clip(Math.log(Math.abs(gr + EPS) * 55) | 0, 0, 255)];
      this.protein_colors[3] = this.protein_colors[5] = [0, 0, 0];
      this.protein_colors[9] = this.protein_colors[10] = this.protein_colors[11] = [0, 0, 0];
      this.protein_colors[12] = this.protein_colors[13] = [0, 0, 0];
    }

    computeTotalCharge(upper, reach) {
      const cfg = CellConfig;
      let WM, NW;
      if (cfg && cfg.autonomous_network_genes) {
        reach = this.reach; NW = this.genes[4]; WM = Math.sqrt(NW) | 0;
      } else WM = 2 * reach + 1;
      let ch = this.bias;
      for (let k = 0; k < upper.length; k++) {
        const [dx, dy, cell] = upper[k];
        const wi = (dx + reach) * WM + (dy + reach);
        if (wi >= 0 && wi < this.weights.length) ch += cell.charge * this.weights[wi];
      }
      this.charge = clip(ch, -10, 10);
    }
    computeTotalChargeReverse(lower, reach) {
      const cfg = CellConfig;
      const WM = 2 * reach + 1;
      let ch = this.bias;
      for (let k = 0; k < lower.length; k++) {
        const [dx, dy, cell] = lower[k];
        let cr, cwm;
        if (cfg && cfg.autonomous_network_genes) {
          cr = cell.reach; cwm = Math.sqrt(cell.genes[4]) | 0;
        } else { cr = reach; cwm = WM; }
        const rx = -dx + cr, ry = -dy + cr;
        const wi = ry * cwm + rx;
        if (wi >= 0 && wi < cell.weights.length) ch += cell.charge * cell.weights[wi];
      }
      this.charge = clip(ch, -10, 10);
    }
    relu(x) {
      const cfg = CellConfig;
      const s = (cfg && cfg.autonomous_network_genes) ? this.genes[11] : (cfg ? cfg.activation_slope : 0.01);
      return x > 0 ? x : s * x;
    }
    reluDerivative(x) {
      const cfg = CellConfig;
      const s = (cfg && cfg.autonomous_network_genes) ? this.genes[11] : (cfg ? cfg.activation_slope : 0.01);
      return x > 0 ? 1 : s;
    }
    getUpperLayerCells(cells, reach) {
      const cfg = CellConfig;
      if (cfg && cfg.autonomous_network_genes) reach = this.reach;
      const u = [];
      for (let dx = -reach; dx <= reach; dx++) for (let dy = -reach; dy <= reach; dy++) {
        const nx = this.x + dx, ny = this.y + dy;
        if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
          const c = cells[nx][ny][this.layer - 1];
          if (c) u.push([dx, dy, c]);
        }
      }
      this.number_of_upper_layer_cells = u.length;
      return u;
    }
    getLayerBelowCells(cells, reach, maxBelow) {
      const cfg = CellConfig;
      if (cfg && cfg.autonomous_network_genes) reach = this.reach;
      const out = [];
      if (this.layer + 1 >= ARRAY_LAYERS) { this.number_of_lower_layer_cells = 0; return out; }
      let maxR = maxBelow;
      if (maxR == null) {
        maxR = 0;
        for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
          const c = cells[x][y][this.layer + 1];
          if (c) maxR = Math.max(maxR, c.reach);
        }
      }
      for (let dx = -maxR; dx <= maxR; dx++) for (let dy = -maxR; dy <= maxR; dy++) {
        const nx = this.x + dx, ny = this.y + dy;
        if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
          const b = cells[nx][ny][this.layer + 1];
          if (b) {
            const brr = (cfg && cfg.autonomous_network_genes) ? b.reach : reach;
            if (Math.abs(dx) <= brr && Math.abs(dy) <= brr) out.push([dx, dy, b]);
          }
        }
      }
      this.number_of_lower_layer_cells = out.length;
      return out;
    }
    computeErrorSignal(desired, conn, reach) {
      const cfg = CellConfig;
      if (cfg && cfg.autonomous_network_genes) reach = this.reach;
      let es = EPS;
      if (desired != null) {
        es = (this.charge - desired) * this.reluDerivative(this.charge);
        this.error = clip(es, -10, 10);
      } else if (conn != null && reach != null) {
        for (let k = 0; k < conn.length; k++) {
          const [dx, dy, cell] = conn[k];
          let cr, cwm;
          if (cfg && cfg.autonomous_network_genes) {
            cr = cell.reach; cwm = Math.sqrt(cell.genes[4]) | 0;
          } else { cr = reach; cwm = 2 * reach + 1; }
          const wi = (dx + cr) * cwm + (dy + cr);
          const rev = cell.weights.length - 1 - wi;
          if (rev >= 0 && rev < cell.weights.length)
            es += cell.error * cell.weights[rev] * this.reluDerivative(this.charge);
        }
        this.error = clip(es, -10, 10);
      } else this.error = EPS;
    }
    updateWeightsAndBias(conn, lr, reach, wd) {
      const cfg = CellConfig;
      const gcr = cfg ? cfg.gradient_clip_range : 1;
      let WM;
      if (cfg && cfg.autonomous_network_genes) {
        reach = this.reach;
        WM = Math.sqrt(this.genes[4]) | 0;
        wd = this.genes[8]; lr = this.genes[9];
      } else WM = 2 * reach + 1;
      if (this.error == null) this.error = EPS;
      for (let k = 0; k < conn.length; k++) {
        const [dx, dy, cell] = conn[k];
        let g = this.error * cell.charge;
        g = clip(g, -gcr, gcr);
        this.gradient = g;
        this.updateGradientImportance(g);
        const wi = (dx + reach) * WM + (dy + reach);
        if (wi >= 0 && wi < this.weights.length)
          this.weights[wi] -= lr * g + wd * this.weights[wi];
      }
      this.bias -= lr * this.error;
    }
    accumulateWeightGradients(conn, reach) {
      const cfg = CellConfig;
      const gcr = cfg ? cfg.gradient_clip_range : 1;
      let WM;
      if (cfg && cfg.autonomous_network_genes) {
        reach = this.reach;
        WM = Math.sqrt(this.genes[4]) | 0;
      } else WM = 2 * reach + 1;
      if (this.error == null) this.error = EPS;
      if (!this._dw || this._dw.length !== this.weights.length) this._dw = new Float64Array(this.weights.length);
      if (this._db == null) this._db = 0;
      for (let k = 0; k < conn.length; k++) {
        const [dx, dy, cell] = conn[k];
        let g = this.error * cell.charge;
        g = clip(g, -gcr, gcr);
        this.gradient = g;
        this.updateGradientImportance(g);
        const wi = (dx + reach) * WM + (dy + reach);
        if (wi >= 0 && wi < this.weights.length) this._dw[wi] += g;
      }
      this._db += this.error;
    }
    clearMinibatchAccumulator() {
      if (this._dw) this._dw.fill(0);
      this._db = 0;
    }
    applyAccumulatedWeights(conn, lr, reach, wd, batchDenom) {
      const cfg = CellConfig;
      if (!this._dw || batchDenom < 1) return;
      let WM;
      if (cfg && cfg.autonomous_network_genes) {
        reach = this.reach;
        WM = Math.sqrt(this.genes[4]) | 0;
        wd = this.genes[8]; lr = this.genes[9];
      } else WM = 2 * reach + 1;
      const invB = 1 / batchDenom;
      for (let k = 0; k < conn.length; k++) {
        const [dx, dy, cell] = conn[k];
        const wi = (dx + reach) * WM + (dy + reach);
        if (wi >= 0 && wi < this.weights.length) {
          const gbar = this._dw[wi] * invB;
          this.weights[wi] -= lr * gbar + wd * this.weights[wi];
        }
      }
      this.bias -= lr * ((this._db || 0) * invB);
      this._dw.fill(0);
      this._db = 0;
    }
    updateCharge(nc, dir) {
      const cfg = CellConfig, hm = cfg ? cfg.how_much_training_data : 20;
      const cdThresh = (cfg && cfg.autonomous_network_genes) ? this.genes[7] : (cfg ? cfg.charge_delta : this.genes[7]);
      if (dir === 'forward') {
        this.forward_charges.push(nc);
        if (this.forward_charges.length > hm) this.forward_charges.shift();
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < this.forward_charges.length; i++) {
          const v = this.forward_charges[i]; if (v < mn) mn = v; if (v > mx) mx = v;
        }
        this.max_charge_diff_forward = mx - mn;
        if (this.max_charge_diff_forward > cdThresh) this.significant_charge_change_forward = true;
      } else if (dir === 'reverse') {
        this.reverse_charges.push(nc);
        if (this.reverse_charges.length > hm) this.reverse_charges.shift();
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < this.reverse_charges.length; i++) {
          const v = this.reverse_charges[i]; if (v < mn) mn = v; if (v > mx) mx = v;
        }
        this.max_charge_diff_reverse = mx - mn;
        if (this.max_charge_diff_reverse > cdThresh) this.significant_charge_change_reverse = true;
      }
      this.charge = nc;
      this._recomputeContributionScore();
    }
    updateGradientImportance(ng) {
      const cfg = CellConfig, hm = cfg ? cfg.how_much_training_data : 20;
      const gt = (cfg && cfg.autonomous_network_genes) ? this.genes[10] : (cfg ? cfg.gradient_threshold : 1e-7);
      this.gradient_history.push(Math.abs(ng));
      if (this.gradient_history.length > hm) this.gradient_history.shift();
      this.avg_gradient_magnitude = mean(this.gradient_history);
      if (this.avg_gradient_magnitude > gt) this.significant_gradient_change = true;
      this._recomputeContributionScore();
    }
    /* Contribution score = max(charge variability across directions) × avg gradient.
       Combines "is this cell active?" with "is it learning?" */
    _recomputeContributionScore() {
      const cd = Math.max(this.max_charge_diff_forward, this.max_charge_diff_reverse);
      this.contributionScore = cd * this.avg_gradient_magnitude;
    }
    resetDirectionalChargeHistory(dir) {
      if (dir === 'forward' || dir === '+++++>>>>>') {
        this.forward_charges.length = 0; this.max_charge_diff_forward = 0;
        this.significant_charge_change_forward = false;
      } else if (dir === 'reverse' || dir === '<<<<<-----') {
        this.reverse_charges.length = 0; this.max_charge_diff_reverse = 0;
        this.significant_charge_change_reverse = false;
      }
      this.gradient_history.length = 0; this.avg_gradient_magnitude = 0; this.significant_gradient_change = false;
      this.contributionScore = 0;
    }
    resetGradientChange() {
      const cfg = CellConfig;
      this.significant_gradient_change = false; this.gradient_history.length = 0; this.gradient = 0;
      this.error = cfg ? cfg.epsilon : EPS;
      this.contributionScore = 0;
    }
    remapWeights(reach) {
      const oldM = Math.sqrt(this.weights.length) | 0;
      const newM = 2 * reach + 1;
      if (oldM === newM) return;
      const ok = new Set([9,25,49,81,121,169,225,289,361,441,529,625]);
      if (!ok.has(this.weights.length)) {
        const n = newM * newM, fan = Math.max(this.genes[6], 1), sc = Math.sqrt(2 / (fan + 1e-8));
        this.weights = new Float64Array(n);
        for (let i = 0; i < n; i++) this.weights[i] = randn() * sc;
        this.genes[4] = n; this.reach = ((Math.sqrt(n) | 0) - 1) >> 1; return;
      }
      const oldG = [];
      for (let r = 0; r < oldM; r++) {
        oldG[r] = [];
        for (let c = 0; c < oldM; c++) oldG[r][c] = this.weights[r * oldM + c];
      }
      const newG = [];
      for (let r = 0; r < newM; r++) newG[r] = new Array(newM).fill(0);
      const oc = oldM >> 1, nc = newM >> 1, cr = Math.min(oldM, newM);
      const so = oc - (cr >> 1), sn = nc - (cr >> 1);
      for (let i = 0; i < cr; i++) for (let j = 0; j < cr; j++)
        newG[sn + i][sn + j] = oldG[so + i][so + j];
      if (newM > oldM) {
        const fan = Math.max(this.genes[6], 1), sc = Math.sqrt(2 / (fan + 1e-8));
        for (let i = 0; i < newM; i++) for (let j = 0; j < newM; j++)
          if (newG[i][j] === 0) newG[i][j] = randn() * sc;
      }
      const n = newM * newM;
      this.weights = new Float64Array(n);
      for (let i = 0; i < newM; i++) for (let j = 0; j < newM; j++) this.weights[i * newM + j] = newG[i][j];
      this.genes[4] = n; this.reach = ((Math.sqrt(n) | 0) - 1) >> 1;
    }

    forward(cellsArr) {
      const cfg = CellConfig;
      const reach = (cfg && cfg.autonomous_network_genes) ? this.reach : cfg.length_of_dendrite;
      const up = this.getUpperLayerCells(cellsArr, reach);
      this.computeTotalCharge(up, reach);
      this.charge = this.relu(this.charge);
      this.updateCharge(this.charge, 'forward');
      if (this.backprops_remaining > 0) this.backprops_remaining--;
    }
    backward(cellsArr, lr, maxBelow, accumulateOnly) {
      const cfg = CellConfig;
      let lrn = (cfg && cfg.autonomous_network_genes) ? this.genes[9] : (lr != null ? lr : cfg.learning_rate);
      const reach = (cfg && cfg.autonomous_network_genes) ? this.reach : cfg.length_of_dendrite;
      const wd = (cfg && cfg.autonomous_network_genes) ? this.genes[8] : cfg.weight_decay;
      if (this.layer === cfg.num_layers - 2) {
        /* Last hidden layer: sum error from ALL output cells within reach,
           not just the one at the same (x,y). Fixes pruning bug where cells
           at positions without an output cell got error=EPS (zero gradient). */
        let es = EPS;
        const outZ = cfg.num_layers - 1;
        for (let dx = -reach; dx <= reach; dx++) for (let dy = -reach; dy <= reach; dy++) {
          const nx = this.x + dx, ny = this.y + dy;
          if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
            const t = cellsArr[nx][ny][outZ];
            if (t) es += (this.charge - t.charge) * this.reluDerivative(this.charge);
          }
        }
        this.error = clip(es, -10, 10);
      } else {
        const below = this.getLayerBelowCells(cellsArr, reach, maxBelow);
        this.computeErrorSignal(null, below, reach);
      }
      const up = this.getUpperLayerCells(cellsArr, reach);
      if (accumulateOnly) this.accumulateWeightGradients(up, reach);
      else this.updateWeightsAndBias(up, lrn, reach, wd);
    }
    shouldDie(chargePrune, gradPrune, weightMagPrune, contribPrune, pl) {
      /* Gene 14 immunity: newborn cells are protected until they've had enough backprop passes */
      if (this.backprops_remaining > 0) return false;
      const cfg = CellConfig;
      /* Gradient prune (O key): compare avg_gradient_magnitude against threshold */
      if (gradPrune) {
        const gt = (cfg && cfg.autonomous_network_genes) ? this.genes[10] : (cfg ? cfg.gradient_threshold : 1e-4);
        if (this.avg_gradient_magnitude <= gt) return true;
      }
      /* Charge prune (P key): rolling max_charge_diff vs charge_delta (gene 7) */
      if (chargePrune) {
        const cd = (cfg && cfg.autonomous_network_genes) ? this.genes[7] : (cfg ? cfg.charge_delta : this.genes[7]);
        const fwd = this.max_charge_diff_forward > cd;
        const rev = this.max_charge_diff_reverse > cd;
        if (pl === 'AND' && !(fwd && rev)) return true;
        if (pl === 'OR'  && !(fwd || rev)) return true;
      }
      /* Weight-magnitude pruning (Y key): gene 12 */
      if (weightMagPrune) {
        const wpt = (cfg && cfg.autonomous_network_genes) ? this.genes[12] : (cfg ? cfg.weight_prune_threshold : 0.01);
        if (wpt > 0) {
          let maxW = 0;
          for (let i = 0; i < this.weights.length; i++) { const aw = Math.abs(this.weights[i]); if (aw > maxW) maxW = aw; }
          if (maxW < wpt) return true;
        }
      }
      /* Contribution score pruning (Z key): gene 13 */
      if (contribPrune) {
        const mcs = (cfg && cfg.autonomous_network_genes) ? this.genes[13] : (cfg ? cfg.min_contribution_score : 0);
        if (mcs > 0 && this.contributionScore < mcs) return true;
      }
      return false;
    }
    shouldDieGenetic(nAlive, prot) {
      if (prot && (this.significant_charge_change_forward || this.significant_charge_change_reverse || this.significant_gradient_change)) return false;
      return nAlive <= this.genes[1] || nAlive >= this.genes[0];
    }
    validate() {
      const issues = [];
      for (let i = 0; i < this.weights.length; i++) if (Number.isNaN(this.weights[i])) {
        issues.push('NaN w'); const n = this.genes[4] | 0, fan = Math.max(this.genes[6], 1), sc = Math.sqrt(2 / (fan + 1e-8));
        this.weights = new Float64Array(n);
        for (let j = 0; j < n; j++) this.weights[j] = clip(randn() * sc, -1, 1);
        break;
      }
      if (Number.isNaN(this.charge)) { this.charge = 0; issues.push('NaN c'); }
      if (Number.isNaN(this.error)) { this.error = EPS; issues.push('NaN e'); }
      if (Number.isNaN(this.bias)) { this.bias = 0; issues.push('NaN b'); }
      const ex = ((Math.sqrt(this.genes[4]) | 0) - 1) >> 1;
      if (this.reach !== ex) { this.reach = ex; issues.push('reach'); }
      return issues;
    }
    toJSON() {
      return {
        x:this.x,y:this.y,layer:this.layer,genes:Array.from(this.genes),
        weights:Array.from(this.weights),bias:this.bias,charge:this.charge,error:this.error,gradient:this.gradient,reach:this.reach,
        forward_charges:this.forward_charges.slice(),reverse_charges:this.reverse_charges.slice(),
        max_charge_diff_forward:this.max_charge_diff_forward,max_charge_diff_reverse:this.max_charge_diff_reverse,
        significant_charge_change_forward:this.significant_charge_change_forward,significant_charge_change_reverse:this.significant_charge_change_reverse,
        gradient_history:this.gradient_history.slice(),avg_gradient_magnitude:this.avg_gradient_magnitude,significant_gradient_change:this.significant_gradient_change,
        contributionScore:this.contributionScore, backprops_remaining:this.backprops_remaining,
      };
    }
    static fromJSON(j) {
      const c = Object.create(Cell.prototype);
      c.x=j.x;c.y=j.y;c.layer=j.layer;c.genes=j.genes.slice();
      c.weights=Float64Array.from(j.weights);c.bias=j.bias;c.charge=j.charge;c.error=j.error;c.gradient=j.gradient;c.reach=j.reach;
      c.forward_charges=(j.forward_charges||[]).slice();c.reverse_charges=(j.reverse_charges||[]).slice();
      c.max_charge_diff_forward=j.max_charge_diff_forward||0;c.max_charge_diff_reverse=j.max_charge_diff_reverse||0;
      c.significant_charge_change_forward=!!j.significant_charge_change_forward;c.significant_charge_change_reverse=!!j.significant_charge_change_reverse;
      c.gradient_history=(j.gradient_history||[]).slice();c.avg_gradient_magnitude=j.avg_gradient_magnitude||0;c.significant_gradient_change=!!j.significant_gradient_change;
      c.contributionScore=j.contributionScore||0;
      c.backprops_remaining=j.backprops_remaining||0;
      /* backward-compat: old saves may have 12-14 genes, pad to 15 */
      while (c.genes.length < 14) c.genes.push(c.genes.length === 12 ? 0.01 : 0);
      if (c.genes.length < 15) c.genes.push(50);
      c.colors=new Array(15).fill(0);c.protein_colors=new Array(15).fill(0);c.colorGenes();c.colorProteins();
      return c;
    }
    clone() { return Cell.fromJSON(this.toJSON()); }
    toString() {
      let g911 = '';
      if (this.genes.length >= 12) g911 = `  LR=${this.genes[9].toFixed(4)}, GT=${this.genes[10].toExponential(2)}, AS=${this.genes[11].toFixed(2)}\n`;
      if (this.genes.length >= 15) g911 += `  WPT=${this.genes[12].toExponential(2)}, MCS=${this.genes[13].toExponential(2)}, IP=${this.genes[14]}\n`;
      const w = []; for (let i = 0; i < this.weights.length; i++) { if (i && i % 7 === 0) w.push('\n'); w.push(this.weights[i].toFixed(4)); }
      return `Neuron: layer=${this.layer} x=${this.x} y=${this.y}\n` +
        `Genes (breeding):\n  OT=${this.genes[0]}, IT=${this.genes[1]}, BT=${this.genes[2]}, MR=${this.genes[3]}\n` +
        `Genes (network):\n  WG=${this.genes[4]}, BR=${this.genes[5]}, AW=${this.genes[6]}, CD=${this.genes[7].toExponential(4)}, WD=${this.genes[8].toExponential(4)}\n` +
        g911 + `Proteins:\n  charge=${this.charge.toFixed(4)}, error=${this.error.toExponential(4)}, bias=${this.bias.toFixed(4)}, gradient=${this.gradient.toFixed(4)}\n` +
        `  reach=${this.reach}, avgGrad=${this.avg_gradient_magnitude.toExponential(2)}, contribScore=${this.contributionScore.toExponential(2)}\n` +
        `  chargeDiffFwd=${this.max_charge_diff_forward.toFixed(4)}, chargeDiffRev=${this.max_charge_diff_reverse.toFixed(4)}\n` +
        `weights=${w.join(', ')}`;
    }
  }

  function forwardPropagation(state, config) {
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (c) c.forward(state.cells);
      }
  }
  function reverseForwardPropagation(state, config) {
    for (let z = config.num_layers - 2; z > 0; z--)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (!c) continue;
        if (z === config.num_layers - 2) {
          const o = state.cells[x][y][config.num_layers - 1];
          if (o) c.updateCharge(o.charge, 'reverse');
        } else {
          const reach = (config.autonomous_network_genes) ? c.reach : config.length_of_dendrite;
          const low = c.getLayerBelowCells(state.cells, reach, null);
          c.computeTotalChargeReverse(low, reach);
          c.updateCharge(c.charge, 'reverse');
        }
      }
  }
  function backPropagation(state, config, renderBackpropFn, accumulateOnly) {
    for (let z = config.num_layers - 2; z > 0; z--) {
      let maxBelow = null;
      if (z < config.num_layers - 2) maxBelow = state.getMaxReachForLayer(z + 1, state.cells);
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z];
        if (c) {
          c.backward(state.cells, config.learning_rate, maxBelow, !!accumulateOnly);
          if (renderBackpropFn && state.show_3d_view && state.show_backprop_view) renderBackpropFn(z, x, y);
        }
      }
    }
  }
  function clearAllMinibatchAcc(state, config) {
    const nl = config.num_layers;
    for (let z = 1; z < nl - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z];
        if (c) c.clearMinibatchAccumulator();
      }
  }
  function applyAccumulatedGradients(state, config, batchDenom) {
    const nl = config.num_layers;
    for (let z = nl - 2; z > 0; z--) {
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z];
        if (!c) continue;
        const reach = config.autonomous_network_genes ? c.reach : config.length_of_dendrite;
        const lrn = config.autonomous_network_genes ? c.genes[9] : config.learning_rate;
        const wd = config.autonomous_network_genes ? c.genes[8] : config.weight_decay;
        const up = c.getUpperLayerCells(state.cells, reach);
        c.applyAccumulatedWeights(up, lrn, reach, wd, batchDenom);
      }
    }
  }
  /** Layers 1..num_layers-2: hidden stack + logits layer (same range as Nuke). If empty, guesses are not meaningful (stale bias / all-zero ties). */
  function hasAnyHiddenStackCell(state, config) {
    const nl = config.num_layers;
    for (let z = 1; z < nl - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++)
        if (state.cells[x][y][z]) return true;
    return false;
  }
  function predictionToActual(state, config) {
    const nl = config.num_layers, cells = state.cells;
    const guess = [], actual = [];
    for (let i = 9; i <= 18; i++) {
      const a = cells[i][14][nl - 2], b = cells[i][14][nl - 1];
      guess.push(a ? a.charge : 0); actual.push(b ? b.charge : 0);
    }
    let loss = 0;
    for (let i = 0; i < guess.length; i++) {
      const pc = clip(guess[i], 1e-7, 1 - 1e-7);
      loss -= actual[i] * Math.log(pc);
    }
    state.total_loss += loss; state.total_predictions += 1;
    state.running_avg_loss = state.total_loss / state.total_predictions;
    state.last_step_loss = loss;
    let pg = 0, pa = 0, mg = guess[0], ma = actual[0];
    for (let i = 1; i < guess.length; i++) { if (guess[i] > mg) { mg = guess[i]; pg = i; } if (actual[i] > ma) { ma = actual[i]; pa = i; } }
    /* Without at least one cell in the stack, do not increment Bingo/Max — avoids fake “accuracy” from always-guess-0 or leftover pixels. */
    /* Per-digit accuracy tracking */
    if (!state._epoch_digit_correct) { state._epoch_digit_correct = new Array(10).fill(0); state._epoch_digit_total = new Array(10).fill(0); }
    if (pa >= 0 && pa < 10) state._epoch_digit_total[pa]++;
    if (hasAnyHiddenStackCell(state, config) && pg === pa) {
      state.bingo_count++;
      if (state.bingo_count > state.max_bingo_count) state.max_bingo_count = state.bingo_count;
      if (pa >= 0 && pa < 10) state._epoch_digit_correct[pa]++;
    }
  }
  function trainNetwork(state, config, renderBackpropFn, useAccum, applyAccum, accumBatchLen) {
    for (let e = 0; e < state.epochs; e++) {
      if (state.direction_of_charge_flow === '+++++>>>>>') {
        forwardPropagation(state, config);
        if (state.back_prop) {
          backPropagation(state, config, renderBackpropFn, useAccum);
          if (useAccum && applyAccum) applyAccumulatedGradients(state, config, accumBatchLen);
        }
        predictionToActual(state, config);
      }
      if (state.direction_of_charge_flow === '<<<<<-----') reverseForwardPropagation(state, config);
    }
  }
  function trainOnSample(state, config, seqIndex, renderBackpropFn, minibatchPlotCtx) {
    /* seqIndex = position in epoch (0..setSize-1); dataIndex = actual sample after shuffle */
    const dataIndex = (state._shuffle_order && state._shuffle_order.length > seqIndex) ? state._shuffle_order[seqIndex] : seqIndex;
    if (dataIndex < 0 || dataIndex >= state.training_data_layer_0.length) {
      console.warn(`trainOnSample: dataIndex ${dataIndex} out of range (${state.training_data_layer_0.length} samples). Skipping.`);
      return;
    }
    applyLayerFromGrid(state.cells, 0, state.training_data_layer_0[dataIndex]);
    applyLayerFromGrid(state.cells, config.num_layers - 1, state.training_data_num_layer_minus_1[dataIndex]);
    state.total_weights = sumAbsWeights(state.cells, config.num_layers);
    const setSize = config.how_much_training_data;
    if (state.total_weights_list.length < setSize) state.total_weights_list = new Float64Array(setSize + 10);
    state.total_weights_list[seqIndex] = state.total_weights;
    const B = Math.max(1, config.gradient_minibatch_size | 0);
    const posInBatch = seqIndex % B;
    const isLastInMini = posInBatch === B - 1 || seqIndex === setSize - 1;
    if (posInBatch === 0) {
      state._mini_loss_sum = 0;
      state._mini_n = 0;
    }
    if (B > 1 && state.back_prop && posInBatch === 0) clearAllMinibatchAcc(state, config);
    const useAccum = B > 1 && state.back_prop;
    trainNetwork(state, config, renderBackpropFn, useAccum, useAccum && isLastInMini, posInBatch + 1);
    state._batch_loss_sum += state.last_step_loss;
    state._batch_sample_count++;
    state._mini_loss_sum += state.last_step_loss;
    state._mini_n++;
    if (B === 1 || isLastInMini) {
      const vMini = state._mini_loss_sum / Math.max(1, state._mini_n);
      predictionPlotMinibatch(state, minibatchPlotCtx, vMini);
      state._mini_loss_sum = 0;
      state._mini_n = 0;
    }
  }

  function training(state, config, drawFn, renderBackpropFn, minibatchPlotCtx) {
    state.bingo_count = 0;
    state._batch_loss_sum = 0;
    state._batch_sample_count = 0;
    const setSize = config.how_much_training_data;
    for (let i = 0; i < setSize; i++) {
      trainOnSample(state, config, i, renderBackpropFn, minibatchPlotCtx);
      if (state.display_updating && drawFn) drawFn();
    }
  }

  function applyLayerFromGrid(cells, layer, grid) {
    for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
      const g = grid[x][y];
      cells[x][y][layer] = g ? (g instanceof Cell ? g.clone() : Cell.fromJSON(g)) : null;
    }
  }
  function copyLayerGrid(grid) {
    const g = [];
    for (let x = 0; x < WIDTH; x++) {
      g[x] = [];
      for (let y = 0; y < HEIGHT; y++) {
        const c = grid[x][y];
        g[x][y] = c ? (c instanceof Cell ? c.clone() : Cell.fromJSON(c)) : null;
      }
    }
    return g;
  }

  function buildNeighborOffsets(z, numLayers) {
    const o = [];
    if (z === 1) {
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz of [0, 1]) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        o.push([dx, dy, dz]);
      }
    } else if (z === numLayers - 2) {
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz of [0, -1]) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        o.push([dx, dy, dz]);
      }
    } else {
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        o.push([dx, dy, dz]);
      }
    }
    return o;
  }

  function updateCells(state, config) {
    const start = 1, stop = config.num_layers - 1;
    const cells = state.cells;
    let gridTopoChanged = false;
    for (let z = start; z < stop; z++) {
      const neighOfs = buildNeighborOffsets(z, config.num_layers);
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        if (cells[x][y][z]) {
          const cell = cells[x][y][z];
          if (cell.shouldDie(state.prune, state.gradient_prune, state.weight_mag_prune, state.contrib_score_prune, state.prune_logic)) {
            cells[x][y][z] = null; gridTopoChanged = true; state.invalidateNeighborCache(); continue;
          }
        }
        if (cells[x][y][z]) {
          const cell = cells[x][y][z];
          if (Math.random() < cell.genes[3] / 100000) {
            if (Math.random() < 0.5) cell.initalizeBreedingGenes();
            else cell.initalizeNetworkGenes(config.number_of_weights, config.bias_range, config.avg_weights_cell,
              config.charge_delta, config.weight_decay, config.mutation_rate, cells);
          }
        }
        if (state.andromida_mode) {
          const alive = [];
          for (let k = 0; k < neighOfs.length; k++) {
            const [dx, dy, dz] = neighOfs[k];
            const nx = x + dx, ny = y + dy, nz = z + dz;
            if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT && nz >= 0 && nz < ARRAY_LAYERS && cells[nx][ny][nz])
              alive.push(cells[nx][ny][nz]);
          }
          const numAlive = alive.length;
          if (cells[x][y][z] == null && alive.length) {
            let newGenes = null;
            if (alive.length >= 2) {
              const p1 = alive[(Math.random() * alive.length) | 0];
              const p2 = alive[(Math.random() * alive.length) | 0];
              newGenes = [];
              for (let g = 0; g < p1.genes.length; g++) newGenes.push(Math.random() < 0.5 ? p1.genes[g] : p2.genes[g]);
            } else {
              newGenes = alive[0].genes.slice();
            }
            if (numAlive === newGenes[2]) {
              cells[x][y][z] = new Cell(z, x, y, config.number_of_weights, config.bias_range, config.avg_weights_cell,
                config.charge_delta, config.weight_decay, config.mutation_rate, newGenes);
              gridTopoChanged = true; state.invalidateNeighborCache();
              const nc = cells[x][y][z];
              if (Math.random() < nc.genes[3] / 1000) nc.initalizeBreedingGenes();
              if (Math.random() < nc.genes[3] / 1000)
                nc.initalizeNetworkGenes(config.number_of_weights, config.bias_range, config.avg_weights_cell,
                  config.charge_delta, config.weight_decay, config.mutation_rate, cells);
            }
          }
          if (cells[x][y][z]) {
            const cell = cells[x][y][z];
            if (cell.shouldDieGenetic(numAlive, state.charge_change_protection)) {
              cells[x][y][z] = null; gridTopoChanged = true; state.invalidateNeighborCache();
            }
          }
        }
      }
    }
    if (gridTopoChanged) state._3d_dirty = true;
  }

  /* Percentile pruning: rank all hidden cells by contributionScore, kill bottom N%.
     Called at epoch boundaries when config.prune_percentile > 0 and pruning is active. */
  function percentilePrune(state, config) {
    const pct = config.prune_percentile;
    if (!pct || pct <= 0) return 0;
    const nl = config.num_layers;
    const scored = [];
    for (let z = 1; z < nl - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z];
        if (c && c.backprops_remaining <= 0) scored.push({ x, y, z, score: c.contributionScore });
      }
    if (scored.length === 0) return 0;
    scored.sort((a, b) => a.score - b.score);
    const cutIdx = Math.max(1, Math.floor(scored.length * pct / 100));
    const cutoff = scored[Math.min(cutIdx, scored.length - 1)].score;
    let killed = 0;
    for (let i = 0; i < cutIdx; i++) {
      const s = scored[i];
      state.cells[s.x][s.y][s.z] = null;
      killed++;
    }
    if (killed > 0) { state.invalidateNeighborCache(); state._3d_dirty = true; }
    return killed;
  }

  function resetAllGradientChanges(state, config) {
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z];
        if (c) { c.resetGradientChange(); c.significant_charge_change_forward = false; c.significant_charge_change_reverse = false; }
      }
  }
  function resetDirectionalChargeHistoryAll(state, config, dir) {
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (c) c.resetDirectionalChargeHistory(dir);
      }
  }

  function computeTelemetry(state, config) {
    const telem = [];
    for (let z = 1; z < config.num_layers - 1; z++) {
      const t = { layer_id: z, num_cells: 0, num_active: 0, avg_charge: 0, avg_error: 0, avg_gradient: 0,
        max_gradient: 0, min_gradient: 0, avg_weight_magnitude: 0, avg_weights_per_cell: 0, nan_count: 0 };
      const layerCells = [];
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (c) layerCells.push(c);
      }
      t.num_cells = layerCells.length;
      const active = layerCells.filter(c => c.weights.length > 0 && c.weights.some(w => Math.abs(w) > config.epsilon));
      t.num_active = active.length;
      let nan = 0;
      for (let i = 0; i < layerCells.length; i++) nan += layerCells[i].validate().length;
      t.nan_count = nan;
      if (active.length) {
        const grads = active.map(c => c.gradient);
        t.avg_gradient = mean(grads); t.max_gradient = Math.max(...grads); t.min_gradient = Math.min(...grads);
        t.avg_error = mean(active.map(c => Math.abs(c.error)));
        t.avg_charge = mean(active.map(c => Math.abs(c.charge)));
        t.avg_weight_magnitude = mean(active.map(c => {
          let s = 0; for (let j = 0; j < c.weights.length; j++) s += Math.abs(c.weights[j]); return s / c.weights.length;
        }));
        t.avg_weights_per_cell = mean(active.map(c => c.weights.length));
      }
      telem.push(t);
    }
    return telem;
  }
  function formatTelemetry(telem) {
    const lines = ['=== Layer Telemetry ==='];
    let total = 0;
    for (let i = 0; i < telem.length; i++) {
      const t = telem[i];
      lines.push(`Layer ${t.layer_id}: ${t.num_active}/${t.num_cells} active`);
      lines.push(`  Charge: ${t.avg_charge.toFixed(4)} | Error: ${t.avg_error.toExponential(4)}`);
      lines.push(`  Gradient: avg=${t.avg_gradient.toExponential(4)} max=${t.max_gradient.toExponential(4)} min=${t.min_gradient.toExponential(4)}`);
      lines.push(`  Weights: mag=${t.avg_weight_magnitude.toFixed(4)} per_cell=${t.avg_weights_per_cell.toFixed(1)}`);
      if (t.nan_count) lines.push(`  *** NaN issues: ${t.nan_count} ***`);
      total += t.nan_count;
    }
    lines.push(total ? `\n*** TOTAL NaN ISSUES: ${total} ***` : '\nNo NaN issues detected.');
    lines.push('');
    lines.push('── Key ───────────────────────────────────');
    lines.push('Active  = cells with at least one |w| > ε');
    lines.push('Charge  = avg |charge| of active cells');
    lines.push('Error   = avg |error| of active cells');
    lines.push('Gradient= signed gradient (avg/max/min)');
    lines.push('Weights = avg |weight| per active cell');
    lines.push('NaN     = numeric overflow/corruption');
    return lines.join('\n');
  }

  function drawGrid(ctx) {
    ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
    for (let x = 0; x <= WINDOW_WIDTH; x += WINDOW_WIDTH / 4) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, WINDOW_HEIGHT); ctx.stroke();
    }
    for (let y = 0; y <= WINDOW_HEIGHT; y += WINDOW_HEIGHT / 4) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(WINDOW_WIDTH, y); ctx.stroke();
    }
  }
  function drawCells(state, config, ctx) {
    const disp = state.display;
    for (let z = 0; z < config.num_layers; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const cell = state.cells[x][y][z]; if (!cell) continue;
        if (disp === 'genes') cell.colorGenes(); else cell.colorProteins();
        const px = (i) => x * CELL_SIZE + (i % 3) * (CELL_SIZE / 3) + (z % 4) * (WINDOW_WIDTH / 4);
        const py = (i) => y * CELL_SIZE + ((i / 3) | 0) % 3 * (CELL_SIZE / 3) + ((z / 4) | 0) * (WINDOW_HEIGHT / 4);
        const pal = disp === 'genes' ? cell.colors : cell.protein_colors;
        const cs = CELL_SIZE / 3;
        for (let i = 0; i < 9; i++) {
          const col = pal[i];
          if (!col || col === 0) continue;
          const [r, g, b] = col;
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(px(i), py(i), cs, cs);
        }
      }
  }

  function updateCellTypes(cells, config) {
    const types = Object.create(null);
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = cells[x][y][z]; if (!c) continue;
        const key = JSON.stringify(c.genes.slice(0, 4));
        types[key] = (types[key] || 0) + 1;
      }
    return types;
  }
  function updatePhenotypeCellTypes(cells, config) {
    let countPos = 0, total = 0;
    const charges = [], biases = [], errors = [], wavg = [], grads = [], mf = [], mr = [];
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = cells[x][y][z]; if (!c) continue;
        total++;
        if (c.weights.every(w => w === 0 || w === config.epsilon)) continue;
        countPos++;
        charges.push(c.charge); biases.push(c.bias); errors.push(c.error);
        let s = 0; for (let i = 0; i < c.weights.length; i++) s += c.weights[i];
        wavg.push(s / c.weights.length); grads.push(c.gradient);
        mf.push(c.max_charge_diff_forward); mr.push(c.max_charge_diff_reverse);
      }
    if (!countPos) return [0, total, {}];
    const cm = mean(charges), cs = Math.sqrt(mean(charges.map(v => (v - cm) * (v - cm)))) || 1e-8;
    const bm = mean(biases), bs = Math.sqrt(mean(biases.map(v => (v - bm) * (v - bm)))) || 1e-8;
    const em = mean(errors), es = Math.sqrt(mean(errors.map(v => (v - em) * (v - em)))) || 1e-8;
    const wm = mean(wavg), ws = Math.sqrt(mean(wavg.map(v => (v - wm) * (v - wm)))) || 1e-8;
    const gm = mean(grads), gs = Math.sqrt(mean(grads.map(v => (v - gm) * (v - gm)))) || 1e-8;
    const fm = mean(mf), fs = Math.sqrt(mean(mf.map(v => (v - fm) * (v - fm)))) || 1e-8;
    const rm = mean(mr), rs = Math.sqrt(mean(mr.map(v => (v - rm) * (v - rm)))) || 1e-8;
    const dict = Object.create(null);
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = cells[x][y][z]; if (!c) continue;
        let s = 0; for (let i = 0; i < c.weights.length; i++) s += c.weights[i];
        const wa = s / c.weights.length;
        const ph = [
          'charge:' + (c.charge > cm ? '+' : '-') + (Math.abs(c.charge - cm) / cs | 0),
          'bias:' + (c.bias > bm ? '+' : '-') + (Math.abs(c.bias - bm) / bs | 0),
          'error:' + (c.error > em ? '+' : '-') + (Math.abs(c.error - em) / es | 0),
          'weights:' + (wa > wm ? '+' : '-') + (Math.abs(wa - wm) / ws | 0),
          'gradient:' + (c.gradient > gm ? '+' : '-') + (Math.abs(c.gradient - gm) / gs | 0),
          'max_fwd:' + (c.max_charge_diff_forward > fm ? '+' : '-') + (Math.abs(c.max_charge_diff_forward - fm) / fs | 0),
          'max_rev:' + (c.max_charge_diff_reverse > rm ? '+' : '-') + (Math.abs(c.max_charge_diff_reverse - rm) / rs | 0),
        ];
        const key = ph.join('|');
        if (!dict[key]) dict[key] = [];
        dict[key].push(c);
      }
    return [countPos, total, dict];
  }
  function formatStatistics(cellTypes) {
    const sorted = Object.entries(cellTypes).sort((a, b) => b[1] - a[1]).slice(0, 5);
    let mf = 0; for (const k of Object.keys(cellTypes)) mf += JSON.parse(k)[3] | 0;
    const avg = Object.keys(cellTypes).length ? mf / Object.keys(cellTypes).length : 0;
    const lines = ['Top 5 Cell Types:', '[OT IT BT] CH WG ER BI: Overcrowding Isolation Birth'];
    for (const [ct, cnt] of sorted) lines.push(`Type: ${ct} | Count: ${cnt}`);
    lines.push(`Average Mutation Frequency: ${avg.toFixed(2)}`);
    return lines.join('\n');
  }
  function formatPhenotypeStatistics(phen) {
    const [_, __, dict] = phen;
    const rows = Object.entries(dict).map(([ph, cl]) => {
      const ac = mean(cl.map(c => c.charge)), ab = mean(cl.map(c => c.bias)), ae = mean(cl.map(c => c.error));
      const aw = mean(cl.map(c => { let s = 0; for (let i = 0; i < c.weights.length; i++) s += c.weights[i]; return s / c.weights.length; }));
      const ag = mean(cl.map(c => c.gradient)), amf = mean(cl.map(c => c.max_charge_diff_forward)), amr = mean(cl.map(c => c.max_charge_diff_reverse));
      return [ph, cl.length, ac, ab, ae, aw, ag, amf, amr];
    }).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const lines = ['Top 5 Phenotypes:', 'Phenotype: Charge | Bias | Error | Weights | Gradient | MaxFwd | MaxRev'];
    for (const [ph, cnt, ac, ab, ae, aw, ag, amf, amr] of rows)
      lines.push(`${ph} (${cnt})\n  Ch:${ac.toFixed(4)} Bi:${ab.toFixed(4)} Er:${ae.toExponential(4)} W:${aw.toFixed(4)} G:${ag.toExponential(4)} Fwd:${amf.toFixed(4)} Rev:${amr.toFixed(4)}`);
    return lines.join('\n');
  }
  function formatMaxChargeDiff(state, config, N) {
    const diffs = [];
    for (let z = 1; z < config.num_layers - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (!c) continue;
        const cd = state.direction_of_charge_flow === '+++++>>>>>' ? c.max_charge_diff_forward : c.max_charge_diff_reverse;
        diffs.push([x, y, z, cd]);
      }
    diffs.sort((a, b) => b[3] - a[3]);
    const lines = ['Max Charge Diff (top 5):'];
    lines.push(...diffs.slice(0, N).map(([x, y, l, cd]) => `Cell (${x},${y},${l}): ${cd.toFixed(2)}`));
    return lines.join('\n');
  }
  function formatAverages(state, config) {
    const nl = config.num_layers, eps = config.epsilon;
    const avgG = new Float64Array(nl), maxG = new Float64Array(nl), minG = new Float64Array(nl);
    const avgE = new Float64Array(nl), avgC = new Float64Array(nl), avgW = new Float64Array(nl), avgWpc = new Float64Array(nl);
    let tcells = 0, activeCells = 0;
    for (let z = 1; z < nl - 1; z++) {
      const inLayer = [];
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) { const c = state.cells[x][y][z]; if (c) inLayer.push(c); }
      tcells += inLayer.length;
      const active = inLayer.filter(c => c.weights.some(w => w !== 0 && w !== eps));
      activeCells += active.length;
      if (active.length) {
        const grads = active.map(c => c.gradient);
        avgG[z] = mean(grads); maxG[z] = Math.max(...grads); minG[z] = Math.min(...grads);
        avgE[z] = mean(active.map(c => Math.abs(c.error)));
        avgC[z] = mean(active.map(c => Math.abs(c.charge)));
        avgW[z] = mean(active.map(c => { let s = 0; for (let i = 0; i < c.weights.length; i++) s += Math.abs(c.weights[i]); return s / c.weights.length; }));
        avgWpc[z] = mean(active.map(c => c.weights.length));
      }
    }
    const out = [];
    out.push(`═══ Network Averages ═══`);
    out.push(`Predictions: ${state.total_predictions} | Avg Loss: ${state.running_avg_loss.toFixed(2)}`);
    out.push(`Active Cells: ${activeCells}/${tcells}`);
    out.push('');
    for (let z = 1; z < nl - 1; z++) {
      out.push(`Layer ${z}:`);
      out.push(`  Grad: avg=${avgG[z].toExponential(2)} max=${maxG[z].toExponential(2)} min=${minG[z].toExponential(2)}`);
      out.push(`  |Error|: ${avgE[z].toExponential(2)}  |Charge|: ${avgC[z].toFixed(4)}`);
      out.push(`  |Weight|: ${avgW[z].toFixed(4)}  Wts/Cell: ${avgWpc[z].toFixed(0)}`);
    }
    out.push('');
    out.push('── Key ───────────────────────────────────');
    out.push('Grad     = raw gradient (signed, not abs)¹');
    out.push('|Error|  = avg absolute error per active cell¹');
    out.push('|Charge| = avg absolute charge per active cell¹');
    out.push('|Weight| = avg absolute weight per active cell');
    out.push('Wts/Cell = avg weight array size per cell');
    out.push('Active   = cells with at least one |w| > ε');
    out.push('');
    out.push('¹ These are instantaneous per-cell values from');
    out.push('  the last sample processed. Pruning uses');
    out.push('  rolling epoch-window averages instead');
    out.push('  (see V screen 0 for pruning readiness).');
    return out.join('\n');
  }
  /* Plot arrays now store raw loss values (not pre-baked pixel Y). Rendering maps loss→pixel using current ymax (slider). */
  const PLOT_MAX_POINTS = 800;
  function pushLossValue(arr, maxPts, val) {
    arr.push(val);
    if (arr.length > maxPts) arr.shift();
  }
  function renderLossStrip(ctx, arr, ymax, margin, strokeStyle, dotFill, label) {
    const W = ctx.canvas.width, H = ctx.canvas.height;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
    const n = arr.length;
    /* Y-axis label + latest value */
    ctx.fillStyle = '#888'; ctx.font = '10px monospace';
    const latest = n > 0 ? arr[n - 1].toFixed(2) : '--';
    ctx.fillText(label + '  Y:0-' + ymax.toFixed(0) + '  last=' + latest, margin + 2, 10);
    if (n === 0) return;
    const usable = H - 2 * margin;
    const xSpan = Math.max(1, W - 2 * margin);
    /* Fixed X step based on max window size — gives constant spacing and scrolling behavior */
    const step = xSpan / Math.max(1, PLOT_MAX_POINTS - 1);
    function toY(v) { return H - margin - (clip(v, 0, ymax) / ymax) * Math.max(1, usable); }
    ctx.strokeStyle = strokeStyle; ctx.lineWidth = 1.5; ctx.fillStyle = dotFill;
    if (n === 1) {
      ctx.beginPath(); ctx.arc(margin, toY(arr[0]), 2, 0, Math.PI * 2); ctx.fill();
      return;
    }
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const xd = margin + i * step, yd = toY(arr[i]);
      if (i === 0) ctx.moveTo(xd, yd); else ctx.lineTo(xd, yd);
    }
    ctx.stroke();
    for (let i = 0; i < n; i++) {
      ctx.beginPath(); ctx.arc(margin + i * step, toY(arr[i]), 2, 0, Math.PI * 2); ctx.fill();
    }
  }
  /** Bottom strip: mean loss over one gradient minibatch (K images, key K), one point per weight update. */
  function predictionPlotMinibatch(state, predCtx, meanLossInMinibatch) {
    if (!predCtx || !Number.isFinite(meanLossInMinibatch)) return;
    pushLossValue(state.minibatchLossPoints, PLOT_MAX_POINTS, meanLossInMinibatch);
    renderLossStrip(predCtx, state.minibatchLossPoints, state.plotMinibatchYmax, 2, '#2266ff', '#00f', 'Minibatch loss');
  }
  /** Top strip: mean loss over one full epoch (one pass through all loaded training samples). */
  function predictionPlotEpoch(state, predCtx, epochMeanLoss) {
    if (!predCtx || !Number.isFinite(epochMeanLoss)) return;
    pushLossValue(state.epochLossPoints, PLOT_MAX_POINTS, epochMeanLoss);
    renderLossStrip(predCtx, state.epochLossPoints, state.plotEpochYmax, 2, '#44dd88', '#0a6', 'Epoch loss');
  }
  function updateTrainingStats(state, config) {
    const eps = config.epsilon, nl = config.num_layers;
    const reach = config.length_of_dendrite;
    /* Per-layer detailed stats */
    const layers = [];
    let totalCells = 0, totalActive = 0, totalFanIn = 0, totalDeadNeurons = 0;
    let totalWeightSlots = 0, totalNonzeroWeights = 0, totalWeightMag = 0;
    for (let z = 1; z < nl - 1; z++) {
      const lc = [];
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (c) lc.push(c);
      }
      const nCells = lc.length;
      totalCells += nCells;
      const grads = [], charges = [], errors = [];
      let layerFanIn = 0, layerDead = 0, layerNzW = 0, layerWSlots = 0, layerWMag = 0;
      for (let k = 0; k < lc.length; k++) {
        const c = lc[k];
        grads.push(Math.abs(c.gradient));
        charges.push(c.charge);
        errors.push(Math.abs(c.error));
        /* Effective fan-in: count real neighbor cells */
        const r = config.autonomous_network_genes ? c.reach : reach;
        let fan = 0;
        for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) {
          const nx = c.x + dx, ny = c.y + dy;
          if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT && state.cells[nx][ny][z - 1]) fan++;
        }
        layerFanIn += fan;
        /* Dead neuron: charge essentially zero */
        if (Math.abs(c.charge) < 0.001) layerDead++;
        /* Weight utilization */
        layerWSlots += c.weights.length;
        for (let i = 0; i < c.weights.length; i++) {
          const aw = Math.abs(c.weights[i]);
          layerWMag += aw;
          if (aw > 0.01) layerNzW++;
        }
        if (c.weights.some(w => Math.abs(w) > eps)) totalActive++;
      }
      totalFanIn += layerFanIn; totalDeadNeurons += layerDead;
      totalWeightSlots += layerWSlots; totalNonzeroWeights += layerNzW; totalWeightMag += layerWMag;
      charges.sort((a, b) => a - b);
      const p = (arr, frac) => arr.length ? arr[Math.min(arr.length - 1, (frac * arr.length) | 0)] : 0;
      layers.push({
        z, nCells,
        avgGrad: mean(grads), maxGrad: grads.length ? Math.max(...grads) : 0,
        avgErr: mean(errors),
        chMin: charges.length ? charges[0] : 0, chP25: p(charges, 0.25), chMed: p(charges, 0.5), chP75: p(charges, 0.75), chMax: charges.length ? charges[charges.length - 1] : 0,
        avgFanIn: nCells ? layerFanIn / nCells : 0,
        deadCount: layerDead,
        weightUtil: layerWSlots ? (layerNzW / layerWSlots * 100) : 0,
        avgWeightMag: layerWSlots ? layerWMag / layerWSlots : 0,
      });
    }
    /* Per-digit accuracy */
    const digitCorrect = new Array(10).fill(0), digitTotal = new Array(10).fill(0);
    if (state._epoch_digit_correct) {
      for (let d = 0; d < 10; d++) { digitCorrect[d] = state._epoch_digit_correct[d] || 0; digitTotal[d] = state._epoch_digit_total[d] || 0; }
    }
    /* Gradient flow ratio: max grad layer 1 vs last hidden */
    const gfFirst = layers.length ? layers[0].maxGrad : 0;
    const gfLast = layers.length > 1 ? layers[layers.length - 1].maxGrad : gfFirst;
    const gradFlowRatio = gfLast > eps ? gfFirst / gfLast : 0;

    state.training_stats_buffer = {
      layers, totalCells, totalActive, totalDeadNeurons,
      avgFanIn: totalCells ? totalFanIn / totalCells : 0,
      weightUtil: totalWeightSlots ? (totalNonzeroWeights / totalWeightSlots * 100) : 0,
      avgWeightMag: totalWeightSlots ? totalWeightMag / totalWeightSlots : 0,
      gradFlowRatio, digitCorrect, digitTotal,
    };
  }
  function formatTrainingStats(state) {
    const b = state.training_stats_buffer;
    if (!b || !b.layers) return '';
    const s = [];
    s.push(`═══ Training Stats (Epoch ${state.training_cycles}) ═══`);
    s.push(`Accuracy: ${state.bingo_count}/${CellConfig.how_much_training_data} (Max: ${state.max_bingo_count})`);
    const lastEL = state.epochLossPoints.length ? state.epochLossPoints[state.epochLossPoints.length - 1].toFixed(2) : '--';
    s.push(`Epoch Loss: ${lastEL}  |  Cells: ${b.totalCells} (${b.totalActive} active)`);
    s.push(`Avg Fan-In: ${b.avgFanIn.toFixed(1)}  |  Silent¹: ${b.totalDeadNeurons}`);
    s.push(`Weight Util: ${b.weightUtil.toFixed(1)}%  |  Avg|W|: ${b.avgWeightMag.toFixed(4)}`);
    s.push(`Grad Flow (L1/Llast): ${b.gradFlowRatio.toFixed(2)}`);
    s.push('');
    /* Per-digit accuracy — epoch-level */
    const da = [];
    for (let d = 0; d < 10; d++) {
      const t = b.digitTotal[d], c = b.digitCorrect[d];
      da.push(`${d}:${c}/${t}`);
    }
    s.push('Per-Digit (epoch): ' + da.join(' '));
    s.push('');
    /* Table 1: Structure — narrow columns that fit ~50 chars */
    s.push('── Structure ──────────────────────');
    s.push('Lyr  Cells  FanIn  Slnt  W%   |W|');
    s.push('───  ─────  ─────  ────  ───  ────');
    for (const L of b.layers) {
      s.push(
        ` ${String(L.z).padStart(2)}  ${String(L.nCells).padStart(5)}  ${L.avgFanIn.toFixed(1).padStart(5)}` +
        `  ${String(L.deadCount).padStart(4)}  ${L.weightUtil.toFixed(0).padStart(3)}  ${L.avgWeightMag.toFixed(3)}`
      );
    }
    s.push('');
    /* Table 2: Dynamics — gradient and charge distribution */
    s.push('── Dynamics ──────────────────────────────');
    s.push('Lyr  Grad(avg)   Grad(max)   Chg(min/med/max)');
    s.push('───  ─────────   ─────────   ────────────────');
    for (const L of b.layers) {
      s.push(
        ` ${String(L.z).padStart(2)}  ${L.avgGrad.toExponential(1).padStart(9)}` +
        `   ${L.maxGrad.toExponential(1).padStart(9)}` +
        `   ${L.chMin.toFixed(2)}/${L.chMed.toFixed(2)}/${L.chMax.toFixed(2)}`
      );
    }
    s.push('');
    s.push('── Key ───────────────────────────────────');
    s.push('FanIn    = avg connected upstream cells per neuron');
    s.push('Silent   = neurons with |charge| < 0.001 (not firing)¹');
    s.push('           NOTE: "silent" ≠ "pruned". Silent cells are');
    s.push('           still alive. Pruning (V screen 0) uses');
    s.push('           different thresholds (charge_diff, gradient).');
    s.push('W%       = % of weight slots with |w| > 0.01');
    s.push('|W|      = mean absolute weight across all slots');
    s.push('Grad     = |gradient| from last sample in epoch²');
    s.push('Chg      = charge distribution from last sample in epoch²');
    s.push('GradFlow = max grad layer 1 / max grad last hidden');
    s.push('Per-Digit= correct/total per digit class (full epoch)');
    s.push('');
    s.push('── When measured ─────────────────────────');
    s.push('¹ Silent: charge snapshot at end of epoch (from');
    s.push('  last sample). A cell may fire for other images');
    s.push('  but read 0 on the last one — low count is normal.');
    s.push('² Grad/Chg: snapshot from last sample in epoch.');
    s.push('');
    s.push('── How pruning differs from these stats ──');
    s.push('Pruning does NOT use single-sample snapshots.');
    s.push('Integrative memory proteins accumulate rolling-window metrics:');
    s.push(' • max_charge_diff = max−min charge over last');
    s.push('   epoch-worth of samples (window = training set)');
    s.push(' • avg_gradient_magnitude = mean |grad| over');
    s.push('   last epoch-worth of samples');
    s.push(' • contributionScore = charge_diff × avg_grad');
    s.push('These rolling values are what pruning checks.');
    s.push('See V screen 0 for pruning readiness details.');
    return s.join('\n');
  }

  function getAllSettings(state, config) {
    const sug = suggestParams(state, config);
    const vs = (cur, sugVal) => sug && sugVal !== undefined ? `${cur}  ← net: ${sugVal}` : `${cur}`;
    let out = `══════ CURRENT SETTINGS ══════
 Num Layers:         ${config.num_layers}
 Dendrite Length:     ${config.length_of_dendrite}  (matrix ${config.weight_matrix}×${config.weight_matrix} = ${config.number_of_weights} weights)
 Mutation Rate:       ${config.mutation_rate}  (per 10k cycles)
 Allele Range:        ${config.lower_allele_range} – ${config.upper_allele_range}
 Weight Chg Thresh:   ${config.weight_change_threshold}

══════ KEY LEARNING PARAMS ══════  (← net = measured suggestion)
 Learning Rate:       ${vs(config.learning_rate, sug ? sug.lr : undefined)}
 Bias Range:          ${vs(config.bias_range, sug ? sug.bias_range : undefined)}
 Avg Weights/Cell:    ${vs(config.avg_weights_cell, sug ? sug.avg_weights_cell : undefined)}
 Weight Decay:        ${vs(config.weight_decay, sug ? sug.weight_decay : undefined)}
 Charge Delta:        ${vs(config.charge_delta, sug ? sug.charge_delta : undefined)}
 Gradient Threshold:  ${vs(config.gradient_threshold, sug ? sug.gradient_threshold : undefined)}
 Gradient Clip:       ${vs(config.gradient_clip_range, sug ? sug.gradient_clip_range : undefined)}
 Activation Slope:    ${config.activation_slope}
 Weight Prune Thresh: ${vs(config.weight_prune_threshold, sug ? sug.weight_prune_threshold : undefined)}  [gene 12]
 Min Contrib Score:   ${config.min_contribution_score}  (0=off, charge_diff×gradient)  [gene 13]
 Immune Period:       ${config.immune_period} backprops  (newborn protection)  [gene 14]
 Prune Percentile:    ${config.prune_percentile}%  (0=off, bottom N% killed each epoch)

══════ TRAINING CONFIG ══════
 Training Data Size:  ${config.how_much_training_data}
 Minibatch K:         ${config.gradient_minibatch_size}
 Shuffle Epochs:      ${config.shuffle_epoch}
 Start Index:         ${config.start_index}

══════ STATE ══════
 Display:             ${state.display}
 Charge Flow:         ${state.direction_of_charge_flow}
 Prune Logic:         ${state.prune_logic}
 Training:            ${state.training_mode}  Backprop: ${state.back_prop}
 P:Charge=${state.prune}  O:Grad=${state.gradient_prune}  Y:Weight=${state.weight_mag_prune}  Z:Contrib=${state.contrib_score_prune}`;
    if (sug) {
      out += `\n\n══════ NETWORK MEASUREMENTS ══════
 Cells:               ${sug.nCells}
 Avg Fan-in:          ${sug.avgFanIn.toFixed(1)}
 Median |gradient|:   ${sug.medGrad.toExponential(2)}
 Median |weight|:     ${sug.medWeight.toFixed(4)}
 Median |charge|:     ${sug.medCharge.toFixed(4)}
 Median |error|:      ${sug.medError.toExponential(2)}
 Median |bias|:       ${sug.medBias.toFixed(4)}`;
    }
    /* Pruning readiness: show how many cells would survive/die at current settings */
    const nl = config.num_layers;
    const csScores = [];
    let wouldDieCharge = 0, wouldDieGrad = 0, wouldDieWeight = 0, wouldDieCS = 0, totalH = 0, immuneCount = 0;
    for (let z = 1; z < nl - 1; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = state.cells[x][y][z]; if (!c) continue;
        totalH++;
        if (c.backprops_remaining > 0) immuneCount++;
        csScores.push(c.contributionScore);
        const cd = config.autonomous_network_genes ? c.genes[7] : config.charge_delta;
        const fwd = c.max_charge_diff_forward > cd;
        const rev = c.max_charge_diff_reverse > cd;
        if (state.prune_logic === 'AND' ? !(fwd && rev) : !(fwd || rev)) wouldDieCharge++;
        const gt = config.autonomous_network_genes ? c.genes[10] : config.gradient_threshold;
        if (c.avg_gradient_magnitude <= gt) wouldDieGrad++;
        let maxW = 0;
        for (let i = 0; i < c.weights.length; i++) { const aw = Math.abs(c.weights[i]); if (aw > maxW) maxW = aw; }
        const wpt = config.autonomous_network_genes ? c.genes[12] : config.weight_prune_threshold;
        if (maxW < wpt) wouldDieWeight++;
        const mcs = config.autonomous_network_genes ? c.genes[13] : config.min_contribution_score;
        if (mcs > 0 && c.contributionScore < mcs) wouldDieCS++;
      }
    if (totalH > 0) {
      csScores.sort((a, b) => a - b);
      const pct = (p) => csScores[Math.min(((p * csScores.length) | 0), csScores.length - 1)];
      out += `\n\n══════ PRUNING READINESS ══════  (${totalH} hidden cells)
 Would die P (charge):  ${wouldDieCharge}/${totalH}  (${state.prune_logic}) ${state.prune ? '[ON]' : '[off]'}
 Would die O (gradient): ${wouldDieGrad}/${totalH} ${state.gradient_prune ? '[ON]' : '[off]'}
 Would die Y (weight):  ${wouldDieWeight}/${totalH} ${state.weight_mag_prune ? '[ON]' : '[off]'}
 Would die Z (contrib):  ${wouldDieCS}/${totalH} ${state.contrib_score_prune ? '[ON]' : '[off]'}
 Immune (newborn):      ${immuneCount}/${totalH}  (protected by gene 14)`;
      if (config.prune_percentile > 0) {
        const cutN = Math.max(1, Math.floor(totalH * config.prune_percentile / 100));
        out += `\n Percentile prune: bottom ${config.prune_percentile}% = ${cutN} cells`;
      }
      out += `\n\n Contribution score distribution:
   p10=${pct(0.10).toExponential(2)}  p25=${pct(0.25).toExponential(2)}  p50=${pct(0.50).toExponential(2)}
   p75=${pct(0.75).toExponential(2)}  p90=${pct(0.90).toExponential(2)}  max=${pct(1.0).toExponential(2)}`;
    }
    out += `\n\n── Key ───────────────────────────────────
 ← net       = suggestion measured from live network
 [gene 12/13]= threshold lives inside each cell
 Fan-in      = avg connected upstream cells per neuron
 Would die   = cells that fail this test RIGHT NOW¹
 Contrib     = max(charge_diff) × avg|gradient|
 Percentile  = bottom N% killed by rank each epoch²

── How pruning metrics work ──────────────
¹ "Would die" previews integrative memory proteins.
  These proteins use ROLLING WINDOWS (not single
  samples):
   • charge_diff = max−min charge over last N
     samples (N = training set size = 1 epoch)
   • avg_gradient = mean |gradient| over last
     N samples
   • contributionScore = charge_diff × avg_grad
  P/O/Y/Z pruning checks these rolling values each
  evolution step. Cells only die after they've
  been consistently inactive across many samples.
  Newborn cells are IMMUNE for gene 14 training
  cycles (protein: backprops_remaining counts down
  each forward pass, not just backprop).
² Percentile pruning runs once at EPOCH BOUNDARY.
  Ranks all cells by contributionScore and kills
  the bottom N%. Set via C key.`;
    return out;
  }

  function serializeCells(cells) {
    const o = [];
    for (let x = 0; x < WIDTH; x++) {
      o[x] = [];
      for (let y = 0; y < HEIGHT; y++) {
        o[x][y] = [];
        for (let z = 0; z < ARRAY_LAYERS; z++) {
          const c = cells[x][y][z];
          o[x][y][z] = c ? c.toJSON() : null;
        }
      }
    }
    return o;
  }
  function deserializeCells(data, config) {
    Cell.setConfig(config);
    const cells = makeCellsGrid();
    for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) for (let z = 0; z < ARRAY_LAYERS; z++) {
      const j = data[x][y][z];
      cells[x][y][z] = j ? Cell.fromJSON(j) : null;
    }
    return cells;
  }
  function configToJSON(config) {
    return {
      num_layers: config.num_layers, length_of_dendrite: config.length_of_dendrite, weight_matrix: config.weight_matrix,
      number_of_weights: config.number_of_weights, mutation_rate: config.mutation_rate,
      lower_allele_range: config.lower_allele_range, upper_allele_range: config.upper_allele_range,
      autonomous_network_genes: config.autonomous_network_genes,
      learning_rate: config.learning_rate, bias_range: config.bias_range, avg_weights_cell: config.avg_weights_cell,
      weight_decay: config.weight_decay, charge_delta: config.charge_delta, gradient_threshold: config.gradient_threshold,
      gradient_clip_range: config.gradient_clip_range, weight_change_threshold: config.weight_change_threshold,
      activation_slope: config.activation_slope, how_much_training_data: config.how_much_training_data, start_index: config.start_index,
      gradient_minibatch_size: config.gradient_minibatch_size,
      shuffle_epoch: config.shuffle_epoch,
      weight_prune_threshold: config.weight_prune_threshold,
      min_contribution_score: config.min_contribution_score,
      prune_percentile: config.prune_percentile,
      immune_period: config.immune_period,
      epsilon: config.epsilon,
    };
  }
  function configFromJSON(j) {
    const c = Object.assign(setDefaultValues(), j);
    c.updateDerived();
    return c;
  }

  function buildSyntheticTraining(config) {
    Cell.setConfig(config);
    const out0 = [], outL = [];
    for (let x = 0; x < WIDTH; x++) {
      out0[x] = []; outL[x] = [];
      for (let y = 0; y < HEIGHT; y++) {
        out0[x][y] = new Cell(0, x, y, config.number_of_weights, config.bias_range, config.avg_weights_cell,
          config.charge_delta, config.weight_decay, config.mutation_rate, null);
        out0[x][y].charge = Math.random();
        outL[x][y] = new Cell(config.num_layers - 1, x, y, config.number_of_weights, config.bias_range, config.avg_weights_cell,
          config.charge_delta, config.weight_decay, config.mutation_rate, null);
        outL[x][y].charge = 0;
      }
    }
    const label = (Math.random() * 10) | 0;
    for (let i = 0; i < 10; i++) outL[9 + i][14].charge = (i === label) ? 1 : 0;
    return { layer0: copyLayerGrid(out0), layerLast: copyLayerGrid(outL) };
  }

  /** Map pointer to bitmap coords when CSS resizes the canvas (fixes wrong hit below fold). */
  function canvasPointerToInternal(canvas, clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / Math.max(r.width, 1e-6);
    const sy = canvas.height / Math.max(r.height, 1e-6);
    const mx = (clientX - r.left) * sx;
    const my = (clientY - r.top) * sy;
    return [clip(mx, 0, WINDOW_WIDTH - 1e-6), clip(my, 0, WINDOW_HEIGHT - 1e-6)];
  }

  function convertXYToIndex(mx, my) {
    const lx = (mx / (WINDOW_WIDTH / 4)) | 0;
    const ly = (my / (WINDOW_HEIGHT / 4)) | 0;
    const layer = lx + ly * 4;
    const ax = mx - lx * (WINDOW_WIDTH / 4);
    const ay = my - ly * (WINDOW_HEIGHT / 4);
    const cellX = Math.min((ax / CELL_SIZE) | 0, WIDTH - 1);
    const cellY = Math.min((ay / CELL_SIZE) | 0, HEIGHT - 1);
    return [cellX, cellY, layer];
  }

  // --- Three.js 3D cache ---
  let g3 = {
    scene: null, camera: null, renderer: null,
    root: null,
    ptsIn: null, ptsHid: null, ptsOut: null,
    lines: null,
    cellRefs: [],
    nIn: 0, nHid: 0, nOut: 0,
    backpropGroup: null,
  };

  function hsvToRgb(h, s, v) {
    if (s === 0) return [v, v, v];
    const i = (h * 6) | 0;
    const f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    const m = i % 6;
    if (m === 0) return [v, t, p];
    if (m === 1) return [q, v, p];
    if (m === 2) return [p, v, t];
    if (m === 3) return [p, q, v];
    if (m === 4) return [t, p, v];
    return [v, p, q];
  }
  function layerColor(layer, nl, charge) {
    const t = nl <= 1 ? 0.5 : layer / (nl - 1);
    const h = 0.6 * (1 - t);
    const s = 0.85, v = 0.5 + 0.5 * Math.min(Math.abs(charge), 1);
    return hsvToRgb(h, s, v);
  }

  const _3D_COLOR_MODE_NAMES = ['Charge', 'Error', 'Gradient', 'Weight Strength', 'Contribution'];
  const _3D_COLOR_MODE_COUNT = _3D_COLOR_MODE_NAMES.length;

  /* Mode 1: Error — blue (negative) → dark (zero) → red (positive) */
  function errorColor(cell) {
    if (!cell) return [0.1, 0.1, 0.1];
    const e = clip(cell.error, -2, 2);
    const mag = Math.abs(e) / 2;
    if (e > 0) return [0.3 + 0.7 * mag, 0.1 * (1 - mag), 0.1 * (1 - mag)];
    return [0.1 * (1 - mag), 0.1 * (1 - mag), 0.3 + 0.7 * mag];
  }

  /* Mode 2: Gradient flow — black (dead) → yellow (active learning) */
  function gradientColor(cell) {
    if (!cell) return [0.05, 0.05, 0.05];
    const g = Math.min(Math.abs(cell.gradient), 1);
    const v = Math.sqrt(g);
    return [v, v * 0.9, v * 0.1];
  }

  /* Mode 3: Weight strength — black (weak) → cyan (strong connections) */
  function weightStrengthColor(cell) {
    if (!cell || !cell.weights || !cell.weights.length) return [0.08, 0.08, 0.08];
    let sum = 0;
    for (let i = 0; i < cell.weights.length; i++) sum += Math.abs(cell.weights[i]);
    const avg = sum / cell.weights.length;
    const v = Math.min(avg * 4, 1);
    return [v * 0.15, v * 0.7 + 0.1, v * 0.9 + 0.1];
  }

  /* Mode 4: Contribution score — black (dead) → magenta (high contribution).
     Shows which cells are both active AND learning — the best pruning predictor. */
  function contributionColor(cell) {
    if (!cell) return [0.05, 0.05, 0.05];
    const s = cell.contributionScore;
    const v = Math.min(Math.sqrt(s * 1000), 1);
    return [0.3 + 0.7 * v, 0.1 * (1 - v), 0.3 + 0.5 * v];
  }

  /* Build a small round-disc texture so points render as circles, not squares */
  function _makeDiscTexture(sz) {
    const c = document.createElement('canvas');
    c.width = c.height = sz;
    const g = c.getContext('2d');
    const half = sz / 2;
    const grad = g.createRadialGradient(half, half, 0, half, half, half);
    grad.addColorStop(0.0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.7, 'rgba(255,255,255,1)');
    grad.addColorStop(1.0, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, sz, sz);
    const tex = new THREE.CanvasTexture(c);
    tex.needsUpdate = true;
    return tex;
  }

  function init3d(canvas) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 80);
    camera.position.set(0, 0, 22);
    camera.up.set(0, 1, 0);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    renderer.setClearColor(0x101018, 1);
    const root = new THREE.Group();
    scene.add(root);
    const discMap = _makeDiscTexture(64);
    const matIn = new THREE.PointsMaterial({ size: 0.35, map: discMap, vertexColors: true, sizeAttenuation: true, transparent: true, alphaTest: 0.05, depthWrite: false });
    const matH  = new THREE.PointsMaterial({ size: 0.45, map: discMap, vertexColors: true, sizeAttenuation: true, transparent: true, alphaTest: 0.05, depthWrite: false });
    const matO  = new THREE.PointsMaterial({ size: 0.7,  map: discMap, vertexColors: true, sizeAttenuation: true, transparent: true, alphaTest: 0.05, depthWrite: false });
    const ptsIn = new THREE.Points(new THREE.BufferGeometry(), matIn);
    const ptsHid = new THREE.Points(new THREE.BufferGeometry(), matH);
    const ptsOut = new THREE.Points(new THREE.BufferGeometry(), matO);
    root.add(ptsIn); root.add(ptsHid); root.add(ptsOut);
    const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 });
    const lines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMat);
    root.add(lines);
    const backpropGroup = new THREE.Group();
    root.add(backpropGroup);
    g3 = { scene, camera, renderer, root, ptsIn, ptsHid, ptsOut, lines, discMap, cellRefs: [], nIn: 0, nHid: 0, nOut: 0, backpropGroup };
  }

  function rebuild3dCache(state, config) {
    const cells = state.cells, nl = config.num_layers;
    let nIn = 0, nHid = 0, nOut = 0;
    for (let z = 0; z < nl; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        if (!cells[x][y][z]) continue;
        if (z === 0) nIn++; else if (z === nl - 1) nOut++; else nHid++;
      }
    const total = nIn + nHid + nOut;
    const pos = new Float32Array(Math.max(total, 1) * 3);
    const col = new Float32Array(Math.max(total, 1) * 3);
    const refs = [];
    const connPos = [];
    const connCol = [];
    const halfW = WIDTH / 2, halfH = HEIGHT / 2;
    let idxI = 0, idxH = nIn, idxO = nIn + nHid;

    function putPoint(idx, x, y, z, layer, c) {
      const px = x - halfW, py = y - halfH, pz = layer * 2 - nl;
      pos[idx * 3] = px; pos[idx * 3 + 1] = py; pos[idx * 3 + 2] = pz;
      const [r, g, b] = layerColor(layer, nl, c.charge);
      col[idx * 3] = r; col[idx * 3 + 1] = g; col[idx * 3 + 2] = b;
      refs[idx] = [x, y, layer];

      if (layer === 0) return;
      if (layer === nl - 1) {
        if (cells[x][y][layer - 1]) {
          const prevZ = (layer - 1) * 2 - nl;
          connPos.push(px, py, pz, px, py, prevZ);
          connCol.push(0.2, 0.9, 0.2, 0.2, 0.9, 0.2);
        }
        return;
      }
      const cell = c;
      const reach = config.autonomous_network_genes ? cell.reach : config.length_of_dendrite;
      const prevZ = (layer - 1) * 2 - nl;
      const wm = config.autonomous_network_genes ? (Math.sqrt(cell.genes[4]) | 0) : config.weight_matrix;
      for (let dx = -reach; dx <= reach; dx++) {
        const nx = x + dx; if (nx < 0 || nx >= WIDTH) continue;
        for (let dy = -reach; dy <= reach; dy++) {
          const ny = y + dy; if (ny < 0 || ny >= HEIGHT) continue;
          if (!cells[nx][ny][layer - 1]) continue;
          const wi = (dx + reach) * wm + (dy + reach);
          if (wi >= cell.weights.length) continue;
          const w = cell.weights[wi];
          if (Math.abs(w) < WEIGHT_DRAW_THRESHOLD) continue;
          const inten = Math.max(0.25, Math.min(Math.abs(w), 1));
          const npx = nx - halfW, npy = ny - halfH;
          connPos.push(px, py, pz, npx, npy, prevZ);
          if (w > 0) connCol.push(0, inten, 0, 0, inten, 0);
          else connCol.push(inten, 0, 0, inten, 0, 0);
        }
      }
    }

    for (let z = 0; z < nl; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = cells[x][y][z]; if (!c) continue;
        if (z === 0) putPoint(idxI++, x, y, z, z, c);
      }
    for (let z = 0; z < nl; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = cells[x][y][z]; if (!c) continue;
        if (z > 0 && z < nl - 1) putPoint(idxH++, x, y, z, z, c);
      }
    for (let z = 0; z < nl; z++)
      for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
        const c = cells[x][y][z]; if (!c) continue;
        if (z === nl - 1) putPoint(idxO++, x, y, z, z, c);
      }

    g3.nIn = nIn; g3.nHid = nHid; g3.nOut = nOut; g3.cellRefs = refs;

    function setGeom(pts, start, count, posArr, colArr) {
      const g = pts.geometry;
      const np = count * 3;
      const psub = new Float32Array(np);
      const csub = new Float32Array(np);
      for (let i = 0; i < np; i++) { psub[i] = posArr[start * 3 + i]; csub[i] = colArr[start * 3 + i]; }
      g.setAttribute('position', new THREE.BufferAttribute(psub, 3));
      g.setAttribute('color', new THREE.BufferAttribute(csub, 3));
      g.computeBoundingSphere();
      pts.geometry = g;
    }
    if (total === 0) {
      g3.ptsIn.geometry.dispose();
      g3.ptsHid.geometry.dispose();
      g3.ptsOut.geometry.dispose();
      g3.ptsIn.geometry = new THREE.BufferGeometry();
      g3.ptsHid.geometry = new THREE.BufferGeometry();
      g3.ptsOut.geometry = new THREE.BufferGeometry();
      g3.lines.geometry.dispose();
      g3.lines.geometry = new THREE.BufferGeometry();
    } else {
      setGeom(g3.ptsIn, 0, nIn, pos, col);
      setGeom(g3.ptsHid, nIn, nHid, pos, col);
      setGeom(g3.ptsOut, nIn + nHid, nOut, pos, col);
      const lg = new THREE.BufferGeometry();
      if (connPos.length) {
        lg.setAttribute('position', new THREE.Float32BufferAttribute(connPos, 3));
        lg.setAttribute('color', new THREE.Float32BufferAttribute(connCol, 3));
      }
      g3.lines.geometry.dispose();
      g3.lines.geometry = lg;
    }
    state._3d_dirty = false;
  }

  function _cellColorForMode(cell, layer, nl, mode) {
    if (!cell) return [0.1, 0.1, 0.1];
    switch (mode) {
      case 1: return errorColor(cell);
      case 2: return gradientColor(cell);
      case 3: return weightStrengthColor(cell);
      case 4: return contributionColor(cell);
      default: return layerColor(layer, nl, cell.charge);
    }
  }

  function refresh3dColors(state, config) {
    const refs = g3.cellRefs;
    if (!refs.length) return;
    const nl = config.num_layers;
    const mode = state._3d_color_mode;
    const geoms = [g3.ptsIn.geometry, g3.ptsHid.geometry, g3.ptsOut.geometry];
    const starts = [0, g3.nIn, g3.nIn + g3.nHid];
    const counts = [g3.nIn, g3.nHid, g3.nOut];
    for (let g = 0; g < 3; g++) {
      const attr = geoms[g].getAttribute('color');
      if (!attr) continue;
      const arr = attr.array;
      let off = 0;
      const base = starts[g];
      for (let i = 0; i < counts[g]; i++) {
        const ref = refs[base + i];
        if (!ref) continue;
        const [x, y, z] = ref;
        const cell = state.cells[x][y][z];
        const [r, gr, b] = _cellColorForMode(cell, z, nl, mode);
        arr[off] = r; arr[off + 1] = gr; arr[off + 2] = b; off += 3;
      }
      attr.needsUpdate = true;
    }
  }

  function apply3dCamera(state) {
    const root = g3.root;
    root.rotation.set(state.rotation_x * Math.PI / 180, state.rotation_y * Math.PI / 180, 0, 'YXZ');
    root.position.z = state.zoom * 0.4;
  }

  function dispose3dBackpropChildren() {
    const bg = g3.backpropGroup;
    while (bg.children.length) {
      const ch = bg.children[0];
      bg.remove(ch);
      if (ch.geometry) ch.geometry.dispose();
      if (ch.material) ch.material.dispose();
    }
  }

  function render3dNetwork(state, config) {
    if (state._3d_dirty) rebuild3dCache(state, config);
    refresh3dColors(state, config);
    apply3dCamera(state);
    dispose3dBackpropChildren();
    g3.renderer.render(g3.scene, g3.camera);
    const hud = document.getElementById('hud3d');
    hud.style.display = 'block';
    hud.textContent = [
      `Layers: ${config.num_layers} | Cells: ${g3.nIn + g3.nHid + g3.nOut} | Color: ${_3D_COLOR_MODE_NAMES[state._3d_color_mode]} (G=cycle)`,
      `Epoch: ${state.training_cycles} | Correct: ${state.bingo_count}/${config.how_much_training_data} | Max: ${state.max_bingo_count}`,
      `Loss: ${state.running_avg_loss.toFixed(2)} | LR: ${config.learning_rate} | Dendrite: ${config.length_of_dendrite}`,
      `Train: ${state.training_mode} | BackProp: ${state.back_prop} | Prune: P=${state.prune} O=${state.gradient_prune} Y=${state.weight_mag_prune} Z=${state.contrib_score_prune}`,
    ].join('\n');
  }

  function render3dBackprop(state, config, cz, cx, cy) {
    if (state._3d_dirty) rebuild3dCache(state, config);
    refresh3dColors(state, config);
    apply3dCamera(state);
    dispose3dBackpropChildren();
    const nl = config.num_layers, halfW = WIDTH / 2, halfH = HEIGHT / 2;
    const z = cz * 2 - nl;
    const px = cx - halfW, py = cy - halfH;
    const cell = state.cells[cx][cy][cz];
    if (!cell) { g3.renderer.render(g3.scene, g3.camera); return; }
    const addPoints = (pts, cols, size) => {
      if (!pts.length) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      const m = new THREE.PointsMaterial({ size: size * 0.06, map: g3.discMap, vertexColors: true, sizeAttenuation: true, transparent: true, alphaTest: 0.05, depthWrite: false });
      g3.backpropGroup.add(new THREE.Points(g, m));
    };
    const addLines = (seg, cols) => {
      if (!seg.length) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(cols || new Float32Array(seg.length / 3).fill(1), 3));
      const m = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 });
      g3.backpropGroup.add(new THREE.LineSegments(g, m));
    };
    const pHi = [], cHi = [];
    pHi.push(px, py, z); cHi.push(1, 1, 0);
    const reach = config.autonomous_network_genes ? cell.reach : config.length_of_dendrite;
    const wm = config.autonomous_network_genes ? (Math.sqrt(cell.genes[4]) | 0) : config.weight_matrix;
    const lineSegs = [], lineCols = [];
    if (cz > 0) {
      const prevZz = (cz - 1) * 2 - nl;
      for (let dx = -reach; dx <= reach; dx++) for (let dy = -reach; dy <= reach; dy++) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || nx >= WIDTH || ny < 0 || ny >= HEIGHT) continue;
        const u = state.cells[nx][ny][cz - 1]; if (!u) continue;
        const ci = Math.min(Math.abs(u.charge), 1);
        const npx = nx - halfW, npy = ny - halfH;
        pHi.push(npx, npy, prevZz);
        cHi.push(0.2, 0.5 + 0.5 * ci, 1);
        /* Draw dendrite line from current cell to this predecessor, colored by weight sign */
        const wi = (dx + reach) * wm + (dy + reach);
        if (wi < cell.weights.length) {
          const w = cell.weights[wi];
          const inten = Math.max(0.3, Math.min(Math.abs(w), 1));
          lineSegs.push(px, py, z, npx, npy, prevZz);
          if (w > 0) lineCols.push(0, inten, 0.4, 0, inten, 0.4);
          else lineCols.push(inten, 0, 0.4, inten, 0, 0.4);
        }
      }
    }
    addPoints(pHi, cHi, 10);
    if (lineSegs.length) addLines(lineSegs, lineCols);
    g3.renderer.render(g3.scene, g3.camera);
  }

  function main() {
    const cv2d = document.getElementById('cv2d');
    const cv3d = document.getElementById('cv3d');
    const ctx2d = cv2d.getContext('2d');
    const helpScroll = document.getElementById('helpScroll');
    const statsScroll = document.getElementById('statsScroll');
    const sideSplitter = document.getElementById('sideSplitter');
    const colSplitter = document.getElementById('colSplitter');
    const statusLines = document.getElementById('statusLines');
    const predEl = document.getElementById('pred');
    const predWrap = document.getElementById('predWrap');
    const predBatchEl = document.getElementById('predBatch');
    const predBatchWrap = document.getElementById('predBatchWrap');
    const plotStripEl = document.getElementById('plotStrip');
    function sizePredCanvas() {
      if (!predWrap || !predEl) return;
      const r = predWrap.getBoundingClientRect();
      const w = Math.max(1, r.width | 0);
      const h = Math.max(1, r.height | 0);
      if (predEl.width !== w || predEl.height !== h) {
        predEl.width = w;
        predEl.height = h;
      }
    }
    function sizePredBatchCanvas() {
      if (!predBatchWrap || !predBatchEl) return;
      const r = predBatchWrap.getBoundingClientRect();
      const w = Math.max(1, r.width | 0);
      const h = Math.max(1, r.height | 0);
      if (predBatchEl.width !== w || predBatchEl.height !== h) {
        predBatchEl.width = w;
        predBatchEl.height = h;
      }
    }
    function sizeBothPlotCanvases() {
      sizePredBatchCanvas();
      sizePredCanvas();
    }
    sizeBothPlotCanvases();
    requestAnimationFrame(sizeBothPlotCanvases);
    if (typeof ResizeObserver !== 'undefined' && plotStripEl) new ResizeObserver(sizeBothPlotCanvases).observe(plotStripEl);
    const pred = predEl.getContext('2d');
    const predBatch = predBatchEl ? predBatchEl.getContext('2d') : null;
    const modalMask = document.getElementById('modalMask');
    const modalPrompt = document.getElementById('modalPrompt');
    const modalInput = document.getElementById('modalInput');
    let modalResolve = null;

    function showModal(prompt, defVal) {
      return new Promise((resolve) => {
        modalPrompt.textContent = prompt;
        modalInput.value = defVal != null ? String(defVal) : '';
        modalMask.classList.add('on');
        modalResolve = resolve;
      });
    }
    document.getElementById('modalOk').onclick = () => { modalMask.classList.remove('on'); if (modalResolve) modalResolve(modalInput.value); modalResolve = null; };
    document.getElementById('modalCancel').onclick = () => { modalMask.classList.remove('on'); if (modalResolve) modalResolve(null); modalResolve = null; };

    const config = setDefaultValues();
    const state = SimState();
    Cell.setConfig(config);
    const sliderEpochY = document.getElementById('plotYEpoch');
    const sliderMinibatchY = document.getElementById('plotYMinibatch');
    if (sliderEpochY) {
      sliderEpochY.min = String(PLOT_YMAX_MIN); sliderEpochY.max = String(PLOT_YMAX_MAX);
      sliderEpochY.value = String(state.plotEpochYmax);
      sliderEpochY.addEventListener('input', () => {
        state.plotEpochYmax = clip(+sliderEpochY.value, PLOT_YMAX_MIN, PLOT_YMAX_MAX);
      });
    }
    if (sliderMinibatchY) {
      sliderMinibatchY.min = String(PLOT_YMAX_MIN); sliderMinibatchY.max = String(PLOT_YMAX_MAX);
      sliderMinibatchY.value = String(state.plotMinibatchYmax);
      sliderMinibatchY.addEventListener('input', () => {
        state.plotMinibatchYmax = clip(+sliderMinibatchY.value, PLOT_YMAX_MIN, PLOT_YMAX_MAX);
      });
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }
    let helpPanelDomBuilt = false;
    function ensureHelpPanelDom() {
      if (helpPanelDomBuilt || !helpScroll) return;
      helpPanelDomBuilt = true;
      const body = (typeof README_HTML === 'string' && README_HTML.length)
        ? README_HTML
        : '<p><strong>README not embedded.</strong> Run <code>python3 build_neurosim_web.py</code> (needs <code>pip install markdown</code>).</p>';
      helpScroll.innerHTML =
        '<section class="quick-start"><h2 class="quick-h">Quick start (MNIST &amp; keys)</h2>' +
        '<pre class="quick-pre">' + escapeHtml(QUICK_START) + '</pre></section>' +
        '<article class="readme-body">' + body + '</article>' +
        '<hr class="help-sep" /><h2 class="help-log-h">Session log</h2>' +
        '<pre id="helpLog" class="help-log"></pre>';
    }
    function drawHelpPanel() {
      if (!helpScroll) return;
      ensureHelpPanelDom();
      const logEl = document.getElementById('helpLog');
      if (logEl) logEl.textContent = state.side_panel_text.slice(-40).join('\n');
    }
    (function initColSplitter() {
      if (!colSplitter || !document.getElementById('side')) return;
      const side = document.getElementById('side');
      let dragging = false;
      colSplitter.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        const startX = e.clientX;
        const startW = side.getBoundingClientRect().width;
        function move(ev) {
          if (!dragging) return;
          let nw = startW + (ev.clientX - startX);
          nw = clip(nw, 260, window.innerWidth * 0.92 - 120);
          side.style.flex = `0 0 ${nw}px`;
          side.style.width = `${nw}px`;
        }
        function up() {
          dragging = false;
          window.removeEventListener('mousemove', move);
          window.removeEventListener('mouseup', up);
        }
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
      });
    })();
    /* Horizontal splitter between help and stats panes */
    (function initSideSplitter() {
      if (!sideSplitter || !helpScroll || !statsScroll) return;
      let dragging = false;
      sideSplitter.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        const startY = e.clientY;
        const sideRect = helpScroll.parentElement.getBoundingClientRect();
        const startHelpH = helpScroll.getBoundingClientRect().height;
        const startStatsH = statsScroll.getBoundingClientRect().height;
        function move(ev) {
          if (!dragging) return;
          const dy = ev.clientY - startY;
          const totalH = startHelpH + startStatsH;
          let newHelpH = clip(startHelpH + dy, 40, totalH - 40);
          const helpPct = (newHelpH / totalH) * 100;
          helpScroll.style.flex = `1 1 ${helpPct}%`;
          statsScroll.style.flex = `1 1 ${100 - helpPct}%`;
        }
        function up() {
          dragging = false;
          window.removeEventListener('mousemove', move);
          window.removeEventListener('mouseup', up);
        }
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
      });
    })();
    /* Stats pane text (training stats, telemetry, network info) */
    state._stats_text = '';
    function updateStatsPane(text) {
      state._stats_text = text;
      if (statsScroll) { const prev = statsScroll.scrollTop; statsScroll.textContent = text; statsScroll.scrollTop = prev; }
    }
    function uiPrint(text, position) {
      if (state.show_3d_view) return;
      if (position == null) {
        const lines = String(text).split('\n');
        state.side_panel_text.push(...lines);
        state.side_panel_text = state.side_panel_text.slice(-50);
      }
      drawHelpPanel();
    }

    function doDraw2d() {
      ctx2d.fillStyle = '#fff'; ctx2d.fillRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
      drawCells(state, config, ctx2d);
      drawGrid(ctx2d);
    }
    // After M/J load: paint first sample so input (layer 0) and labels (last layer) are visible before T.
    function showLoadedTrainingPreview() {
      if (!state.training_data_layer_0.length) return;
      applyLayerFromGrid(state.cells, 0, state.training_data_layer_0[0]);
      applyLayerFromGrid(state.cells, config.num_layers - 1, state.training_data_num_layer_minus_1[0]);
      state.invalidateNeighborCache();
      if (!state.show_3d_view) doDraw2d();
      else state._3d_dirty = true;
    }
    /** M=MNIST F=Fashion; first letter of other dataset names; U if missing. */
    function trainingDatasetCodeFromRoot(obj) {
      const d = obj && obj.dataset != null ? String(obj.dataset).toLowerCase().trim() : '';
      if (d === 'mnist') return 'M';
      if (d === 'fashion') return 'F';
      if (d.length && /^[a-z]/i.test(d)) return d.charAt(0).toUpperCase();
      return 'U';
    }
    /** One letter from best epoch correct/total: P=100%, A=90–99%, … J=10–19%, K=<10% nonzero, N=0, X=no samples. */
    function saveFilenameAccuracyLetter(maxCorrect, total) {
      if (total <= 0) return 'X';
      if (maxCorrect >= total) return 'P';
      const r = maxCorrect / total;
      if (r >= 0.9) return 'A';
      if (r >= 0.8) return 'B';
      if (r >= 0.7) return 'C';
      if (r >= 0.6) return 'D';
      if (r >= 0.5) return 'E';
      if (r >= 0.4) return 'F';
      if (r >= 0.3) return 'G';
      if (r >= 0.2) return 'H';
      if (r >= 0.1) return 'I';
      if (r > 0) return 'J';
      return 'N';
    }
    /** Build a compact filename: saved_N_980_1000_L8W9_250402_045.json
     *  N/F = dataset, maxCorrect_epochSize, L=layers W=weights, YYMMDD, 3-digit ms. */
    function downloadSimulationJson(tag) {
      const d = /^[A-Z]$/i.test(state.training_dataset_code || '') ? String(state.training_dataset_code).toUpperCase() : 'X';
      // N for number-MNIST, keep F for Fashion, others as-is
      const dLetter = (d === 'M') ? 'N' : d;
      const maxC = state.max_bingo_count || 0;
      const epSz = config.how_much_training_data || 0;
      const now = new Date();
      const yy = String(now.getFullYear()).slice(-2);
      const mm = String(now.getMonth() + 1).padStart(2, '0');
      const dd = String(now.getDate()).padStart(2, '0');
      const ms3 = String(now.getMilliseconds()).padStart(3, '0');
      const payload = {
        version: 1, config: configToJSON(config), cells: serializeCells(state.cells), training_cycles: state.training_cycles,
        training_dataset_code: d, max_bingo_count: state.max_bingo_count, bingo_count: state.bingo_count,
      };
      const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      const core = `saved_${dLetter}_${maxC}_${epSz}_L${config.num_layers}W${config.number_of_weights}_${yy}${mm}${dd}_${ms3}`;
      const fname = tag ? `${core}${tag}.json` : `${core}.json`;
      a.download = fname;
      a.click();
      URL.revokeObjectURL(a.href);
      return fname;
    }
    /* Must match backPropagation: renderBackpropFn(z, x, y) — not (layer, [x,y]). */
    function doRenderBackprop(layer, x, y) {
      render3dBackprop(state, config, layer, x, y);
    }

    init3d(cv3d);

    function layout2d3d() {
      if (state.show_3d_view) {
        cv2d.style.display = 'none';
        cv3d.style.display = 'block';
      } else {
        cv3d.style.display = 'none';
        cv2d.style.display = 'block';
        document.getElementById('hud3d').style.display = 'none';
      }
    }

    let dragging = false, lastMx = 0, lastMy = 0;
    cv3d.addEventListener('mousedown', (e) => { dragging = true; lastMx = e.clientX; lastMy = e.clientY; });
    window.addEventListener('mouseup', () => { dragging = false; });
    cv3d.addEventListener('mousemove', (e) => {
      if (state.show_3d_view && dragging) {
        state.rotation_y += (e.clientX - lastMx) * 0.5;
        state.rotation_x += (e.clientY - lastMy) * 0.5;
        lastMx = e.clientX; lastMy = e.clientY;
      }
    });
    cv3d.addEventListener('wheel', (e) => { if (state.show_3d_view) { state.zoom += e.deltaY > 0 ? -0.5 : 0.5; e.preventDefault(); } }, { passive: false });

    cv2d.addEventListener('mousedown', (e) => {
      const [mx, my] = canvasPointerToInternal(cv2d, e.clientX, e.clientY);
      const [cx, cy, layer] = convertXYToIndex(mx, my);
      if (e.button === 2 || (e.button === 0 && e.ctrlKey)) {
        const c = state.cells[cx][cy][layer];
        if (c) {
          const ct = updateCellTypes(state.cells, config);
          const phen = updatePhenotypeCellTypes(state.cells, config);
          const tc = phen[1], cp = phen[0];
          uiPrint(`Cells ${tc} | Positive ${cp} | Fraction ${(cp / (tc + EPS)).toFixed(2)}`);
          uiPrint(c.toString());
        } else uiPrint('No cell at this location');
        return;
      }
        if (e.button === 0) {
        if (state.cells[cx][cy][layer] == null) {
          state.cells[cx][cy][layer] = new Cell(layer, cx, cy, config.number_of_weights, config.bias_range, config.avg_weights_cell,
            config.charge_delta, config.weight_decay, config.mutation_rate, null);
        } else state.cells[cx][cy][layer] = null;
        state.mouse_up = false; state.invalidateNeighborCache(); state._3d_dirty = true;
        doDraw2d();
      }
    });
    cv2d.addEventListener('mousemove', (e) => {
      if (state.mouse_up || e.buttons !== 1) return;
      const [mx, my] = canvasPointerToInternal(cv2d, e.clientX, e.clientY);
      const [cx, cy, layer] = convertXYToIndex(mx, my);
      if (state.cells[cx][cy][layer] == null) {
        state.cells[cx][cy][layer] = new Cell(layer, cx, cy, config.number_of_weights, config.bias_range, config.avg_weights_cell,
          config.charge_delta, config.weight_decay, config.mutation_rate, null);
        state.invalidateNeighborCache(); state._3d_dirty = true;
        doDraw2d();
      }
    });
    cv2d.addEventListener('mouseup', () => { state.mouse_up = true; });
    cv2d.addEventListener('contextmenu', (e) => e.preventDefault());
    cv2d.addEventListener('click', () => { try { cv2d.focus(); } catch (_) {} });
    cv3d.addEventListener('click', () => { try { cv3d.focus(); } catch (_) {} });
    try { cv2d.focus(); } catch (_) {}

    async function onKeyDown(e) {
      const k = e.key;
      if (k === '?' || (k === '/' && e.shiftKey)) {
        if (helpScroll) helpScroll.scrollTop = 0;
        drawHelpPanel();
        return;
      }
      if (!state.show_3d_view && k.toLowerCase() === 'h') {
        ensureHelpPanelDom();
        /* Quick-start title first, then every ## in the README preview */
        const heads = helpScroll ? helpScroll.querySelectorAll('.quick-start h2.quick-h, .readme-body h2') : [];
        if (heads.length === 0) return;
        state.current_index = (state.current_index + 1) % heads.length;
        heads[state.current_index].scrollIntoView({ block: 'start', behavior: 'smooth' });
        return;
      }
      if (k === ' ') { state.running = !state.running; return; }
      if (k.toLowerCase() === 'u') {
        config.autonomous_network_genes = !config.autonomous_network_genes;
        uiPrint(`Autonomous network genes: ${config.autonomous_network_genes}`);
        return;
      }
      if (k.toLowerCase() === 'p') { state.prune = !state.prune; uiPrint('Charge pruning (P): ' + (state.prune ? 'ON' : 'OFF')); return; }
      if (k.toLowerCase() === 'o') { state.gradient_prune = !state.gradient_prune; uiPrint('Gradient pruning (O): ' + (state.gradient_prune ? 'ON' : 'OFF')); return; }
      if (k.toLowerCase() === 'y') { state.weight_mag_prune = !state.weight_mag_prune; uiPrint('Weight-magnitude pruning (Y): ' + (state.weight_mag_prune ? 'ON' : 'OFF')); return; }
      if (k.toLowerCase() === 'z') { state.contrib_score_prune = !state.contrib_score_prune; uiPrint('Contribution-score pruning (Z): ' + (state.contrib_score_prune ? 'ON' : 'OFF')); return; }
      if (k === '=' || k === '+') { state.prune_logic = state.prune_logic === 'AND' ? 'OR' : 'AND'; return; }
      if (k.toLowerCase() === 'c' && !state.show_3d_view && state.charge_change_protection) {
        const sug = suggestParams(state, config);
        const hint = sug ? ` [suggested: ${sug.charge_delta}]` : '';
        const hintG = sug ? ` [suggested: ${sug.gradient_threshold}]` : '';
        const a = await showModal(`Charge delta (current: ${config.charge_delta})${hint}:`, sug ? sug.charge_delta : config.charge_delta);
        const b = await showModal(`Gradient threshold (current: ${config.gradient_threshold})${hintG}:`, sug ? sug.gradient_threshold : config.gradient_threshold);
        if (a != null && !Number.isNaN(+a)) config.charge_delta = +a;
        if (b != null && !Number.isNaN(+b)) config.gradient_threshold = +b;
        /* Contribution score threshold (Strategy 2) */
        const medCS = sug ? (sug.medGrad * Math.max(sug.medCharge, 0.001)) : 0;
        const hintCS = sug ? ` [median contribution: ${medCS.toExponential(2)}]` : '';
        const cs = await showModal(`Min contribution score (charge_diff×gradient, 0=off, current: ${config.min_contribution_score})${hintCS}:`, config.min_contribution_score);
        if (cs != null && !Number.isNaN(+cs)) config.min_contribution_score = Math.max(0, +cs);
        /* Percentile pruning (Strategy 3) */
        const pp = await showModal(`Prune bottom N% by contribution score each epoch (0=off, e.g. 10, current: ${config.prune_percentile}):`, config.prune_percentile);
        if (pp != null && !Number.isNaN(+pp)) config.prune_percentile = clip(+pp, 0, 50);
        return;
      }
      if (k.toLowerCase() === 'd') { state.display_updating = !state.display_updating; return; }
      if (k.toLowerCase() === 'i') {
        const sug = suggestParams(state, config);
        const hint = sug ? ` [suggested: ${sug.lr} based on fan-in ${sug.avgFanIn.toFixed(1)}]` : '';
        const v = await showModal(`Learning rate (current: ${config.learning_rate})${hint}:`, sug ? sug.lr : config.learning_rate);
        if (v != null && !Number.isNaN(+v)) config.learning_rate = +v;
        return;
      }
      if (k.toLowerCase() === 'k' && !state.show_3d_view) {
        const v = await showModal(
          `Gradient minibatch (1 = weight update every image; N = sum grads over N images then one update; MNIST often 16–64). Current:`,
          config.gradient_minibatch_size
        );
        if (v != null && !Number.isNaN(+v)) {
          config.gradient_minibatch_size = Math.max(1, (+v) | 0);
          uiPrint(`gradient_minibatch_size=${config.gradient_minibatch_size} (per-image forward+back; weights change every ${config.gradient_minibatch_size} samples when BackProp is on)`);
        }
        return;
      }
      if (k.toLowerCase() === 'f') { state.direction_of_charge_flow = '+++++>>>>>'; return; }
      if (k.toLowerCase() === 'r') { state.direction_of_charge_flow = '<<<<<-----'; return; }
      if (k.toLowerCase() === 'b') { state.back_prop = !state.back_prop; return; }
      if (k.toLowerCase() === 'a') { state.andromida_mode = !state.andromida_mode; return; }
      if (k.toLowerCase() === 'w') { resetAllGradientChanges(state, config); return; }
      if (k.toLowerCase() === 'g' && state.show_3d_view) {
        state._3d_color_mode = (state._3d_color_mode + 1) % _3D_COLOR_MODE_COUNT;
        uiPrint('3D color: ' + _3D_COLOR_MODE_NAMES[state._3d_color_mode]);
        return;
      }
      if (k.toLowerCase() === 'g' && !state.show_3d_view) { state.display = state.display === 'genes' ? 'proteins' : 'genes'; return; }
      if (k.toLowerCase() === 'q' && !state.show_3d_view) { updateStatsPane(formatTelemetry(computeTelemetry(state, config))); return; }
      if (k === '3') {
        state.show_3d_view = !state.show_3d_view; state._3d_dirty = true; layout2d3d(); return;
      }
      if (k === '4') {
        state.show_backprop_view = !state.show_backprop_view;
        if (state.show_backprop_view) {
          state.show_3d_view = true; layout2d3d();
          /* Auto-switch to Error color mode so you see backprop activity across the full network */
          state._3d_color_mode = 1;
          uiPrint('Backprop view ON — color: Error (G to cycle). Full network updates each sample.');
        } else {
          state._3d_color_mode = 0;
          uiPrint('Backprop view OFF — color: Charge');
        }
        state._3d_dirty = true; return;
      }
      if (k.toLowerCase() === 't') {
        if (!state.training_data_loaded) uiPrint('Load training data first (M)');
        else {
          state.training_mode = !state.training_mode;
          state._training_sample_i = state.training_mode ? 0 : null;
        }
        return;
      }
      if (k.toLowerCase() === 'm' && !state.show_3d_view) {
        const ds = await showModal(
          'Load training data\n\n'
          + 'Type J + OK = choose a local .json file (from mnist_to_neurosim_web_json.py)\n'
          + 'Type D + OK = fetch a 500-sample demo from same folder / GitHub (then pick MNIST vs Fashion)\n'
          + 'Type M + OK = synthetic random images (no file)\n\n'
          + 'Enter J, D, or M:',
          'J'
        );
        const n = await showModal('How many samples per training cycle? (must be ≤ rows in JSON file, e.g. 500)', config.how_much_training_data);
        const st = await showModal('Start index into dataset (usually 0):', config.start_index);
        let ns = n != null && !Number.isNaN(+n) ? Math.max(1, (+n) | 0) : config.how_much_training_data;
        const ss = st != null && !Number.isNaN(+st) ? Math.max(0, (+st) | 0) : 0;
        if (ss + ns > 5000) uiPrint('Total exceeds 5000 — cancelled');
        else {
          config.start_index = ss;
          state.training_data_layer_0 = [];
          state.training_data_num_layer_minus_1 = [];
          if ((ds || 'm').toLowerCase() === 'j') {
            uiPrint('Select your .json file in the chooser (must contain "samples" array).');
            await new Promise((res) => {
              const fp = document.getElementById('filePick');
              fp.onchange = () => {
                const f = fp.files[0];
                fp.value = '';
                if (!f) { res(); return; }
                const fr = new FileReader();
                fr.onload = () => {
                  try {
                    const obj = JSON.parse(fr.result);
                    const ar = obj.samples != null ? obj.samples : (Array.isArray(obj) ? obj : null);
                    if (!Array.isArray(ar) || !ar.length) {
                      uiPrint('JSON needs { "samples": [ { "layer0": [...], "layerLast": [...] }, ... ] }');
                      res();
                      return;
                    }
                    /* Detect compact format (v2): samples have {c:[...], l:N} instead of {layer0, layerLast} */
                    const isCompact = ar[0] && ar[0].c != null && ar[0].l != null;
                    const layerLastIdx = obj.layer_last_index_note || (config.num_layers - 1);
                    const fromFile = ar.length - ss;
                    const nUse = Math.max(0, Math.min(ns, fromFile));
                    if (nUse === 0) {
                      uiPrint(`No samples for start_index=${ss} (file has ${ar.length} samples).`);
                      res();
                      return;
                    }
                    for (let i = 0; i < nUse; i++) {
                      const raw = ar[ss + i];
                      const row = isCompact ? expandCompactSample(raw, layerLastIdx) : raw;
                      if (!row || !row.layer0 || !row.layerLast) {
                        uiPrint(`Bad sample at index ${ss + i}; need layer0 + layerLast`);
                        continue;
                      }
                      state.training_data_layer_0.push(row.layer0);
                      state.training_data_num_layer_minus_1.push(row.layerLast);
                    }
                    if (state.training_data_layer_0.length === 0) {
                      uiPrint('No valid samples loaded.');
                      res();
                      return;
                    }
                    config.how_much_training_data = state.training_data_layer_0.length;
                    state.total_weights_list = new Float64Array(config.how_much_training_data + 10);
                    state.training_data_loaded = true;
                    state.training_dataset_code = trainingDatasetCodeFromRoot(obj);
                    state.reset_training_metrics();
                    state._training_sample_i = null;
                    showLoadedTrainingPreview();
                    uiPrint(`Loaded ${state.training_data_layer_0.length} samples (start ${ss}). First sample on grid. Press T to train.`);
                    if (nUse < ns) uiPrint(`Note: only ${nUse} available from index ${ss} (you asked ${ns}).`);
                  } catch (err) { uiPrint('JSON error: ' + err); }
                  res();
                };
                fr.readAsText(f);
              };
              fp.click();
            });
          } else if ((ds || '').toLowerCase() === 'd') {
            const demoPick = await showModal(
              'Which hosted demo (500 samples each)?\n\n'
              + 'M + OK = MNIST (mnist_demo_500.json)\n'
              + 'F + OK = Fashion-MNIST (fashion-mnist_demo_500.json)\n\n'
              + 'Enter M or F:',
              'M'
            );
            const useFashion = (demoPick || 'm').toLowerCase().trim().startsWith('f');
            const demoFile = useFashion ? 'fashion-mnist_demo_500.json' : 'mnist_demo_500.json';
            if (useFashion) uiPrint(FASHION_LABELS);
            /* Fetch demo JSON from server (works on GitHub Pages or local http server) */
            uiPrint('Fetching ' + demoFile + ' …');
            try {
              const resp = await fetch(demoFile);
              if (!resp.ok) throw new Error(`HTTP ${resp.status} – is ${demoFile} in the same folder?`);
              const obj = JSON.parse(await resp.text());
              const ar = obj.samples != null ? obj.samples : (Array.isArray(obj) ? obj : null);
              if (!Array.isArray(ar) || !ar.length) throw new Error('No samples array in file');
              const isCompact = ar[0] && ar[0].c != null && ar[0].l != null;
              const layerLastIdx = obj.layer_last_index_note || (config.num_layers - 1);
              const fromFile = ar.length - ss;
              const nUse = Math.max(0, Math.min(ns, fromFile));
              if (nUse === 0) { uiPrint(`No samples at start_index=${ss}`); }
              else {
                for (let i = 0; i < nUse; i++) {
                  const raw = ar[ss + i];
                  const row = isCompact ? expandCompactSample(raw, layerLastIdx) : raw;
                  if (!row || !row.layer0 || !row.layerLast) continue;
                  state.training_data_layer_0.push(row.layer0);
                  state.training_data_num_layer_minus_1.push(row.layerLast);
                }
                config.how_much_training_data = state.training_data_layer_0.length;
                state.total_weights_list = new Float64Array(config.how_much_training_data + 10);
                state.training_data_loaded = true;
                state.training_dataset_code = trainingDatasetCodeFromRoot(obj);
                state.reset_training_metrics();
                state._training_sample_i = null;
                showLoadedTrainingPreview();
                uiPrint(`Loaded ${state.training_data_layer_0.length} demo samples. Press T to train.`);
              }
            } catch (err) { uiPrint('Demo fetch failed: ' + err.message); }
          } else {
            config.how_much_training_data = ns;
            for (let i = 0; i < config.how_much_training_data; i++) {
              const pair = buildSyntheticTraining(config);
              state.training_data_layer_0.push(pair.layer0.map(row => row.map(c => c ? c.toJSON() : null)));
              state.training_data_num_layer_minus_1.push(pair.layerLast.map(row => row.map(c => c ? c.toJSON() : null)));
            }
            state.total_weights_list = new Float64Array(config.how_much_training_data + 10);
            state.training_data_loaded = true;
            state.training_dataset_code = 'S';
            state.reset_training_metrics();
            state._training_sample_i = null;
            showLoadedTrainingPreview();
            uiPrint(`Synthetic training: ${state.training_data_layer_0.length} samples. First sample on grid. Press T to train.`);
          }
        }
        return;
      }
      if (k.toLowerCase() === 'e') {
        const oldNumWeights = config.number_of_weights;
        const sug = suggestParams(state, config);
        const s = (label, cur, sugVal) => {
          const hint = sug && sugVal !== undefined ? ` [net suggests: ${sugVal}]` : '';
          return `${label} (current: ${cur})${hint}:`;
        };
        const nl = await showModal(`num_layers (4-16):`, config.num_layers);
        const ld = await showModal(`dendrite length:`, config.length_of_dendrite);
        if (nl != null) config.num_layers = clip(+nl | 0, 4, 16);
        if (ld != null) config.length_of_dendrite = clip(+ld | 0, 1, 4);
        config.mutation_rate = +(await showModal('mutation_rate:', config.mutation_rate) ?? config.mutation_rate);
        config.lower_allele_range = +(await showModal('lower_allele:', config.lower_allele_range) ?? config.lower_allele_range);
        config.upper_allele_range = +(await showModal('upper_allele:', config.upper_allele_range) ?? config.upper_allele_range);
        config.weight_change_threshold = +(await showModal('weight_change_threshold:', config.weight_change_threshold) ?? config.weight_change_threshold);
        config.avg_weights_cell = +(await showModal(s('avg_weights_cell', config.avg_weights_cell, sug ? sug.avg_weights_cell : undefined), sug ? sug.avg_weights_cell : config.avg_weights_cell) ?? config.avg_weights_cell);
        config.weight_decay = +(await showModal(s('weight_decay', config.weight_decay, sug ? sug.weight_decay : undefined), sug ? sug.weight_decay : config.weight_decay) ?? config.weight_decay);
        config.bias_range = +(await showModal(s('bias_range', config.bias_range, sug ? sug.bias_range : undefined), sug ? sug.bias_range : config.bias_range) ?? config.bias_range);
        config.learning_rate = +(await showModal(s('learning_rate', config.learning_rate, sug ? sug.lr : undefined), sug ? sug.lr : config.learning_rate) ?? config.learning_rate);
        config.charge_delta = +(await showModal(s('charge_delta', config.charge_delta, sug ? sug.charge_delta : undefined), sug ? sug.charge_delta : config.charge_delta) ?? config.charge_delta);
        config.gradient_threshold = +(await showModal(s('gradient_threshold', config.gradient_threshold, sug ? sug.gradient_threshold : undefined), sug ? sug.gradient_threshold : config.gradient_threshold) ?? config.gradient_threshold);
        config.activation_slope = +(await showModal(s('activation_slope', config.activation_slope, sug ? sug.activation_slope : undefined), config.activation_slope) ?? config.activation_slope);
        config.immune_period = clip(+(await showModal('immune_period (10-100, training cycles before cell can be pruned):', config.immune_period) ?? config.immune_period) | 0, 10, 100);
        config.updateDerived();
        /* Desktop neurosim/main.py: remap only when weight count (dendrite footprint) changes. */
        if (config.number_of_weights !== oldNumWeights) {
          for (let z = 1; z < config.num_layers - 1; z++)
            for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
              const c = state.cells[x][y][z];
              if (c) c.remapWeights(config.length_of_dendrite);
            }
        }
        state.invalidateNeighborCache(); state._3d_dirty = true;
        /* Auto-update avg_weights_cell when dendrite changes */
        if (config.number_of_weights !== oldNumWeights) {
          const mfi = computeAvgFanIn(state, config);
          config.avg_weights_cell = Math.max(1, Math.round(mfi));
          uiPrint(`avg_weights_cell auto-set to ${config.avg_weights_cell} (fan-in: ${mfi.toFixed(1)})`);
        }
        uiPrint(`Updated dendrite=${config.length_of_dendrite} weights=${config.number_of_weights}`);
        return;
      }
      if (k.toLowerCase() === 'x') {
        /* Auto-compute avg_weights_cell from actual fan-in before re-init for proper He scaling */
        const sug = suggestParams(state, config);
        if (sug) {
          config.avg_weights_cell = sug.avg_weights_cell;
          config.bias_range = sug.bias_range;
        } else {
          const measuredFanIn = computeAvgFanIn(state, config);
          config.avg_weights_cell = Math.max(1, Math.round(measuredFanIn));
        }
        for (let z = 1; z < config.num_layers - 1; z++)
          for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) {
            const c = state.cells[x][y][z];
            if (c) {
              c.initalizeNetworkGenes(config.number_of_weights, config.bias_range, config.avg_weights_cell,
                config.charge_delta, config.weight_decay, config.mutation_rate, state.cells);
              c.colorGenes(); c.initializeNetworkProteins(); c.colorProteins();
            }
          }
        state.not_saved_yet = true; state._3d_dirty = true; state.reset_training_metrics();
        uiPrint(`Reset weights+bias. fan-in=${config.avg_weights_cell}, bias_range=${config.bias_range}, He scale=√(2/${config.avg_weights_cell})=${Math.sqrt(2/config.avg_weights_cell).toFixed(3)}`);
        return;
      }
      if (k.toLowerCase() === 'n') {
        const ok = await showModal('Nuke hidden layers? y/n', 'n');
        if ((ok || '').toLowerCase() === 'y') {
          for (let z = 1; z < config.num_layers - 1; z++)
            for (let x = 0; x < WIDTH; x++) for (let y = 0; y < HEIGHT; y++) state.cells[x][y][z] = null;
          state.invalidateNeighborCache(); state._3d_dirty = true; state.reset_training_metrics();
        }
        return;
      }
      if (k.toLowerCase() === 's') {
        const fn = downloadSimulationJson('');
        state.not_saved_yet = false;
        uiPrint('Saved ' + fn + '  (N=numbers F=fashion, maxCorrect_epoch, L=layers W=weights, YYMMDD, ms)');
        return;
      }
      if (k.toLowerCase() === 'l' && !state.show_3d_view) {
        await new Promise((res) => {
          const fp = document.getElementById('filePick');
          fp.onchange = () => {
            const f = fp.files[0];
            if (!f) { res(); return; }
            const fr = new FileReader();
            fr.onload = () => {
              try {
                const obj = JSON.parse(fr.result);
                Object.assign(config, configFromJSON(obj.config));
                Cell.setConfig(config);
                state.cells = deserializeCells(obj.cells, config);
                state.training_cycles = obj.training_cycles || 0;
                if (obj.training_dataset_code != null && String(obj.training_dataset_code).length === 1)
                  state.training_dataset_code = String(obj.training_dataset_code).toUpperCase();
                if (obj.max_bingo_count != null) state.max_bingo_count = +obj.max_bingo_count;
                /* Reset mid-epoch training state so the next tick starts a fresh epoch.
                   Without this, _training_sample_i can point past the data array (crash),
                   and bingo_count from the saved file gets double-counted into the current epoch. */
                state._training_sample_i = null;
                state._shuffle_order = null;
                state.bingo_count = 0;
                state.invalidateNeighborCache(); state._3d_dirty = true;
                uiPrint('Loaded JSON state');
              } catch (err) { uiPrint('Load error: ' + err); }
              res();
            };
            fr.readAsText(f);
          };
          fp.click();
        });
        return;
      }
      if (k.toLowerCase() === 'v' && !state.show_3d_view) {
        state.display_set = (state.display_set + 1) % 3;
        if (state.display_set === 0) updateStatsPane(getAllSettings(state, config));
        else if (state.display_set === 1) updateStatsPane(formatAverages(state, config));
        else {
          const ct = updateCellTypes(state.cells, config);
          const pt = updatePhenotypeCellTypes(state.cells, config);
          updateStatsPane(
            formatStatistics(ct) + '\n\n' +
            formatPhenotypeStatistics(pt) + '\n\n' +
            formatMaxChargeDiff(state, config, 5)
          );
        }
        state.show_training_stats = !state.show_training_stats;
        return;
      }
    }
    window.addEventListener('keydown', (e) => { onKeyDown(e); });

    let startT = performance.now();
    function tick() {
      requestAnimationFrame(tick);
      /* neurosim/main.py: autosave when a full training batch is ever classified perfectly */
      if (state.max_bingo_count === config.how_much_training_data && state.not_saved_yet) {
        const fn = downloadSimulationJson('-perfect');
        state.not_saved_yet = false;
        uiPrint('Perfect epoch: auto-saved ' + fn);
      }
      if (state.running) updateCells(state, config);
      if (state.training_mode) {
        /* Guard: if data was reloaded or network loaded, clamp setSize to actual data length */
        const actualDataLen = state.training_data_layer_0.length;
        if (config.how_much_training_data > actualDataLen && actualDataLen > 0) {
          config.how_much_training_data = actualDataLen;
        }
        const setSize = config.how_much_training_data;
        try {
          /* Always one sample per rAF — keeps browser responsive. D key only toggles cell drawing, not training pace or graphs. */
          if (state._training_sample_i == null) state._training_sample_i = 0;
          /* If sample counter somehow exceeds data, reset to start of next epoch */
          if (state._training_sample_i >= setSize) state._training_sample_i = 0;
          if (state._training_sample_i === 0) {
            state.bingo_count = 0;
            state._batch_loss_sum = 0;
            state._batch_sample_count = 0;
            state._epoch_digit_correct = new Array(10).fill(0);
            state._epoch_digit_total = new Array(10).fill(0);
            state.training_cycles++;
            /* Shuffle sample order at the start of each epoch */
            if (config.shuffle_epoch) {
              state._shuffle_order = shuffleIndices(makeIndexArray(setSize));
            }
          }
          const i = state._training_sample_i;
          trainOnSample(state, config, i, null, pred);
          if (state.display_updating && !state.show_3d_view) doDraw2d();
          state._training_sample_i++;
          if (state._training_sample_i >= setSize) {
            const ba = state._batch_loss_sum / Math.max(1, state._batch_sample_count);
            predictionPlotEpoch(state, predBatch, ba);
            state._training_sample_i = 0;
            /* Epoch-boundary percentile pruning (competitive / trophic factor) */
            if (config.prune_percentile > 0 && (state.prune || state.gradient_prune || state.weight_mag_prune || state.contrib_score_prune)) {
              const killed = percentilePrune(state, config);
              if (killed > 0) uiPrint(`Epoch ${state.training_cycles}: pruned ${killed} cells (bottom ${config.prune_percentile}% by contribution score)`);
            }
            if (state.show_training_stats && state.training_cycles % state.stats_update_frequency === 0) {
              updateTrainingStats(state, config);
              updateStatsPane(formatTrainingStats(state));
            }
          }
        } catch (err) { console.error(err); uiPrint('Training error: ' + err); }
      }
      if (state.display_updating) {
        if (state.show_3d_view) {
          render3dNetwork(state, config);
        } else {
          doDraw2d();
        }
      }
      const now = performance.now();
      const elapsed = (now - startT) / 1000;
      startT = now;
      const col = state.running ? '#0f0' : '#fff';
      /* Per-epoch and per-minibatch loss for status bar */
      const lastEpochLoss = state.epochLossPoints.length ? state.epochLossPoints[state.epochLossPoints.length - 1].toFixed(2) : '--';
      const lastMinibatchLoss = state.minibatchLossPoints.length ? state.minibatchLossPoints[state.minibatchLossPoints.length - 1].toFixed(2) : '--';
      const epochProgress = state.training_mode && state._training_sample_i != null
        ? ` (${state._training_sample_i}/${config.how_much_training_data})`
        : '';
      const K = Math.max(1, config.gradient_minibatch_size | 0);
      const pAny = state.prune || state.gradient_prune || state.weight_mag_prune || state.contrib_score_prune;
      const pStr = pAny
        ? `P:Charge=${state.prune} O:Grad=${state.gradient_prune} Y:Weight=${state.weight_mag_prune} Z:Contrib=${state.contrib_score_prune} ${state.prune_logic}`
        : 'Pruning=off';
      statusLines.innerHTML = `<span style="color:${col}">Run=${state.running} Andromida=${state.andromida_mode} ${pStr} | ` +
        `Train=${state.training_mode} LR=${config.learning_rate.toFixed(4)} | CellAutonomous(4-13)=${config.autonomous_network_genes}</span><br/>` +
        `Dir=${state.direction_of_charge_flow} BackProp=${state.back_prop} Display=${state.display_updating} | ` +
        `Epoch=${state.training_cycles}${epochProgress} Samples=${config.how_much_training_data} K=${K} | ` +
        `EpochLoss=${lastEpochLoss} BatchLoss=${lastMinibatchLoss} | Correct=${state.bingo_count}/${config.how_much_training_data} Max=${state.max_bingo_count}`;
      /* When training stops, keep the last loss graphs visible (don't blank them).
         Only clear plots when there is no training data at all. */
      if (!state.training_mode && !state.training_data_loaded && !state.show_3d_view) {
        pred.fillRect(0, 0, pred.canvas.width, pred.canvas.height);
        if (predBatch) predBatch.fillRect(0, 0, predBatch.canvas.width, predBatch.canvas.height);
      }
    }

    drawHelpPanel();
    doDraw2d();
    tick();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', main);
  else main();
})();
