def get_defs():
     
    jmr_defs = """
    The function of each gene:

    Overcrowding Tolerance (OT): This gene 
    determines the maximum number of 
    neighbors a cell can have before it 
    dies due to overcrowding.
    For example, if OT = 3, the cell will 
    die if it has more than 3 neighbors.

    Isolation Tolerance (IT): This gene 
    determines the minimum number of 
    neighbors a cell needs to survive.
    For example, if IT = 2, a cell with 
    fewer than 2 neighbors will die.

    Birth Threshold (BT): This gene 
    determines the number of neighbors an 
    empty grid space needs to produce a 
    new cell. If an empty space has 
    exactly this many neighbors, a new 
    cell will be born.
    For example, if BT = 3, a new cell 
    will be born in an empty space that 
    has exactly 3 neighbors. And inherit 
    the genes by recombination between two 
    parents ONE of whom could give it the 
    BT = 3 allele.

    Mutation Rate (MR): This gene
    determines the chance of a cell's
    genes mutating. Two contexts:
    - At BIRTH: chance = MR / 1000
      (germline mutation during crossover)
    - During LIFE: chance = MR / 100000
      (somatic mutation each evolution tick)

    A "germ-line" mutation in the BT gene
    gives a new cell the chance to be BORN
    even if neither parent had a BT gene
    value allowing it to be born in that
    position.
    Not until the next cycle will a cell
    be checked for OT & IT. So many cells
    are born that die in the next cycle.

    Somatic mutation: any gene randomly
    changes -- breeding genes to a new
    allele in the configured range, network
    genes re-randomized per their type.
    """

    jmr_defs2 = """
    Cells inherit these genes from their
    parent cells (random independent
    selection of two parents if multiple
    are available, and then independent
    assortment of genes), with a chance of
    (germline) mutation based on their
    inherited Mutation Rate each cycle.

    The world is made up of 3D grid of
    potential cells. Each cell has a set
    of genes that determine its behavior.
    Each cell is displayed as an
    informative 3 x 3 arrangement of
    colored blocks.

    3x3 Display:  OT      IT      BT
                  MR      Charge  Error
                  Bias    Weight  Gradient

    -- Breeding Genes (always per-cell) --
    Overcrowding Tolerance (OT): .genes[0]
    neighbors <= OT (cell dies if more)
    Isolation Tolerance (IT)   : .genes[1]
    neighbors >= IT (cell dies if fewer)
    Birth Threshold (BT)       : .genes[2]
    neighbors == BT for birth

    OT >= neighbors >= IT, OT > IT

    -- Network Genes (per-cell when U on) --
    Mutation Rate (MR)         : .genes[3]
    Dendrite Size (WG)         : .genes[4]
    Bias Range (BR)            : .genes[5]
    Fan-In / Avg Weights (AW)  : .genes[6]
    Charge Delta (CD)          : .genes[7]
    Weight Decay (WD)          : .genes[8]
    Learning Rate (LR)         : .genes[9]
    Gradient Threshold (GT)    : .genes[10]
    Activation Slope (AS)      : .genes[11]
    """

    conways_defs = """
    Original Conway's Game of Life has
    fixed rules for all cells:

    Overcrowding (OT=3): Any cell with
    more than 3 neighbors dies.
    Isolation (IT=2): Any cell with
    fewer than 2 neighbors dies.
    Birth (BT=3): Empty space with
    exactly 3 neighbors spawns a cell.

    In Conway's: no genes, no mutation,
    no evolution. Rules are static.

    JMR's version makes OT, IT, BT into
    per-cell GENES that are inherited:
    OT: .genes[0]  neighbors <= OT
    IT: .genes[1]  neighbors >= IT
    BT: .genes[2]  neighbors == BT

    Survival: IT <= neighbors <= OT
    (with OT > IT enforced)
    """

    how_network_works = """
    The weights represent the influence 
    that a given neuron (in this case, a 
    cell) in one layer has on the neurons 
    in the subsequent layer. By adjusting 
    these weights, we allow the network to 
    learn over time. Loaded data 
    introduced into layer 0.

    During Training, adjustments made to 
    weights throughout the network, based 
    on error signals determined by the 
    difference between the actual outputs 
    in layer 14 and the desired outputs in 
    layer 15 (This is NUM_LAYERS-2 and 
    NUM_LAYERS-1) and so on for each 
    subsequence layer. The training 
    process involves adjusting weights to 
    minimize the error between the 
    network's output and this desired 
    output.

    This is a form of supervised learning, 
    where the network is guided towards 
    the correct output by feedback 
    provided in the form of a desired 
    output.
    """

    forward_pass = """
    Forward pass moves input data through
    the network layer by layer. Each cell
    computes:
      charge = leaky_ReLU(bias +
        sum(upstream_charge * weight))

    Layer N-2:       Layer N-1:       Layer N:

    B11 B12 B13      C11 C12 C13      W11 W12 W13
    B21 B22 B23  --> C21 C22 C23  --> W21 W22 W23
    B31 B32 B33      C31 C32 C33      W31 W32 W33

    1. Each cell gathers charges from
    upstream cells within its dendrite
    reach (gene 4 controls reach size).

    2. The cell computes a weighted sum
    of upstream charges using its own
    weights, adds its bias, then applies
    leaky ReLU activation (gene 11
    controls the negative slope).

    3. The resulting charge is clipped
    to [-10, 10] and stored as the
    cell's new charge value.
    """

    how_backprop_works = """
    The weights of the current cell are
    updated based on the error of the
    current cell and the charge of the
    cells in the layer above.

    `get_upper_layer_cells` gathers the
    upstream cells within dendrite reach.
    These are passed to
    `update_weights_and_bias` to compute
    gradient = error * upstream_charge
    and update each weight:
      w -= lr * gradient + decay * w

    `get_layer_below_cells` gathers the
    downstream cells. These are passed to
    `compute_error_signal` to propagate
    error backward through the reversed
    weight index."""

    how_backprop_works2 = """
    Backprop figure:
    Layer N-1 (Above) Layer N (Current) Layer N+1 (Below)

    Cell A          Cell X                 Cell 1
    Cell B          Cell Y                 Cell 2
    Cell C          Cell Z                 Cell 3

    Cells A, B, C are in the layer above.
    Cells X, Y, Z are in the current layer.
    Cells 1, 2, 3 are in the layer below.

    WEIGHT UPDATE (update_weights_and_bias):
    For Cell X, we use charges of A, B, C
    (layer above) and the error of Cell X.
    weight_index = (dx+reach)*matrix+(dy+reach)

    ERROR SIGNAL (compute_error_signal):
    For Cell X, we use errors and weights
    of Cells 1, 2, 3 (layer below) and
    the leaky ReLU derivative of Cell X.
    reversed_index = len(weights)-1-weight_index
    This maps (dx,dy) to (-dx,-dy), which
    is equivalent to transposing the weight
    matrix in standard backprop.
    """

    controls = """
    Left-click/Drag - Place cells
    Right-click/Ctrl+click - Inspect cell
    Close Window - Quit

    Space - Toggle Running (evolution loop)
    while running:
        U - Toggle Autonomous Network Genes
        A - Toggle Andromida mode (genetic
            birth/death rules)
        C - Change Charge Delta & Gradient
            Threshold
        P - Toggle Activity Prune (charge)
        = - Toggle Prune Logic AND/OR
        O - Toggle Gradient Prune

    T - Toggle Training
        I - Set Learning Rate
        F - Forward Prop direction
        R - Reverse Prop direction
        B - Toggle Back Prop

    G - Toggle Protein/Gene display
    X - Reset network genes and proteins
    D - Toggle Display updates
    V - Cycle statistics views
    H - Scroll through help screens
    W - Reset gradient tracking
    Q - Dump telemetry report
    L - Load layers
    S - Save layers
    M - Load MNIST training data
    E - Enter Parameters
    N - Nuke all cells
    3 - Toggle 3D OpenGL view
    4 - Toggle 3D backprop view
    """

    return jmr_defs, jmr_defs2, conways_defs, how_network_works, forward_pass, how_backprop_works, how_backprop_works2, controls