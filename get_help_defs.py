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
    HOW THE NETWORK IS DIFFERENT

    In a traditional NN, weights live in
    layer-to-layer matrices owned by the
    network. Here, weights live INSIDE
    each cell's dendrites -- a flat 1D
    array (self.weights) that the cell
    owns, carries, and updates itself.

    The Grid:
    Layer 0    Layer 1..N-2   Layer N-1
    (Input)    (Hidden)       (Desired)
    MNIST  --> cells learn --> labels
    pixels     & evolve       (0-9)

    Training data loaded into layer 0.
    Desired outputs loaded into layer N-1.
    Cells in layers 1 to N-2 learn to
    map inputs to outputs by adjusting
    their own weights via backprop.

    Each cell has its own learning rate
    (gene 9), weight decay (gene 8), and
    activation curve (gene 11). No global
    optimizer -- each cell runs its own
    gradient descent.
    """

    forward_pass = """
    FORWARD PASS (one cell's view)

    Cell X looks at the layer ABOVE it
    and gathers charges from neighbors
    within its dendrite reach:

    Layer above:        This cell:
     [A][B][C]
     [D][E][F] ---->    Cell X
     [G][H][I]          at (x,y)
     3x3 reach
     (gene 4 = 9)

    Step 1: Gather upstream charges
      (skip empty positions)

    Step 2: Weighted sum + bias
      charge = bias
            + A.charge * w[0]
            + B.charge * w[1]
            + ... + I.charge * w[8]

    Step 3: Leaky ReLU activation
      if charge > 0: keep it
      if charge < 0: charge *= slope
        (gene 11 controls slope)

    Step 4: Clip to [-10, 10], store

    Weight index formula:
      idx = (dx+reach)*matrix + (dy+reach)
      (-1,-1)->0  (-1,0)->1  (-1,+1)->2
      ( 0,-1)->3  ( 0,0)->4  ( 0,+1)->5
      (+1,-1)->6  (+1,0)->7  (+1,+1)->8
    """

    how_backprop_works = """
    BACKWARD PASS: 2 JOBS PER CELL

    Job 1: COMPUTE ERROR SIGNAL
    Output layer (N-2):
      error = (my_charge - desired)
              * ReLU_derivative(charge)

    Hidden layers:
      Look at cells BELOW (layer+1).
      They already have their errors.
      error = SUM(cell_below.error
        * cell_below.weights[rev_idx])
        * ReLU_derivative(my_charge)

    Job 2: UPDATE MY WEIGHTS
    Look at cells ABOVE (layer-1).
    For each upstream cell:
      gradient = my_error * upstream.charge
      w[idx] -= lr * gradient + decay * w
      bias   -= lr * my_error

    lr = gene 9, decay = gene 8
    (per-cell in autonomous mode)
    """

    how_backprop_works2 = """
    WHY reversed_index WORKS

    Forward: Cell 5 (below) looks UP
    at Cell X through weight_index:
      idx = (dx+reach)*m + (dy+reach)

    Backward: Cell X looks DOWN at
    Cell 5 through reversed_index:
      rev = len(w) - 1 - idx

    For 3x3 weights:
      fwd 0 <-> rev 8  (-1,-1)<->(+1,+1)
      fwd 1 <-> rev 7  (-1, 0)<->(+1, 0)
      fwd 4 <-> rev 4  ( 0, 0)<->( 0, 0)
      fwd 8 <-> rev 0  (+1,+1)<->(-1,-1)

    This maps (dx,dy) to (-dx,-dy).
    Same as transposing the weight
    matrix -- exactly what standard
    backprop does, but computed from
    the flat 1D array stored in each
    cell's dendrites.

    FULL PICTURE:
    Layer above    This cell    Layer below
     [A][B][C]                   [1][2][3]
     [D][E][F]      Cell X       [4][5][6]
     [G][H][I]     error=?       [7][8][9]
        |                            |
        v                            v
    update X's      compute X's error
    weights using   from below cells'
    A-I charges     errors + weights
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