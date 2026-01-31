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
    genes mutating when it reproduces.
    It is a fraction of a user selected 
    denominator, a higher value meaning a 
    higher chance of mutation.
    For example, if MR = 1, there is a 1% 
    chance (If rate was selected per 100) 
    each gene can mutate at the time a new 
    cell is born.

    A "germ-line" mutation in the BT gene 
    gives a new cell the chance to be BORN 
    even if neither parents had a BT gene 
    value allowing it to be born in that 
    position.
    Not until the next cycle will a cell 
    be checked for OT & IT to determine 
    cycle. So many cells are born that die 
    in the next cycle.

    Mutation (somatic) here means that any 
    gene's value has chance to randomly 
    change to any value between 0 and 7 
    (or whatever range is selected by the 
    user).

    Skip Offset (SO): This gene (gene[9]) 
    determines the number of layers above 
    from which a cell receives its connections.
    For example, if Skip Offset = 1, the cell 
    connects to the immediate previous layer;
    if Skip Offset = 2, then it connects to the 
    cell two layers above.
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
    Each cells is displayed as an 
    informative 3 x 3 arrangment colored 
    blocks.			
                                            
    allele/ value:	OT      IT      BT
                    MR      Charge  Error
                    Bias    Weight  Gradient		

    Overcrowding Tolerance (OT): .genes[0]  
    neighbors <= OT 
    Isolation Tolerance (IT)   : .genes[1]  
    neighbors >= IT 
        
    Birth Threshold (BT)       : .genes[2]  
    neighbors == BT for birth
    Mutation Rate (MR)         : .genes[2]
        
    OT >=  neighbors >= IT 
    OT > IT
    -- Future Genes --
    Weight Gene (WG)           : .genes[4]
    Bias Gene (BI)             : .genes[5]
    Error Gene (ER)            : .genes[6]
    Charge Gene (CG)           : .genes[7]
    Skip Offset:                 .genes[9]
    """

    conways_defs = """
    The original Conway's Game of Life has 
    a fixed set of rules that apply to all 
    cells equally:

    Overcrowding Tolerance (OT): Any live 
    cell with more than three (3) live 
    neighbors dies, as if by overcrowding.
    Isolation Tolerance (IT)   : Any live 
    cell with fewer than two (2) live 
    neighbors dies, as if by isolation/
    lonelyness).

    Birth Threshold (BT): Any cell 
    location with exactly three (3) live 
    neighbors becomes a live cell, as if 
    by reproduction (a new cell is born).
    There is no concept of a Gene or 
    Mutation Rate (MR) in the original 
    game, as rules are static and don't 
    change."
                                
    Overcrowding Tolerance (OT): .genes[0]  
    neighbors <= 3 OT 
    Isolation Tolerance (IT)   : .genes[1]  
    neighbors >= 2 IT 
    Birth Threshold (BT)       : .genes[2]  
    neighbors == 3 BT
        
    OT 3 >=  neighbors >=  2 IT
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
    Forward pass is the process of moving 
    the input data through the network 
    layer by layer:
    Layer N-2:       Layer N-1:       Layer N:

    B11 B12 B13      C11 C12 C13      W11 W12 W13
    B21 B22 B23  --> C21 C22 C23  --> W21 W22 W23
    B31 B32 B33      C31 C32 C33      W31 W32 W33

    1. Each cell in Layer N-2 (Bij) 
    calculates its charge based on its 
    inputs and passes it to the 
    corresponding cells in Layer N-1 (Cij) 
    using the weights stored in Cij.

    2. Each cell in Layer N-1 (Cij) 
    calculates its charge based on the 
    received charges and its weights, then 
    passes it to the corresponding cells 
    in Layer N (Wij) using the weights 
    stored in Wij.

    3. Each cell in Layer N (Wij) 
    calculates its final charge based on 
    the received charges and its weights.
    """

    how_backprop_works = """
    The weights of the current cell are 
    updated based on the error of the 
    current cell and the charge of the 
    cells in the layer above. 

    The `get_upper_layer_cells` function 
    is used in the 
    `update_weights_and_bias` function to 
    get the cells from the layer above the 
    current cell. These cells are used to 
    calculate the gradient for updating 
    the weights of the current cell.

    The `get_layer_below_cells` function 
    is used in the 
    `compute_error_signal_other_layers` 
    function to get the cells from the 
    layer below the current cell. These 
    cells are used to calculate the error 
    signal for the current cell."""

    how_backprop_works2 = """

    Here is a simple text-based figure to 
    illustrate the concept:
    Layer N-1 (Above) Layer N (Current) Layer N+1 (Below)

    Cell A          Cell X                 Cell 1
    Cell B          Cell Y                 Cell 2
    Cell C          Cell Z                 Cell 3

    In this figure, Cells A, B, and C are 
    in the layer above the current layer. 
    Cells X, Y, and Z are in the current 
    layer. Cells 1, 2, and 3 are in the 
    layer below the current layer.

    When updating the weights of Cell X 
    during backpropagation, we use the 
    charges of Cells A, B, and C (from the 
    layer above) and the error of Cell X. 
    This is done in the 
    `update_weights_and_bias` function.

    When calculating the error signal for 
    Cell X, we use the errors and weights 
    of Cells 1, 2, and 3 (from the layer 
    below) and the derivative of the 
    activation function of Cell X. This is 
    done in the 
    `compute_error_signal_other_layers` 
    function.

    Regarding the index, (`weight_index`) 
    in the `update_weights_and_bias` 
    function. The index is calculated 
    based on the relative position (dx, 
    dy) of the cell in the layer above to 
    the current cell. The `reversed_index` 
    is used in the 
    `compute_error_signal_other_layers` 
    function to access the weights of the 
    cells in the layer below.
    """

    backprop_view = """
    3D Backprop View shows network learning:

    Normal Mode (K off):
    - Cells: Cyan = positive charge
            Magenta = negative charge
            Brightness shows strength
    - Lines: Color mix shows weights and 
            error values
    
    Weight Change Mode (K on):
    - Cells: Purple = error magnitude
            Orange = gradient magnitude 
            Blue = background cells
    - Lines: Green = positive weights
            Red = negative weights

    Controls in 3D View:
    Mouse drag - Rotate view
    Mouse wheel - Zoom in/out
    ESC - Exit 3D view
    K - Toggle display mode
    """

    controls = """
    Mouse Down/Mouse Drag - Places a Cell
    Mouse Over - Get Cell Info & Statistics
    Right Mouse Button - Display Cell Info
    Close Window - Quit.
    Space - Toggle Running: {running} 
    while running:
        A - Andromida_mode to allow normal 
        growth rules: {andromida_mode}  
        U - Autonomous mode to allow genes to determine specs
        C - Set Charge AND Gradient Pruning level
        J - Toggle Charge Pruning
        O - Toggle Gradient Pruning
        Protection: {charge_change_protection, delta :.4f}
        P - Toggle Prune: {prune} 
        = - Toggle Prune Logic or/and: {prune_logic}
            
    T - Toggle Training: {training_mode}                                  
        I - Set Learning Rate: {learning_rate :.4f}
        F - Forward Prop, or R - Reverse 
        Forward Prop: {direction_of_charge_flow}
        B - Toggle Back Prop: {back_prop}                                  
        
    G - Toggle Gene or Protein Display: 
    {display}   
    W - Resets gradient, charge forward, charge reverse, error
    X - Reset weights and biases to random 
    values         
    D - Toggle 2D Display on and off
    3 - Toggle 3D View & 2D View
    4 - Backprop View in 3D 0 ESC to exit
    K - Toggle Weight Change View in 3D
    ESC - Escape Backprop View in 3D
    V - View all settings
    H - Scroll through help screens  
    L - Load layers     
    S - Save layers     
    M - load MNEST training data    
    E - Enter Global Paramaters for Network
   
    Arrow Keys - Scroll Pages in File Selection
    """

    return jmr_defs, jmr_defs2, conways_defs, how_network_works, forward_pass, how_backprop_works, how_backprop_works2, controls, backprop_view