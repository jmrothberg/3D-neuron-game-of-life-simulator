# JMRs Genetic Game of Neural Network LIFE!! updated with NMIST data July 11
# Fully working forward and reverse flow, fully working backprop Sept 21st
# Sept 21 variable number of layers... and first success of 9 out of 15 test cases with 4 layers.
# Moved mutation outside of Andromedia Sept 25
# 3x 20/20 and 60/60 on MNEST 98 out of 100 on 8 layer network Sept 26
# 100/100 on MNEST 8 layer 49 weights per cell Sept 30, 64% on next 100
# Added info screen Sept 28
# Source .venv/bin/activate when using visual studio code
# Added fashion MNIST data set path for mac Feb 22 2024
# Added platform check for Linux and Mac June 29 2024
# Added settings display "V" sept 7 2024, right button to show cell and statistics
# Added 3D view of network Oct 14 2024
# Added significant charge change flags to cells Oct 15 2024
# Added prediction plot to bottom of screen Oct 17 2024 - a bug at removed it somehow
# Added default values at start so you dont need to enter them every time
# Addeed ability to change weight matrix size with "E" key along with other parameters
# Changed name of LENGTH_OF_AXON to LENGTH_OF_DENDRITE Oct 19 2024, added Genes 4,5,6,7 for Number of weights per cell, bias range, and average weights per cell
# Moved functions to cell class Oct 20 2024
# Added ability to have global weights and biases, both number and range, or go cell autonomous based on genes 4,5,6,7 Oct 22 2024
# Added visual display of GENES and proteins Oct 24 2024
#
# Refactored into neurosim/ package -- see neurosim/ for all code.
# Original monolithic file backed up to 3D_NeuroSim_BACKUP.py

from neurosim.main import main

if __name__ == '__main__':
    main()
