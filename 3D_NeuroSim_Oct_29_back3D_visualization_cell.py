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
# Added gradient threshold for pruning Oct 25 2024
# simplified forward and reverse flow and backpropogation Oct 26 2024 to use cell functions - removed bug using x y from cursor for cell x,y
import sys
import platform as sys_platform
import pygame
import numpy as np
import random
import copy
import pickle
import os
import math
import datetime
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import pygame
from get_help_defs import get_defs
import copy
from OpenGL.GL import *
from OpenGL.GLU import *

system = sys_platform.system()
if system == "Darwin":
    input_path_training_Digits ='/Users/jonathanrothberg/MNIST_5000_0_15_Cells'
    input_path_training_Fashion ='/Users/jonathanrothberg/Fashion_MNIST_5000'
elif system == "Linux":
    # Check for Ubuntu
    if "Ubuntu" in sys_platform.version():
        input_path_training_Digits ='/data/MNIST_5000_0_15_Cells'
        input_path_training_Fashion ='/data/Fashion_MNIST_5000'
    # Check for Raspberry Pi
    elif sys_platform.machine().startswith("armv"):
        pass
        #future where on pi
    else:
        print("Other Linux")
else:
    print("Other OS", system)

fashion = """0 T-shirt/top | 1 Trouser | 2 Pullover | 3 Dress | 4 Coat | 5 Sandal | 6 Shirt | 7 Sneaker | 8 Bag | 9 Ankle boot"""

# Modify these existing variables
EXTENDED_WINDOW_WIDTH = 1508  # Increased to accommodate side panel (1008 + 400)
WINDOW_WIDTH =1008
WINDOW_HEIGHT = 1008
WINDOW_EXTENSION = 100
EXTENDED_WINDOW_HEIGHT = WINDOW_HEIGHT + WINDOW_EXTENSION

# Add these new variables
HELP_PANEL_WIDTH = 500 #
MAIN_SURFACE_WIDTH = EXTENDED_WINDOW_WIDTH - HELP_PANEL_WIDTH

CELL_SIZE = 9  # or 12 with larger window

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
INDIGO = (13, 0, 184)
VIOLET = (179, 0, 255)
PINK = (255,192,203)
LIGHT_GRAY = (200, 200, 200)

COLORS = [INDIGO, ORANGE, PINK, YELLOW, GREEN, RED, VIOLET, BLUE ] # Same length as number of genes 

WIDTH = WINDOW_WIDTH // CELL_SIZE
HEIGHT = WINDOW_HEIGHT // CELL_SIZE

WIDTH = WIDTH // 4  # making a 4 x 4 to represent the 16 layers #28 x 28
HEIGHT = HEIGHT //4  # Each grid is now 1 / 4th. Otherwise it goes off the visible screen

UPPER_ALLELE_LIMIT = 28
ARRAY_LAYERS = 16

show_training_stats = False
training_stats_buffer = {}
stats_update_frequency = 1  # Update stats every 100 training cycles

show_backprop_view = False

class Cell:
    def __init__(self, layer, x, y, weights_per_cell_possible, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate, genes=None):
        global cells
        epsilon = 1e-8
        # Add coordinate validation
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            print(f"Warning: Invalid coordinates: x={x}, y={y}. Must be within grid bounds (0-{WIDTH-1}, 0-{HEIGHT-1})")
            print_to_side_panel(f"Warning: Invalid coordinates: x={x}, y={y}. Must be within grid bounds (0-{WIDTH-1}, 0-{HEIGHT-1})")
            # Set to valid coordinates at edge of grid
            self.x = min(max(0, x), WIDTH-1)
            self.y = min(max(0, y), HEIGHT-1)
        else:
            self.x = x
            self.y = y
            self.layer = layer
            
        #print(f"Initializing cell at {x}, {y}, {layer} {weights_per_cell_possible} {Bias_Range} {Avg_Weights_Cell} {charge_delta}")

        if genes is None: 
            self.initalize_all_genes()
            self.initalize_breeding_genes()
            self.initalize_network_genes(weights_per_cell_possible, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate)
            
            #print(f"Initialized cell at {x}, {y}, {layer} with genes {self.genes}")
            
        else: # genes are passed from parent cell 
            self.genes = genes.copy()
            #print(f"else Initialized cell at {x}, {y}, {layer} with inherited genes {self.genes}")
        
        self.color_genes()
        self.initialize_network_proteins()
        self.initalize_cell_memory()
        self.color_proteins()

    def initalize_cell_memory(self):
        self.forward_charges = []
        self.reverse_charges = []
        self.max_charge_diff_forward = 0
        self.max_charge_diff_reverse = 0
        self.significant_charge_change_forward = False
        self.significant_charge_change_reverse = False
        self.number_of_upper_layer_cells = 0
        self.number_of_lower_layer_cells = 0
        #self.upper_layer_cells = [] # can be recurusive when pickling so not used
        #self.layer_below_cells = [] # can be recurusive when pickling so not used
        self.gradient_history = []
        self.avg_gradient_magnitude = 0
        self.significant_gradient_change = False

    def initalize_all_genes(self): 
        self.genes = [0] * 9
        self.colors = [0] * 9
        self.protein_colors = [0] * 9
        
    def initalize_breeding_genes(self): #these are genes 0 to 3 JUST as we did above same rules as above 
        #randomize only genes 0,1,2,3
        self.genes[0] = random.randint(lower_allele_range, upper_allele_range)
        self.genes[1] = random.randint(lower_allele_range, upper_allele_range)
        self.genes[2] = random.randint(lower_allele_range, upper_allele_range)
        if self.genes[0] < self.genes[1]:  # Swap OT and IT if necessary
            self.genes[0], self.genes[1] = self.genes[1], self.genes[0]
        self.color_genes()

    def initalize_network_genes(self, weights_per_cell_possible, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate): #these are genes 4 to 7
        # Directly assign values to specific gene positions
        global cells
        if not AUTONOMOUS_NETWORK_GENES:
            self.genes[3] = mutation_rate
            self.genes[4] = int(weights_per_cell_possible)
            self.genes[5] = Bias_Range
            self.genes[6] = Avg_Weights_Cell
            self.genes[7] = charge_delta
            self.genes[8] = weight_decay
        else:
            #NEED TO MAKE THIS 9,25,49,81 so random 1,2,3 needs to become 9,25,49...
            self.genes[3] = np.random.randint(0, 100) #somatic mutation rate
            self.genes[4] = (np.random.randint(1, 4) * 2 + 1)**2 #number of dendrites and weights per cell
            self.genes[5] = np.random.choice([0.001, 0.01]) #i want to pick a number either 0.0001 or 0.001 or 0.01
            #is cells global? will need to change if not    
            self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
            self.upper_layer_cells = self.get_upper_layer_cells(cells, self.reach)
            if len(self.upper_layer_cells) > 0:
                self.genes[6] = len(self.upper_layer_cells) #lamarcian gene environment changes it
            else:
                self.genes[6] = Avg_Weights_Cell #if no upper layer cells then it is 1
            self.genes[7] = np.random.uniform(0.000001, 0.01) #charge delta used for threshold for pruning
            self.genes[8] = np.random.uniform(1e-6, 1e-4) #weight decay

        self.color_genes()

    def initialize_network_proteins(self):
        epsilon = 1e-8
        #Set up proteins based on the genes or environment
        self.charge = 0
        self.gradient = 0
        self.error = epsilon # Small not zero so no divison by zero issues
        self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
        #clip the weights at -.8 to .8
        self.weights = np.clip(np.random.randn(int(self.genes[4])) / (np.sqrt(self.genes[6]) + 1e-8), -.8, .8)
        self.bias =  np.random.uniform(0,self.genes[5]) # updated to ONLY be positive Sept 27 and reduced range

    def color_genes(self):
        # Scale genes 0 to 3 directly as they are within the range 0 to 15
        self.colors = [tuple(max(0, min(int(gene * 255 // 15), 255)) for color_component in color) for gene, color in zip(self.genes[:4], COLORS)]

        # Scale genes 4 to 8 based on their specific ranges
        gene_4_scaled = max(0, min(int(((self.genes[4] - 9) / (81 - 9)) * 255), 255))  # Scale 9-81 to 0-255
        gene_5_scaled = max(0, min(int(self.genes[5] * 255), 255))  # Scale 0-1 to 0-255
        gene_6_scaled = max(0, min(int(((self.genes[6] - 5) / (30 - 5)) * 255), 255))  # Scale 5-30 to 0-255
        gene_7_scaled = max(0, min(int(((self.genes[7] - 0.0001) / (0.01 - 0.0001)) * 255), 255))  # Scale 0.0001-0.01 to 0-255
        gene_8_scaled = max(0, min(int(((self.genes[8] - 1e-6) / (1e-4 - 1e-6)) * 255), 255))  # Scale 1e-6 to 1e-4 to 0-255

        # Append the scaled colors for genes 4 to 8
        self.colors.extend([
            (gene_4_scaled, 0, gene_4_scaled),  # Purple color
            (0, gene_5_scaled, gene_5_scaled),  # Cyan color
            (gene_6_scaled, 0, 0),  # Red color
            (0, gene_7_scaled, 0),  # Green color
            (0, 0, gene_8_scaled)   # Blue color
        ])

    def color_proteins(self):
        
        self.protein_colors = [0] * 9

        # Define a small epsilon to prevent division by zero
        epsilon = 1e-8

        # Bias: Scale based on the bias gene range, using epsilon
        bias_range = self.genes[5]
        bias_scaled = max(0, min(int((self.bias / (bias_range + epsilon)) * 255), 255))
        self.protein_colors[0] = (bias_scaled, 0, bias_scaled)  # Magenta color for bias

        # Max Forward Charge Difference: Assuming a typical range, using epsilon
        max_forward_charge_scaled = max(0, min(int((self.max_charge_diff_forward / (1.0 + epsilon)) * 255), 255))  # Adjust 1.0 to expected max
        self.protein_colors[1] = (max_forward_charge_scaled, 0, 0)  # Red color for max forward charge

        # Max Reverse Charge Difference: Assuming a typical range, using epsilon
        max_reverse_charge_scaled = max(0, min(int((self.max_charge_diff_reverse / (1.0 + epsilon)) * 255), 255))  # Adjust 1.0 to expected max
        self.protein_colors[2] = (0, 0, max_reverse_charge_scaled)  # Blue color for max reverse charge

        # Charge: Assuming range 0 to 1
        charge_scaled = max(0, min(int(self.charge * 255), 255))
        self.protein_colors[4] = (charge_scaled, 0, 0)  # Red color for charge

        error_magnitude = abs(self.error + epsilon)
        error_scaled = max(0, min(int(255 * (np.log10(error_magnitude) + 3) / 2), 255))
        self.protein_colors[6] = (0, 0, error_scaled)  # Blue color for error

        # Average Weight: Assuming a typical range, using epsilon
        mean_weight = np.mean(np.abs(self.weights))
        weight_scaled = max(0, min(int((mean_weight / (1.0 + epsilon)) * 255), 255))  # Adjust 1.0 to expected max
        self.protein_colors[7] = (0, weight_scaled, 0)  # Green color for weights

        # Gradient: Assuming a typical range, using epsilon
        gradient = self.charge * self.error
        gradient_scaled = max(0, min(int(np.log(abs(gradient + epsilon) * 55)), 255))
        self.protein_colors[8] = (0, 0, gradient_scaled)  # Blue color for gradient

        self.protein_colors[3] = (0,0,0) #black 

        self.protein_colors[5] = (0,0,0) #black


    def __setstate__(self, state):
            
        self.__dict__.update(state)
        
        if not hasattr(self, 'genes') or len(self.genes) < 9:
            #xxuprint("initalizing genes")
            self.genes = [0] * 9
            if not hasattr(self.genes[3], '__int__'):
                self.initalize_breeding_genes()
            if not hasattr(self.genes[4], '__int__'):
                self.initalize_network_genes()
            if not hasattr(self.genes[5], '__float__'):
                self.genes[5] = Bias_Range
            if not hasattr(self.genes[6], '__float__'):
                self.genes[6] = Avg_Weights_Cell
            if not hasattr(self.genes[7], '__float__'):
                self.genes[7] = charge_delta
            if not hasattr(self.genes[8], '__float__'):
                self.genes[8] = weight_decay

        if not hasattr(self, 'reach'):
            self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
        if not hasattr(self, 'gradient'):
            self.gradient = 0

        if not hasattr(self, 'x'):
            self.x = 0
        if not hasattr(self, 'y'):
            self.y = 0
        if not hasattr(self, 'forward_charges'):
            self.forward_charges = []
        if not hasattr(self, 'reverse_charges'):
            self.reverse_charges = []
        if not hasattr(self, 'max_charge_diff_forward'):
            self.max_charge_diff_forward = 0
        if not hasattr(self, 'max_charge_diff_reverse'):
            self.max_charge_diff_reverse = 0
        if not hasattr(self, 'significant_charge_change_forward'):
            self.significant_charge_change_forward = False
        if not hasattr(self, 'significant_charge_change_reverse'):
            self.significant_charge_change_reverse = False

        if not hasattr(self, 'gradient_history'):
            self.gradient_history = []
        if not hasattr(self, 'avg_gradient_magnitude'):
            self.avg_gradient_magnitude = 0
        if not hasattr(self, 'significant_gradient_change'):
            self.significant_gradient_change = False

        if np.isnan(self.weights).any():
            print(f"Warning: NaN weights detected in cell at ({self.x}, {self.y}, {self.layer})")
            #need to reinitialize weights
            self.weights = np.random.randn(self.genes[4]) / np.sqrt(self.genes[6]) # Good model to use for generating weights, Avg_Weights_Cell is expected, not calculated

        if not hasattr(self, 'protein_colors'):
            self.color_proteins()

    def compute_total_charge(self, upper_layer_cells, reach):
        if AUTONOMOUS_NETWORK_GENES:
            reach = self.reach
            NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
        else:
            WEIGHT_MATRIX = 2 * reach + 1
        charge = 0
        for dx, dy, cell in upper_layer_cells:
            weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
            if 0 <= weight_index < len(self.weights):  # Added lower bound check
                charge += cell.charge * self.weights[weight_index]
        self.charge = np.clip(charge, -10, 10)  # Prevent extreme values
         

    def compute_total_charge_reverse(self, lower_layer_cells, reach):
        if AUTONOMOUS_NETWORK_GENES:
            #reach = self.reach
            #NUMBER_OF_WEIGHTS = self.genes[4]
            #WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
            pass
        else:
            #NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = 2 * reach + 1
        charge = 0
        charge -= self.bias  # Subtract bias for reverse flow
        for dx, dy, cell in lower_layer_cells:
            if AUTONOMOUS_NETWORK_GENES:
                cell_reach = cell.reach
                cell_weight_matrix = int(np.sqrt(cell.genes[4]))
            else:
                cell_reach = reach
                cell_weight_matrix = WEIGHT_MATRIX
            
            # Calculate this cell's position relative to the center of the cell below
            relative_x = -dx + cell_reach
            relative_y = -dy + cell_reach
            
            # Calculate the index in the cell below's weight matrix
            weight_index = relative_y * cell_weight_matrix + relative_x
            
            # Check if the calculated index is within the cell's weight array
            if 0 <= weight_index < len(cell.weights):
                charge += cell.charge * cell.weights[weight_index]
        
        self.charge = np.clip(charge, -10, 10)  # Prevent extreme values
        

    # Comments explaining the changes:
    # 1. We now consider the reach and weight matrix size of each cell in the layer below individually.
    # 2. We calculate the relative position of this cell from the perspective of the cell below.
    # 3. We directly calculate the weight index in the cell below's weight matrix.
    # 4. We perform a bounds check to ensure we're not accessing weights out of range.
    # 5. This approach works correctly whether cells have the same or different numbers of weights.
    # 6. The method now handles both autonomous and non-autonomous modes correctly.

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
    # Increase the leak factor to 0.1 to get stronger gradients
        return np.where(x > 0, 1.0, 0.1)  # Leaky ReLU derivative with larger leak

    def relu_derivativeOLD(self, x):
        return np.greater(x, 0).astype(int)

    def get_upper_layer_cells(self, cells, reach): #reach is the cells reach
        if AUTONOMOUS_NETWORK_GENES:
            reach = self.reach
        upper_layer_cells = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < cells.shape[0] and 0 <= new_y < cells.shape[1] and cells[new_x, new_y, self.layer - 1] is not None:
                    upper_layer_cells.append((dx, dy, cells[new_x, new_y, self.layer - 1]))
        #self.upper_layer_cells = upper_layer_cells
        self.number_of_upper_layer_cells = len(upper_layer_cells)
        #print(f"number of upper layer cells {self.number_of_upper_layer_cells}")
        return upper_layer_cells

    def get_layer_below_cells(self, cells, reach):
        if AUTONOMOUS_NETWORK_GENES:
            reach = self.reach
        layer_below_cells = []
        max_possible_reach = max(cell.reach for row in cells[:, :, self.layer + 1] for cell in row if cell is not None)
        for dx in range(-max_possible_reach, max_possible_reach + 1):
            for dy in range(-max_possible_reach, max_possible_reach + 1):
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < cells.shape[0] and 0 <= new_y < cells.shape[1]:
                    cell_below = cells[new_x, new_y, self.layer + 1]
                    if cell_below is not None:
                        cell_below_reach = cell_below.reach if AUTONOMOUS_NETWORK_GENES else reach
                        if abs(dx) <= cell_below_reach and abs(dy) <= cell_below_reach:
                            layer_below_cells.append((dx, dy, cell_below))
        self.number_of_lower_layer_cells = len(layer_below_cells)
        return layer_below_cells

    def compute_error_signal(self, desired_output=None, connected_cells=None, reach=None):
        if AUTONOMOUS_NETWORK_GENES:
            reach = self.reach
        error_signal = epsilon
        
        # For the layer just before the training data (Num_Layers-2)
        if desired_output is not None:
            error_signal = (self.charge - desired_output) * self.relu_derivative(self.charge)
            self.error = np.clip(error_signal, -10, 10)
            
            #print(f"Layer {self.layer} Error Calculation:")
            #print(f"  charge: {self.charge}, layer {self.layer} x {self.x} y {self.y}, error {self.error}")
            #print(f"  desired: {desired_output}")
            #print(f"  raw diff: {self.charge - desired_output}")
            #print(f"  relu_deriv: {self.relu_derivative(self.charge)}")
            #print(f"  final error: {error_signal}")
            #self.error = error_signal
                
        # For all other hidden layers
        elif connected_cells is not None and reach is not None:
            for dx, dy, cell in connected_cells:
                if AUTONOMOUS_NETWORK_GENES:
                    cell_reach = cell.reach
                    cell_weight_matrix = int(np.sqrt(cell.genes[4]))
                else:
                    cell_reach = reach
                    cell_weight_matrix = 2 * reach + 1
                
                # Calculate the index in the cell below's weight matrix
                weight_index = (dx + cell_reach) * cell_weight_matrix + (dy + cell_reach)
                # Use reversed index like in the working code
                reversed_index = len(cell.weights) - 1 - weight_index
                
                # Check if the calculated index is within the cell's weight array
                if 0 <= reversed_index < len(cell.weights):
                    error_signal += cell.error * cell.weights[reversed_index] * self.relu_derivative(self.charge)
            
            self.error = np.clip(error_signal, -10, 10)
            
        else:
            self.error = epsilon
    
    # 1. For the output layer (when desired_output is provided), the calculation remains the same.
    # 2. For hidden layers, we now use the same approach as in compute_error_signal_other_layers.
    # 3. We calculate the weight index based on the relative position of this cell to the cell in the layer below.
    # 4. We use the reach and weight matrix size of the cell in the layer below, not of the current cell.
    # 5. This method now correctly handles cases where cells have different numbers of weights and reaches.
    # 6. The bounds check ensures we only use valid weight indices.
    

    def update_weights_and_bias(self, connected_cells, learning_rate, reach, weight_decay):
        gradient_clip_range = 1
        if AUTONOMOUS_NETWORK_GENES:
            # Use this cell's reach and weight matrix size
            reach = self.reach
            NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
            # Weight decay constant. Typical values range from 1e-6 to 1e-4.
            weight_decay = self.genes[8]
        else:
            # Use the global reach and weight matrix size
            WEIGHT_MATRIX = 2 * reach + 1
            # Weight decay constant. Typical values range from 1e-6 to 1e-4.
            weight_decay = weight_decay

        if self.error is None:
            self.error = epsilon

        for dx, dy, cell in connected_cells:
            gradient = self.error * cell.charge
            gradient = np.clip(gradient, -gradient_clip_range, gradient_clip_range)
            self.gradient = gradient # set the gradient protein
            self.update_gradient_importance(gradient) # added for prunning and protection
            # Use this cell's weight matrix to index its weights
            weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
            if weight_index < len(self.weights):
                # Apply weight update with weight decay
                self.weights[weight_index] -= learning_rate * gradient + weight_decay * self.weights[weight_index]

        # Update bias (typically, bias is not decayed)
        self.bias -= learning_rate * self.error

    # Comments explaining the changes:
    # 1. We use this cell's reach and weight matrix size when autonomous_network_genes is True.
    #    This is because each cell may have a different reach and number of weights in this mode.
    # 2. We use the global reach when autonomous_network_genes is False, as all cells have the same reach in this case.
    # 3. We always use this cell's weight matrix (WEIGHT_MATRIX) to index its weights, regardless of the mode.
    #    This is because we're updating this cell's weights, so we need to use its own weight matrix structure.
    # 4. The 'reach' used in weight_index calculation is the same as used for WEIGHT_MATRIX,
    #    ensuring consistency in how we map (dx, dy) to weight indices.
    # 5. We don't need to consider the reach of connected cells here, as that was already taken into account
    #    when 'connected_cells' was created in get_upper_layer_cells().

    def __str__(self):
        # Every 7 weights put a new line in the weights string
        weights = [f'{w:.4f}' for w in self.weights]
        weights_with_newlines = []
        for i in range(0, len(weights), 7):
            if i > 0:
                weights_with_newlines.append('\n')
            weights_with_newlines.extend(weights[i:i+7])
        
        weights_str = ', '.join(weights_with_newlines)
        
        bias = f'{self.bias:.4f}'
        error = f'{self.error:.4e}'
        charge = f'{self.charge:.4f}'
        gradient = f'{self.gradient:.4f}'
        reach = f'{self.reach}'

        max_charge_diff_forward = f'{self.max_charge_diff_forward:.4f}'
        significant_charge_change_forward = f'{self.significant_charge_change_forward}'
        max_charge_diff_reverse = f'{self.max_charge_diff_reverse:.4f}'
        significant_charge_change_reverse = f'{self.significant_charge_change_reverse}'

        gradient_average = f'{self.avg_gradient_magnitude:.4f}'
        significant_gradient_change = f'{self.significant_gradient_change}'
        
        return (
            f"Neuron: layer={self.layer} x={self.x} y={self.y}\n"
            f"Genes:\n"
            f"  OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]},\n"
            f"  WG={self.genes[4]}, BR={self.genes[5]}, AW={self.genes[6]}, CD={self.genes[7]}, WD={self.genes[8]}\n"
            f"charge={charge}, error={error}, bias={bias}, gradient={gradient}\n"
            f"max_charge_diff_forward={max_charge_diff_forward}, significant_charge_change_forward={significant_charge_change_forward}\n"
            f"max_charge_diff_reverse={max_charge_diff_reverse}, significant_charge_change_reverse={significant_charge_change_reverse}\n"
            f"gradient_average={gradient_average}, significant_gradient_change={significant_gradient_change}\n"
            f"reach={reach}\n"
            f"weights={weights_str}"
        )

    def remap_weights(self, reach):
        old_matrix = int(np.sqrt(len(self.weights)))
        new_matrix = 2 * reach + 1  # This gives us the actual matrix size

        if old_matrix == new_matrix:
            return  # No change needed if the size is the same

        #if old lenght not a 9,25,49,81 then we just make randome new weights
        if len(self.weights) not in [9,25,49,81,121,169,225,289,361,441,529,625]:
            print ("bad mutant gene repaired")
            print_to_side_panel(f"bad mutant gene repaired {len(self.weights)} fixed to {new_matrix**2}")
            self.weights = np.random.randn(new_matrix**2) / (np.sqrt(self.genes[6]) + 1e-8)
            self.genes[4] = len(self.weights)  # Update the NUMBER_OF_WEIGHTS gene
            return
        
        old_grid = np.reshape(self.weights, (old_matrix, old_matrix))
        new_grid = np.zeros((new_matrix, new_matrix))
        
        old_center = old_matrix // 2
        new_center = new_matrix // 2
        
        # Calculate the range to copy
        copy_range = min(old_matrix, new_matrix)
        start_old = old_center - copy_range // 2
        start_new = new_center - copy_range // 2
        
        # Copy the existing weights
        new_grid[start_new:start_new+copy_range, start_new:start_new+copy_range] = \
            old_grid[start_old:start_old+copy_range, start_old:start_old+copy_range]
        
        # Initialize new weights if expanding
        if new_matrix > old_matrix:
            mask = new_grid == 0
            num_new_weights = np.sum(mask)
            new_weights = np.random.randn(num_new_weights) / (np.sqrt(self.genes[6]) + 1e-8)
            new_grid[mask] = new_weights
        
        self.weights = new_grid.flatten()
        self.genes[4] = len(self.weights)  # Update the NUMBER_OF_WEIGHTS gene

    # Comment explaining the changes:
    # This implementation correctly handles both expansion and contraction of the weight matrix.
    # It preserves the central weights when shrinking and initializes new weights when expanding.
    # The copy_range ensures we always copy the appropriate central portion of the old weights.

    def reset_directional_charge_history(self, direction): #also gradient histories
        if direction == "+++++>>>>>":
            self.forward_charges.clear()
            self.max_charge_diff_forward = 0
            self.significant_charge_change_forward = False
        elif direction == "<<<<<-----":
            self.reverse_charges.clear()
            self.max_charge_diff_reverse = 0
            self.significant_charge_change_reverse = False
        self.gradient_history.clear()
        self.avg_gradient_magnitude = 0
        self.significant_gradient_change = False

    def update_charge(self, new_charge, direction):
        if direction == "forward":
            self.forward_charges.append(new_charge)
            if len(self.forward_charges) > how_much_training_data:
                self.forward_charges.pop(0)
            self.max_charge_diff_forward = max(self.forward_charges) - min(self.forward_charges)
            if self.max_charge_diff_forward > self.genes[7]:
                self.significant_charge_change_forward = True #leaves True until reset
            
        elif direction == "reverse":
            self.reverse_charges.append(new_charge)
            if len(self.reverse_charges) > how_much_training_data:
                self.reverse_charges.pop(0)
            self.max_charge_diff_reverse = max(self.reverse_charges) - min(self.reverse_charges)
            if self.max_charge_diff_reverse > self.genes[7]:
                self.significant_charge_change_reverse = True #leaves True until reset
        self.charge = new_charge

    def reset_gradient_change(self):
        self.significant_gradient_change = False
        self.gradient_history.clear()
        self.gradient = 0
        self.error = epsilon

    def update_gradient_importance(self, new_gradient):
        self.gradient_history.append(abs(new_gradient))
        
        if len(self.gradient_history) > how_much_training_data:
            self.gradient_history.pop(0)
        
        self.avg_gradient_magnitude = np.mean(self.gradient_history)
        if self.avg_gradient_magnitude > gradient_threshold:  #stays true until reset.
            self.significant_gradient_change = True
        

def create_icon(filename, cell_size=1):  #Maked it easy to visualize the state of the network
    with open(filename, 'rb') as f:
        cells = pickle.load(f)

    icon = np.ones((112*cell_size, 112*cell_size))
    number_of_layers = cells.shape[2]
    for k in range(number_of_layers):
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                if cells[i, j, k] is not None:
                    icon_x = (k // 4) * 28*cell_size + j*cell_size
                    icon_y = (k % 4) * 28*cell_size + i*cell_size
                    icon[icon_x:icon_x+cell_size, icon_y:icon_y+cell_size] = 0

    # Convert to 8-bit grayscale
    img = Image.fromarray((icon * 255).astype(np.uint8))
    # Convert the image to RGB format
    img = img.convert("RGB")
    # Convert the image to a pygame surface and return it
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)


def parse_file_name(file_name):
    print (file_name)
    try:
        # Split the file name on the underscore character
        parts = file_name.split('_')

        # The NUM_LAYERS and NUMBER_OF_WEIGHTS are the second and third elements in the list
        num_layers = int(parts[1])
        num_weights = int(parts[2])
    except Exception as e:
        print("Error in parse_file_name", e)
        print (file_name)
        try:
            num_layers = int(pygame_input("Enter number of layers: "),8)
            num_weights = int(pygame_input("Enter number of weights: "),9)
        except:
            num_layers = 8
            num_weights = 25
            
    return num_layers, num_weights
    

def save_file(tag):
    timestamp = datetime.datetime.now().strftime("%d-%H-%M-%S")
    file_dir = "./saved_states/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filename = f"{file_dir}sim_{Num_Layers}_{NUMBER_OF_WEIGHTS}_-{timestamp}_{bingo_count}_{how_much_training_data}-{start_index}{tag}"
    pkl_filename = f"{filename}.pkl"
    txt_filename = f"{filename}.txt"
    icon_filename = f"{filename}.png"
    
    # Save simulation state to pkl file .pkl
    with open(pkl_filename, 'wb') as f:
        pickle.dump(cells, f)
    print(f"Simulation state saved to {pkl_filename}!")
    print_to_side_panel(f"Simulation state saved to {pkl_filename}!")

    # Create icon for each file .png
    icon  = create_icon(pkl_filename, 3)
    pygame.image.save(icon, icon_filename)
    print(f"Icon saved to {icon_filename}!")
    print_to_side_panel(f"Icon saved to {icon_filename}!")

    # Save variables to txt file .txt
    with open(txt_filename, 'w') as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"Cell Settings:\n")
        f.write(f"mutation_rate: {mutation_rate}\n")
        f.write(f"lower_allele_range: {lower_allele_range}\n")
        f.write(f"upper_allele_range: {upper_allele_range}\n")
        f.write(f"Simulation Settings:\n")
        f.write(f"NUM_LAYERS: {Num_Layers}\n")
        f.write(f"LENGTH_OF_DENDRITE: {LENGTH_OF_DENDRITE}\n")
        f.write(f"Bias_Range: {Bias_Range}\n")
        f.write(f"Weight Range based on estimate Avg weights/cell: {Avg_Weights_Cell}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Charge change for prunning and charge protection (delta): {charge_delta}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"how_much_training_data: {how_much_training_data}\n")
        f.write(f"start_index: {start_index}\n")
        f.write(f"training_cycles: {training_cycles}\n")
        f.write(f"Bingo count: {bingo_count}\n")
        f.write(f"max_bingo_count: {max_bingo_count}\n")
        f.write(f"Loss: {running_avg_loss}\n")
        f.write(f"Number of cells: {total_cells}\n")
        f.write(f"Number of weights: {total_weights_list[0]}\n")
        f.write(f"Weight/Cell: {total_weights_list[0]/(total_cells+epsilon):.2f}\n")
    print(f"Variables saved to {txt_filename}!")        
    print_to_side_panel(f"Variables saved to {txt_filename}!")

def update_cell_coordinates():
    global cells
    for layer in range(cells.shape[2]):
        for x in range(cells.shape[0]):
            for y in range(cells.shape[1]):
                if cells[x, y, layer] is not None:
                    if not hasattr(cells[x, y, layer], 'x') or cells[x, y, layer].x == 0:
                        cells[x, y, layer].x = x
                    if not hasattr(cells[x, y, layer], 'y') or cells[x, y, layer].y == 0:
                        cells[x, y, layer].y = y

def load_file():
    global cells, Num_Layers, NUMBER_OF_WEIGHTS, LENGTH_OF_DENDRITE, WEIGHT_MATRIX, not_saved_yet, bingo_count, max_bingo_count, total_loss, total_predictions, running_avg_loss, training_cycles, points  
    file_dir = "./saved_states/"
    # Check if directory exists, if not, create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_list = [f for f in os.listdir(file_dir) if f.startswith("sim") and f.endswith(".pkl")]
    
    if len(file_list) == 0:
        print_to_side_panel("\nNo saved simulation states found!")
        print("No saved simulation states found!")
        return

    growthsurface.fill(WHITE)
    bottom_caption_surface.fill(BLACK)
    screen.blit(growthsurface, (0, 0))
    pygame.display.flip()
    page = 0
    cell_size = 1
    no_file_selected = True
    new_page = True
    NUM_LAY = [0]*1000
    num_wei = [0]*1000

    while no_file_selected:
        if new_page:
            growthsurface.fill(WHITE)
            bottom_caption_surface.fill(BLACK)
            draw_grid()
            for i, file_name in enumerate(file_list[page*16:(page+1)*16]):
                real_i = i + page*16
                # Create icon for each file
                icon = create_icon(os.path.join(file_dir, file_name), cell_size)
                # Display the icon
                icon_x = (i % 4) * 252*cell_size + 75
                icon_y = (i // 4) * 252*cell_size + 50
                screen.blit(icon, (icon_x, icon_y))

                # Clear the area where file names will be displayed
                pygame.draw.rect(screen, WHITE, (icon_x - 65, icon_y + 112*cell_size - 5, 230, 70))

                # Parse the file name to get the NUM_LAYERS and NUMBER_OF_WEIGHTS
                NUM_LAY[real_i], num_wei[real_i] = parse_file_name(file_name)
                
                # Truncate file name if it's too long
                max_width = 220
                truncated_name = file_name
                while font_directory.size(truncated_name)[0] > max_width:
                    truncated_name = truncated_name[:-1]
                if truncated_name != file_name:
                    truncated_name += "..."

                # Render the file name and info
                text1 = font_directory.render(truncated_name, True, (0, 0, 0))
                text2 = font_directory.render(f"Layers: {NUM_LAY[real_i]} | Weights {num_wei[real_i]}", True, (0, 0, 0))
                
                # Display the file name and info
                screen.blit(text1, (icon_x - 65, icon_y + 112*cell_size + 20))  # Moved up by 100
                screen.blit(text2, (icon_x - 65, icon_y + 112*cell_size + 45))  # Moved up by 100

            # Render instructions
            text_surface1 = font.render("Click on a file to load it", True, WHITE)
            bottom_caption_surface.blit(text_surface1, (50, 10))
            text_surface2 = font.render("Use Left & Right Arrow keys to scroll pages", True, WHITE)
            bottom_caption_surface.blit(text_surface2, (50, 40))
            #how to escape without loading a file
            text_surface3 = font.render("Press ESC to return", True, WHITE)
            bottom_caption_surface.blit(text_surface3, (50, 70))

            screen.blit(bottom_caption_surface, (0, EXTENDED_WINDOW_HEIGHT - WINDOW_EXTENSION))
            pygame.display.flip()

        new_page = False

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i in range(16):
                    icon_x = (i % 4) * 252*cell_size + 75
                    icon_y = (i // 4) * 252*cell_size + 50
                    if icon_x <= mouse_x < icon_x + 252*cell_size and icon_y <= mouse_y < icon_y + 252*cell_size:
                        selection = i + page*16
                        if selection < len(file_list):
                            file_path = os.path.join(file_dir, file_list[selection-1])
                            try:
                                with open(file_path, 'rb') as f:
                                    cells = pickle.load(f)

                                # Set the number of layers and number of weights
                                Num_Layers = NUM_LAY[selection]
                                NUMBER_OF_WEIGHTS = num_wei[selection]
                                WEIGHT_MATRIX = int(math.sqrt(NUMBER_OF_WEIGHTS))
                                LENGTH_OF_DENDRITE = int((WEIGHT_MATRIX - 1) / 2)

                                print(f"Simulation state loaded from {file_path} | NUM_LAYER: {Num_Layers} | Number of weights per cell {NUMBER_OF_WEIGHTS} | Length of dendrite {LENGTH_OF_DENDRITE} |  weight_matrix {WEIGHT_MATRIX}")
                                print_to_side_panel(f"Simulation state loaded from {file_path} | NUM_LAYER: {Num_Layers} | Number of weights per cell {NUMBER_OF_WEIGHTS} | Length of dendrite {LENGTH_OF_DENDRITE} |  weight_matrix {WEIGHT_MATRIX}")
                                not_saved_yet = True
                                max_bingo_count = 0
                                bingo_count = 0
                                total_loss, total_predictions, running_avg_loss = 0, 0, 0
                                training_cycles = 0
                                update_cell_coordinates()
                                print ("Cells coordinates updated")
                                print_to_side_panel("Cell coordinates updated")
                                
                                points = []
                                no_file_selected = False
                                return
                            except Exception as e:
                                print("An error occurred: " + str(e))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    #clear the screen
                    growthsurface.fill(WHITE)
                    screen.blit(growthsurface, (0, 0))
                    pygame.display.flip()
                    page = min(page + 1, (len(file_list)) // 16)
                    new_page = True
                elif event.key == pygame.K_LEFT:
                    
                    page = max(page - 1, 0)
                    new_page = True
                elif event.key == pygame.K_ESCAPE:
                    no_file_selected = False    
                    return

        clock.tick(60)
    growthsurface.fill(WHITE)
    bottom_caption_surface.fill(BLACK)
    screen.blit(growthsurface, (0, 0))
    pygame.display.flip()
    return

# Loading Training or Test data from .pkl files. These are NIST datasets converted to cell data by helper program
def load_layers(input_path_training, image_index): 
    global cells  
    file_path = os.path.join(input_path_training, f'simulation_state_layers_0_and_15_image_{image_index}.pkl')
    with open(file_path, 'rb') as f:
        cells[:,:,0], cells[:,:,Num_Layers-1] = pickle.load(f) 

    # Rotate and flip only layer 0 and Num_Layers-1 # MNEST stored in way that images where inverted and rotated
    cells[:,:,0] = np.flip(np.rot90(cells[:,:,0], 3), 1)
    cells[:,:,Num_Layers-1] = np.flip(np.rot90(cells[:,:,Num_Layers-1], 3), 1)

# Function to append training data set and draw it on the screen as loaded
def load_training_data(input_path_training, start_index=0):
    global training_data_layer_0, training_data_NUM_LAYER_MINUS_1, cells # Define these as global variables
    training_data_layer_0 = []  # Initialize as empty list
    training_data_NUM_LAYER_MINUS_1 = []  # Initialize as empty list while called 15 it really is just the answer of the training data layer
    print ("Loading training data from: ",input_path_training )
    print_to_side_panel(f"Loading training data from: {input_path_training}")

    for k in range(start_index, start_index + how_much_training_data):  # For the first 1000 images
        try:
            load_layers(input_path_training, k)
            training_data_layer_0.append(copy.deepcopy(cells[:,:,0]))  # Append data to list not just pickle reference!
            training_data_NUM_LAYER_MINUS_1.append(copy.deepcopy(cells[:,:,Num_Layers -1]))  # Append data to list not just pickle reference!   
            growthsurface.fill(WHITE)
            draw_cells()
            draw_grid()
            screen.blit(growthsurface, (0, 0))
            pygame.display.flip()

        except Exception as e:
            print(f"Error occurred while loading data at index {k}: {e}")
            print_to_side_panel(f"Error occurred while loading data at index {k}: {e}")
            continue

# Control Loop for loading training data
def load_training_data_main():
    global training_data_loaded, how_much_training_data, start_index, total_weights_list, not_saved_yet, max_bingo_count, bingo_count, total_loss, total_predictions, running_avg_loss, training_cycles, points
    default_training_data = how_much_training_data
    default_start_index = start_index
    try:     
        which_data_set = pygame_input ("Enter which data set to load (MNIST, M for MNIST or F for Fashion MNIST): ", "M")
        if which_data_set.lower() == "f":
            training_data = input_path_training_Fashion
            print (fashion)
        else:
            training_data = input_path_training_Digits
        how_much_training_data = int(pygame_input("Enter training set size (20, 1 to 1000): ", 20))
        start_index = int(pygame_input("Start index (0, 0 to 999): ", 0))
    except ValueError:
        print("Invalid input. No New Data loaded")
        print_to_side_panel("Invalid input. No New Data loaded")
    else:
        # Ensure start_index is within the valid range
        if start_index + how_much_training_data > 5000:
            how_much_training_data = default_training_data
            start_index = default_start_index
            print("Total can't exceed 5000. Returning to defaults", how_much_training_data, start_index)
            print_to_side_panel("Total can't exceed 5000. Returning to defaults", how_much_training_data, start_index)
        else:
            try:
                print ("Loading training data from ", training_data, " starting at ", start_index, " for ", how_much_training_data, " sets")
                load_training_data(training_data, start_index)
                total_weights_list = np.zeros(how_much_training_data)
                training_data_loaded = True
                not_saved_yet = True
                max_bingo_count = 0
                bingo_count = 0
                total_loss, total_predictions, running_avg_loss = 0, 0, 0
                training_cycles = 0
                points = []
            except Exception as e:
                print("An error occurred while loading the training data: ", + str(e))
                print_to_side_panel("An error occurred while loading the training data: ", + str(e))

def convert_x_y_to_index(x, y):
     # Determine which of the 16 screen segments (layer) mouse is in
    layer_x = x // (WINDOW_WIDTH // 4)
    layer_y = y // (WINDOW_HEIGHT // 4)
    layer = layer_x + layer_y * 4

    # Adjust x and y for the layer before calculating cell position
    adjusted_x = x - layer_x * (WINDOW_WIDTH // 4)
    adjusted_y = y - layer_y * (WINDOW_HEIGHT // 4)
    
    cell_x = min(adjusted_x // CELL_SIZE, WIDTH - 1)
    cell_y = min(adjusted_y // CELL_SIZE, HEIGHT - 1)
    return cell_x, cell_y, layer


def draw_grid():
    # Draw vertical grid lines
    for x in range(0, WINDOW_WIDTH, WINDOW_WIDTH // 4):
        pygame.draw.line(growthsurface, BLACK, (x, 0), (x, WINDOW_HEIGHT), 2)
    # Draw horizontal grid lines
    for y in range(0, WINDOW_HEIGHT, WINDOW_HEIGHT // 4):
        pygame.draw.line(growthsurface, BLACK, (0, y), (WINDOW_WIDTH, y), 2)
 
 
def update_cell_types(cells):  
    start_layer = 1
    stop_layer = Num_Layers-1
    cell_types = {}
    number_of_genes_to_use = 4 # Only consider the first 3 genes when creating the cell_type tuple
    for layer in range(start_layer, stop_layer):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cell_type = tuple(cells[x, y, layer].genes[:number_of_genes_to_use]) #
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
    return (cell_types)
 

def update_phenotype_cell_types(cells):
    # Collect cell properties in separate lists for overall standard deviation calculation
    phenotype_cell_types = {}
    all_charges = []
    all_biases = []
    all_errors = []
    all_weights = []
    all_gradients = []
    all_max_charge_diff_forward = []
    all_max_charge_diff_reverse = []
    
    count_pos = 0  # Count of cells included in the calculation
    total_cells = 0  # Total number of cells that are not None
    start_layer = 1
    stop_layer = Num_Layers-1
    for layer in range(start_layer, stop_layer):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                
                # Exclude cells with all weights being 0 or epsilon, and cells with error being 0 or epsilon
                if not all(weight in [0, epsilon] for weight in cells[x, y, layer].weights):
                    cell = cells[x, y, layer]
                    all_charges.append(cell.charge)
                    all_biases.append(cell.bias)
                    all_errors.append(cell.error)
                    all_weights.append(np.mean(cell.weights))  # Average weight per cell
                    all_gradients.append(cell.gradient)
                    all_max_charge_diff_forward.append(cell.max_charge_diff_forward)
                    all_max_charge_diff_reverse.append(cell.max_charge_diff_reverse)
                    count_pos += 1  # Increase the count


    if count_pos > 0:
        return (0, 0, {})
    
    # Calculate mean and std for each property across all cells
    charge_mean, charge_std = (np.mean(all_charges), np.std(all_charges)) if all_charges else (0, 0)
    bias_mean, bias_std = (np.mean(all_biases), np.std(all_biases)) if all_biases else (0, 0)
    error_mean, error_std = (np.mean(all_errors), np.std(all_errors)) if all_errors else (0, 0)
    weights_mean, weights_std = (np.mean(all_weights), np.std(all_weights)) if all_weights else (0, 0)
    gradient_mean, gradient_std = (np.mean(all_gradients), np.std(all_gradients)) if all_gradients else (0, 0)
    max_charge_diff_forward_mean, max_charge_diff_forward_std = (np.mean(all_max_charge_diff_forward), np.std(all_max_charge_diff_forward)) if all_max_charge_diff_forward else (0, 0)
    max_charge_diff_reverse_mean, max_charge_diff_reverse_std = (np.mean(all_max_charge_diff_reverse), np.std(all_max_charge_diff_reverse)) if all_max_charge_diff_reverse else (0, 0)

    # Loop again to assign phenotypes based on the overall properties' distributions
    for layer in range(start_layer, stop_layer):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cell = cells[x, y, layer]
                weights_avg = np.mean(cell.weights)
                phenotype = (
                    'charge:' + ('+' if cell.charge > charge_mean else '-') + str(int(abs((cell.charge - charge_mean) / (charge_std + epsilon)))),
                    'bias:' + ('+' if cell.bias > bias_mean else '-') + str(int(abs((cell.bias - bias_mean) / (bias_std + epsilon)))),
                    'error:' + ('+' if cell.error > error_mean else '-') + str(int(abs((cell.error - error_mean) / (error_std + epsilon)))),
                    'weights:' + ('+' if weights_avg > weights_mean else '-') + str(int(abs((weights_avg - weights_mean) / (weights_std + epsilon)))),
                    'gradient:' + ('+' if cell.gradient > gradient_mean else '-') + str(int(abs((cell.gradient - gradient_mean) / (gradient_std + epsilon)))),
                    'max_fwd:' + ('+' if cell.max_charge_diff_forward > max_charge_diff_forward_mean else '-') + str(int(abs((cell.max_charge_diff_forward - max_charge_diff_forward_mean) / (max_charge_diff_forward_std + epsilon)))),
                    'max_rev:' + ('+' if cell.max_charge_diff_reverse > max_charge_diff_reverse_mean else '-') + str(int(abs((cell.max_charge_diff_reverse - max_charge_diff_reverse_mean) / (max_charge_diff_reverse_std + epsilon))))
                )
                if phenotype not in phenotype_cell_types:
                    phenotype_cell_types[phenotype] = []
                phenotype_cell_types[phenotype].append(cell)
    return (count_pos, total_cells, phenotype_cell_types)



def display_statistics(cell_types):
    sorted_types = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]
    statistics = ["[OT IT BT] CH WG ER BI: Overcrowding Isolation Birth"] + [f"Type: {cell_type} | Count: {count}" for cell_type, count in sorted_types]
    
    # Calculate average mutation frequency
    total_mutation_freq = sum(int(cell_type[3]) for cell_type in cell_types.keys())
    avg_mutation_freq = total_mutation_freq / len(cell_types) if cell_types else 0
    
    statistics.append(f"\nAverage Mutation Frequency: {avg_mutation_freq:.2f}")
    
    print_to_side_panel("\nDisplaying statistics for the top 5 cell types:", 10)
    print_to_side_panel("\n".join(statistics))


def display_phenotype_statistics(phenotype_cell_types):
    count_pos, total_cells, phenotype_dict = phenotype_cell_types
    sorted_types = sorted([(phenotype, len(cells), np.mean([cell.charge for cell in cells]), np.mean([cell.bias for cell in cells]), np.mean([cell.error for cell in cells]), np.mean([np.mean(cell.weights) for cell in cells]), np.mean([cell.gradient for cell in cells]), np.mean([cell.max_charge_diff_forward for cell in cells]), np.mean([cell.max_charge_diff_reverse for cell in cells])) for phenotype, cells in phenotype_dict.items()], key=lambda x: x[1], reverse=True)[:5]
    statistics = ["Phenotype: Avg Charge | Avg Bias | Avg Error | Avg Weights | Avg Gradient | Avg Max Fwd | Avg Max Rev"] + [f"{phenotype} | Count: {count:4} \nCharge: {avg_charge:.4f} | Bias: {avg_bias:+.4f} | Error: {avg_error:.4e} | Weights: {avg_weights:.4f} | Gradient: {avg_gradient:.4e} | MaxFwd: {avg_max_fwd:.4f} | MaxRev: {avg_max_rev:.4f}" for phenotype, count, avg_charge, avg_bias, avg_error, avg_weights, avg_gradient, avg_max_fwd, avg_max_rev in sorted_types]
    print_to_side_panel("\nDisplaying statistics for the top 5 phenotypes:")
    print_to_side_panel("\n".join(statistics))


def display_max_charge_diff(N=5):
    output = []
    charge_diffs = []

    for layer in range(1, Num_Layers - 1):  # Skip input and output layers
        for x, y in np.ndindex(cells.shape[:2]):
            cell = cells[x, y, layer]
            if cell is not None:
                if direction_of_charge_flow == "+++++>>>>>":
                    charge_diff = cell.max_charge_diff_forward
                else:
                    charge_diff = cell.max_charge_diff_reverse
                charge_diffs.append((x, y, layer, charge_diff))

    # Sort and get top N
    top_N_charge_diff = sorted(charge_diffs, key=lambda x: x[3], reverse=True)[:N]

    for x, y, layer, charge_diff in top_N_charge_diff:
        output.append(f"Cell at ({x}, {y}, {layer}) - Charge difference: {charge_diff:.2f}")

    print_to_side_panel("\nMax charge difference for top 5 cells:")
    print_to_side_panel("\n".join(output))

def display_averages():
    output = []
    try:
        avg_gradient = np.zeros(Num_Layers)
        max_gradient = np.zeros(Num_Layers)
        min_gradient = np.zeros(Num_Layers)
        avg_error = np.zeros(Num_Layers)
        avg_charge = np.zeros(Num_Layers)
        avg_weights = np.zeros(Num_Layers)
        avg_weights_per_cell = np.zeros(Num_Layers)
        
        total_cells = 0
        active_cells = 0
        
        for layer in range(1, Num_Layers-1):
            cells_in_layer = [cells[x, y, layer] for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, layer] is not None]
            total_cells += len(cells_in_layer)
            active_cells_in_layer = [cell for cell in cells_in_layer if not all(weight in [0, epsilon] for weight in cell.weights)]
            active_cells += len(active_cells_in_layer)
            
            if active_cells_in_layer:
                gradients = [cell.gradient for cell in active_cells_in_layer]
                avg_gradient[layer] = np.mean(gradients)
                max_gradient[layer] = np.max(gradients)
                min_gradient[layer] = np.min(gradients)
                avg_error[layer] = np.mean([abs(cell.error) for cell in active_cells_in_layer])
                avg_charge[layer] = np.mean([abs(cell.charge) for cell in active_cells_in_layer])
                avg_weights[layer] = np.mean([np.mean(abs(cell.weights)) for cell in active_cells_in_layer])
                avg_weights_per_cell[layer] = np.mean([len(cell.weights) for cell in active_cells_in_layer])

        output.append(f"Predictions: {total_predictions} | Average Loss: {running_avg_loss:.4e}")
        output.append(f"Active Cells: {active_cells}/{total_cells}")
        
        output.append("\nLayer-wise Statistics:")
        for layer in range(1, Num_Layers-1):
            output.append(f"Layer {layer}:")
            output.append(f"  Gradient: {avg_gradient[layer]:.4e}, Max: {max_gradient[layer]:.4e}, Min: {min_gradient[layer]:.4e}")
            output.append(f"  Error: {avg_error[layer]:.4e}")
            output.append(f"  Charge: {avg_charge[layer]:.4f}")
            output.append(f"  Weights: {avg_weights[layer]:.4f}")
            output.append(f"  Avg Weights/Cell: {avg_weights_per_cell[layer]:.2f}")
        
        output.append("\nNetwork-wide (excluding input and output layers):")
        output.append(f"  Avg Gradient: {np.mean(avg_gradient[1:-1]):.4e}, Max: {np.max(max_gradient[1:-1]):.4e}, Min: {np.min(min_gradient[1:-1]):.4e}")
        output.append(f"  Avg Error: {np.mean(avg_error[1:-1]):.4e}")
        output.append(f"  Avg Charge: {np.mean(avg_charge[1:-1]):.4f}")
        output.append(f"  Avg Weights: {np.mean(avg_weights[1:-1]):.4f}")
        output.append(f"  Avg Weights/Cell: {np.mean(avg_weights_per_cell[1:-1]):.2f}")

    except Exception as e:
        output.append(f"Error in averages: {e}")
    
    print_to_side_panel("\nNetwork Statistics:", 10)
    print_to_side_panel("\n".join(output), 50)


def prediction_plot():
    global points
    offset = 756
    # Scale the y-coordinate
    y = WINDOW_EXTENSION - (running_avg_loss / 10) * WINDOW_EXTENSION

    # Add the point to the list
    points.append((len(points)+offset, y))

    # If there are more than offset points, remove the first one
    if len(points) > 700:
        points.pop(0)
        # Adjust the x values of the points to create a scrolling effect
        points = [(x-1, y) for x, y in points]

    # Draw the points on the bottom_caption_surface
    for point in points:
        pygame.draw.circle(bottom_caption_surface, BLUE, point, 4)


def prediction_to_actual():
    global cells, bingo_count, max_bingo_count, total_loss, total_predictions, running_avg_loss
    # Extract the one-hot encoded label from the middle of last layer (layer 14 default)
    one_hot_label_guess = np.array([cell.charge if cell is not None else 0 for cell in cells[9:19, 14,  Num_Layers-2]]) 
    # Extract the one-hot encoded label from the middle of training data layer (layer 15 default)
    one_hot_label_actual = np.array([cell.charge if cell is not None else 0 for cell in cells[9:19, 14,  Num_Layers-1 ]]) 

    # Calculate the loss using categorical cross-entropy
    predictions_clipped = np.clip(one_hot_label_guess, 1e-7, 1 - 1e-7)  # To avoid log(0)
    loss = -np.sum(one_hot_label_actual * np.log(predictions_clipped))
    
    # Get the position of the '1' in the one-hot encoded labels
    pos_guess = np.argmax(one_hot_label_guess)
    pos_actual = np.argmax(one_hot_label_actual)        
    
    # Update the total loss and total predictions
    total_loss += loss
    total_predictions += 1

    # Calculate the running average of the loss
    running_avg_loss = total_loss / total_predictions

    # Check if the positions match
    if pos_guess == pos_actual: 
        bingo_count += 1  # Increment the bingo count
        if bingo_count > max_bingo_count: max_bingo_count = bingo_count 


def update_training_stats():
    global training_stats_buffer, cells, total_weights_list
    
    # Calculate average weights per cell
    total_cells = sum(1 for layer in range(1, Num_Layers-1) for cell in cells[:,:,layer].flatten() if cell is not None)
    total_weights = sum(len(cell.weights) for layer in range(1, Num_Layers-1) for cell in cells[:,:,layer].flatten() if cell is not None)
    avg_weights_per_cell = total_weights / (total_cells + epsilon)
    
    # Get top 5 phenotypes
    count_pos, total_cells, phenotype_cell_types = update_phenotype_cell_types(cells)
    #top_phenotypes = sorted(phenotype_cell_types.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    
    # Get top 5 genotypes
    cell_types = update_cell_types(cells)
    top_genotypes = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]

    # New metrics
    total_neurons = sum(1 for layer in range(1, Num_Layers-1) for cell in cells[:,:,layer].flatten() if cell is not None)
    active_neurons = sum(1 for layer in range(1, Num_Layers-1) for cell in cells[:,:,layer].flatten() if cell is not None and np.any(cell.weights != 0))
    avg_connections = sum(np.count_nonzero(cell.weights) for layer in range(1, Num_Layers-1) for cell in cells[:,:,layer].flatten() if cell is not None) / (total_neurons + epsilon)
    
    gradients = [abs(cell.gradient) for layer in range(1, Num_Layers-1) 
                for cell in cells[:,:,layer].flatten() if cell is not None]
    max_gradient = max(gradients) if gradients else 0
    avg_gradient = np.mean(gradients) if gradients else 0
    
    training_stats_buffer = {
        "Accuracy": f"{bingo_count}/{how_much_training_data} (Max: {max_bingo_count})",
        "Avg Loss": f"{running_avg_loss:.4e}",
        "Total Predictions": total_predictions,
        "Avg Weights/Cell": f"{avg_weights_per_cell:.2f}",
        "Total Weights": f"{total_weights:.2f}",
        "Total Cells": total_cells,
        "Active Cells": f"{count_pos}/{total_cells}",
        "Top Genotypes": top_genotypes,
        "Active Neurons": f"{active_neurons}/{total_neurons}",
        "Avg Connections/Neuron": f"{avg_connections:.2f}",
        "Max Gradient": f"{max_gradient:.4e}",
        "Avg Gradient": f"{avg_gradient:.4e}",
    }

def display_training_stats():
    stats_text = "Training Statistics:\n"
    stats_text += f"Accuracy: {training_stats_buffer['Accuracy']}\n"
    stats_text += f"Avg Loss: {training_stats_buffer['Avg Loss']}\n"
    stats_text += f"Total Predictions: {training_stats_buffer['Total Predictions']}\n"
    stats_text += f"Avg Weights/Cell: {training_stats_buffer['Avg Weights/Cell']}\n"
    stats_text += f"Total Weights: {training_stats_buffer['Total Weights']}\n"
    stats_text += f"Total Cells: {training_stats_buffer['Total Cells']}\n"
    stats_text += f"Active Cells: {training_stats_buffer['Active Cells']}\n"
    stats_text += "Top Genotypes:\n"
    for genotype, count in training_stats_buffer['Top Genotypes']:
        stats_text += f"  {genotype}: {count}\n"
    stats_text += f"Active Neurons: {training_stats_buffer['Active Neurons']}\n"
    stats_text += f"Avg Connections/Neuron: {training_stats_buffer['Avg Connections/Neuron']}\n"
    stats_text += f"Max Gradient: {training_stats_buffer['Max Gradient']}\n"
    stats_text += f"Avg Gradient: {training_stats_buffer['Avg Gradient']}\n"
    print_to_side_panel(stats_text, 10)  # Adjusted position for more content


def draw_cells(): # removed clipping from drawing :) Sept 22 note
    if display == "genes":
        for layer in range(Num_Layers):
            for (x, y) in np.ndindex(cells.shape[:2]):
                if cells[x, y, layer] is not None:
                    cells[x, y, layer].color_genes()
                    for i in range(9):  # 3 x 3 of all 9 genes
                        
                        color = cells[x, y, layer].colors[i] 

                        pygame.draw.rect(
                            growthsurface,
                            color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3,
                                CELL_SIZE // 3
                            )
                        )
    else: #proteins
        for layer in range(Num_Layers):
            for (x, y) in np.ndindex(cells.shape[:2]):
                if cells[x, y, layer] is not None:
                    cells[x, y, layer].color_proteins()
                    for i in range(9):  

                        color = cells[x, y, layer].protein_colors[i]                      

                        pygame.draw.rect(
                            growthsurface,
                            color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3,
                                CELL_SIZE // 3
                            )
                        )      


#This is were cells are born, die, and mutate. It has two swithes around pruning - actively killing cells or protecting based  on charge_change lists
def update_cells():  # This is where cells are created, mutated, and killed based on Prune, and Charge as well as genes
    global cells, NUMBER_OF_WEIGHTS
    start_layer = 1 
    stop_layer = Num_Layers-1   
    for layer in range(start_layer, stop_layer):
        # Loop over all positions in the 2D grid within each layer
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cell = cells[x, y, layer]
                
                if gradient_prune == True:
                    if not cell.significant_gradient_change:
                        cells[x, y, layer] = None
                        #if you kill a cell you need to go to the next in the four loop
                        continue
                if prune == True:  # Pruning happens before potential for cell birth
                   
                        if prune_logic == "AND":
                            if not (cell.significant_charge_change_forward and cell.significant_charge_change_reverse):
                                cells[x, y, layer] = None
                        elif prune_logic == "OR":
                            if not (cell.significant_charge_change_forward or cell.significant_charge_change_reverse):
                                cells[x, y, layer] = None      

            # If the current cell is alive somatic mutation possible # this now happens even when not in andromida mode as long as its running.
            if cells[x, y, layer] is not None: # need to check if it was not killed by pruning
                cell = cells[x, y, layer]
                
                # Gene 3 represents mutation rate per 100,000 cycles (range 1-100)
                mutation_chance = cell.genes[3] / 100000  # Convert to probability

                if random.random() < mutation_chance:
                    if random.random() < 0.5:
                        cell.initalize_breeding_genes() # mutate by resetting genes
                        #print("breeding")
                    else:
                        cell.initalize_network_genes(NUMBER_OF_WEIGHTS, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate)
                        #print("network")
                
                #cells[x, y, layer].genes = [gene if random.randint(1, som_mut_rate) > cells[x, y, layer].genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in cells[x, y, layer].genes]
                
            if andromida_mode == True:
                # Define the offsets for neighboring positions in 3D space THESE Are physical neighbors NOT dendrite connections. 
                if layer == 1:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [0, 1] if dx != 0 or dy != 0 or dz != 0]
                elif layer == Num_Layers-2:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [0, -1] if dx != 0 or dy != 0 or dz != 0]
                else:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if dx != 0 or dy != 0 or dz != 0]

                # Generate a list of the living neighboring cells without wrap-around
                alive_neighbors = [
                    cells[x+dx, y+dy, layer+dz]
                    for dx, dy, dz in neighbors
                    if (
                        0 <= x+dx < cells.shape[0]
                        and 0 <= y+dy < cells.shape[1]
                        and 0 <= layer+dz < cells.shape[2]
                        and cells[x+dx, y+dy, layer+dz] is not None
                    )
                ]
                # Calculate the number of living neighboring cells
                num_alive = len(alive_neighbors)

                # If the current cell is Empty and there are living neighbors check if it should be born
                if cells[x, y, layer] is None and alive_neighbors:
                    # Get potential parents from alive neighbors
                    potential_parents = alive_neighbors

                    # If there are at least two potential parents then recombination and mutation and germ-line mutation can occur
                    if len(potential_parents) >= 2:
                        # Randomly choose two parents
                        parent1, parent2 = random.sample(potential_parents, 2)
                        # Combine and potentially mutate the parents' genes to create the new cell's genes
                        new_genes = [random.choice([parent1.genes[i], parent2.genes[i]]) for i in range(9)] 

                    # If there is only one potential parent - asexual reproduction and germ-line mutatino can occur
                    elif potential_parents:
                        # Choose the single parent
                        parent1 = potential_parents[0]  # Choose the first parent
                        # Copy and potentially mutate the parent's genes to create the new cell's genes
                        new_genes = parent1.genes
                          # If the number of alive neighbors is exactly equal to the birth threshold of the new cell, create a new cell
                    
                    if num_alive == new_genes[2]:
                        cells[x, y, layer] = Cell(layer, x, y, NUMBER_OF_WEIGHTS, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate, new_genes)
                        cell = cells[x, y, layer]
                        
                        mutation_chance = cell.genes[3] / 1000  # Convert to probability        

                        if random.random() < mutation_chance:
                            cell.initalize_breeding_genes()

                        if random.random() < mutation_chance:
                            cell.initalize_network_genes(NUMBER_OF_WEIGHTS, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate)
                        #new_genes = [gene if random.randint(1, germ_mut_rate) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]          

                #NEW June 13, protect cells that are connected in network and have charge_change  
                if cells[x, y, layer] is not None:
                    cell = cells[x, y, layer] # could be a new cell! but i assigened it above this is extra
                    # Exclude cells from the list of cells that had their charge changes from genetic selection
                    if charge_change_protection == True: #alays true
                        # Only include significant_gradient_change if gradient_prune is on
                        protection_condition = cell.significant_charge_change_forward or cell.significant_charge_change_reverse
                        #if gradient_prune:
                        protection_condition = protection_condition or cell.significant_gradient_change
                    
                        if not protection_condition:  #This is now a property of the cell! 
                            # Kill the cell if it is overcrowded or isolated based on genes
                            if (num_alive <= cell.genes[1] or num_alive >= cell.genes[0]):
                                cells[x, y, layer] = None             
                    
                    else:
                        # Kill the cell if it is overcrowded or isolated based on genes
                        if (num_alive <= cell.genes[1] or num_alive >= cell.genes[0]):
                            cells[x, y, layer] = None    


def reset_all_gradient_changes(): #  reset manually both forward and reverse and gradiants W key whip clean
    for layer in range(1, Num_Layers - 1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cells[x, y, layer].reset_gradient_change()
                cells[x, y, layer].significant_charge_change_forward = False
                cells[x, y, layer].significant_charge_change_reverse = False
        
def reset_directional_charge_history(direction): #each time you use the same direction you need to reset the history
    for layer in range(1, Num_Layers - 1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cells[x, y, layer].reset_directional_charge_history(direction)

def training(learning_rate, reach, weight_decay):
    global bingo_count, cells, total_weights_list, total_weights
    bingo_count = 0  # Reset the bingo count before each iteration of training or testing data
    set_size = how_much_training_data
    epochs = 1  # Number of epochs to train for

    # Reset charge history only for the current direction of charge flow
    #reset_directional_charge_history(direction_of_charge_flow)
    
    for i in range(set_size):
        cells[:,:,0] = training_data_layer_0[i]
        cells[:,:,Num_Layers-1] = training_data_NUM_LAYER_MINUS_1[i]
        total_weights = sum(np.sum(np.abs(cell.weights)) for layer in range(1, Num_Layers-1) for cell in cells[:,:,layer].flatten() if cell is not None)
        total_weights_list[i] = total_weights

        train_network(epochs, learning_rate, reach, weight_decay)

        if display_updating:
           if show_3d_view:
                render_3d_network()
           else:    
                growthsurface.fill(WHITE)
                draw_cells()
                draw_grid()
                screen.blit(growthsurface, (0, 0))
           pygame.display.flip()


def train_network(epochs, learning_rate,reach, weight_decay):
    for epoch in range(epochs):  # Not implemented yet
        
        if direction_of_charge_flow == "+++++>>>>>":
            forward_propagation(reach)  # Regular forward propogation
            if back_prop:
                back_propagation(learning_rate,reach, weight_decay)  # Compute the back propagation
            prediction_to_actual()
        if direction_of_charge_flow == "<<<<<-----":
            reverse_forward_propagation(reach)  # Compute the reverse_forward propagation
        

def forward_propagation(reach):
    # Iterate through all layers except the output layer
    for layer in range(1, Num_Layers - 1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                # Let the cell handle all charge computations and updates
                upper_layer_cells = cells[x, y, layer].get_upper_layer_cells(cells, reach)
                cells[x, y, layer].compute_total_charge(upper_layer_cells, reach)
                cells[x, y, layer].update_charge(cells[x, y, layer].charge, "forward")


def reverse_forward_propagation(reach):
    # Iterate from last hidden layer backwards
    for layer in range(Num_Layers - 2, 0, -1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                if layer == Num_Layers - 2:  # Special handling for second-to-last layer
                    if cells[x, y, Num_Layers - 1] is not None:
                        cells[x, y, layer].update_charge(cells[x, y, Num_Layers - 1].charge, "reverse")
                else:
                    # Let the cell handle charge computations for all other layers
                    lower_layer_cells = cells[x, y, layer].get_layer_below_cells(cells, reach)
                    cells[x, y, layer].compute_total_charge_reverse(lower_layer_cells, reach)
                    cells[x, y, layer].update_charge(cells[x, y, layer].charge, "reverse")


def back_propagation(learning_rate, reach, weight_decay):
    # Iterate through each layer in reverse order (starting from second-to-last layer)
    for layer in range(Num_Layers - 2, 0, -1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                # Compute error signal based on layer position
                if layer == Num_Layers - 2:
                    # For the layer just before output, use the desired output directly
                    if cells[x, y, Num_Layers - 1] is not None:
                        desired_output = cells[x, y, Num_Layers - 1].charge
                        cells[x, y, layer].compute_error_signal(desired_output=desired_output)
                else:
                    # For all other layers, compute error based on connected cells in layer below
                    layer_below_cells = cells[x, y, layer].get_layer_below_cells(cells, reach)
                    cells[x, y, layer].compute_error_signal(connected_cells=layer_below_cells, reach=reach)

                # Update weights and bias using cells from layer above
                upper_layer_cells = cells[x, y, layer].get_upper_layer_cells(cells, reach)
                cells[x, y, layer].update_weights_and_bias(upper_layer_cells, learning_rate, reach, weight_decay)


            # Visualize the current state of backpropagation if in backprop view mode
                if show_3d_view and show_backprop_view:
                    render_3d_backprop(current_layer=layer, current_pos=(x,y))
                    pygame.display.flip()
                    pygame.time.wait(50)  # Add small delay to make the visualization visible


def render_help_text(surface, text, font, color, x, y, line_height):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        text_surface = font.render(line.strip(), True, color)
        surface.blit(text_surface, (x, y + i * line_height))


def print_to_side_panel(text, position=None):
    global help_surface, side_panel_text
    #if 3D view is on then dont print to side panel
    if show_3d_view:
        return
    if position is None:
        # Existing scrolling behavior
        lines = text.split('\n')
        side_panel_text.extend(lines)
        side_panel_text = side_panel_text[-50:]
        
        help_surface.fill(WHITE)
        y = 10
        for line in side_panel_text:
            text_surface = font_small.render(line, True, BLACK)
            help_surface.blit(text_surface, (10, y))
            y += 20
    else:
       #reset the side panel text
       side_panel_text = [] 
       help_surface.fill(WHITE, (0, position, HELP_PANEL_WIDTH, WINDOW_HEIGHT))
       lines = text.split('\n')
       for i, line in enumerate(lines):
            text_surface = font_small.render(line, True, BLACK)
            help_surface.blit(text_surface, (10, position + i * 20))
    
    # Update the screen
    screen.blit(help_surface, (MAIN_SURFACE_WIDTH+3, 0))
    pygame.display.update(pygame.Rect(MAIN_SURFACE_WIDTH+3, 0, HELP_PANEL_WIDTH, WINDOW_HEIGHT))


def pygame_input(prompt, default_value=None):
    input_box = pygame.Rect(50, 10, 200, 32)  # Adjusted y-position
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_active
    active = True
    text = str(default_value) if default_value is not None else ""
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return default_value
           
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        # Clear the bottom caption surface
        bottom_caption_surface.fill(BLACK)

        # Render the prompt and input text
        txt_surface = font.render(prompt + text, True, color)
        width = max(200, txt_surface.get_width()+10)
        input_box.w = width

        # Draw the input box and text on the bottom_caption_surface
        pygame.draw.rect(bottom_caption_surface, color, input_box, 2)
        bottom_caption_surface.blit(txt_surface, (input_box.x+5, input_box.y+5))

        # Update the screen with the bottom_caption_surface
        screen.blit(bottom_caption_surface, (0, EXTENDED_WINDOW_HEIGHT - WINDOW_EXTENSION))
        pygame.display.flip()

        clock.tick(30)
    return text if text else default_value

def get_user_input(prompt, default_value):
    try:
        user_input = pygame_input(prompt, default_value)
        if user_input == "":
            return default_value
        else:
            return int(user_input)
    except:
        print("Invalid input")
        print_to_side_panel("Invalid input")
        return default_value

def get_user_input_float(prompt, default_value):
    try:
        user_input = pygame_input(prompt, default_value)
        if user_input == "":
            return default_value
        else:
            return float(user_input)
    except:
        print("Invalid input")
        print_to_side_panel("Invalid input")
        return default_value


def get_input_values():
    global Num_Layers, LENGTH_OF_DENDRITE, mutation_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Avg_Weights_Cell, Bias_Range, learning_rate, charge_delta, weight_decay, gradient_threshold

    Num_Layers = get_user_input(f"Enter the number of layers ({Num_Layers}, Range 4 to 16): ", Num_Layers)
    if Num_Layers < 3 or Num_Layers > 16: Num_Layers = 8
    LENGTH_OF_DENDRITE = get_user_input(f"Enter the length of the dendrite ({LENGTH_OF_DENDRITE}, Range 1 to 4): ", LENGTH_OF_DENDRITE)
    mutation_rate = get_user_input(f"Enter the Mutation rates per 100,000 cycles ({mutation_rate}): ", mutation_rate)
    lower_allele_range = get_user_input(f"Enter the lower value for alleles ({lower_allele_range}): ", lower_allele_range)
    upper_allele_range = get_user_input(f"Enter the upper value for alleles ({upper_allele_range}): ", upper_allele_range)
    #weight change threshold is not used for anything yet
    weight_change_threshold = get_user_input_float(f"Enter the weight change threshold for reporting ({weight_change_threshold:.3f}): ", weight_change_threshold)
    Avg_Weights_Cell = get_user_input(f"Enter estimate number neurons per cell to set weight distribution ({Avg_Weights_Cell}, 5 to 50): ", Avg_Weights_Cell)
    weight_decay = get_user_input_float(f"Enter the weight decay ({weight_decay:.3f}, Range 1e-6 to 1e-4): ", weight_decay)
    Bias_Range = get_user_input_float(f"Enter the bias random distribution positive range ({Bias_Range:.3f}, Range 0 to .1): ", Bias_Range)
    learning_rate = get_user_input_float(f"Enter the learning rate ({learning_rate:.4f}, Range .01 to .001): ", learning_rate)
    charge_delta = get_user_input_float(f"Enter the charge delta to protect or prune a cell in protect/prune mode ({charge_delta:.3f}): ", charge_delta)
    gradient_threshold = get_user_input_float(f"Enter the gradient threshold to protect or prune a cell in protect/prune mode ({gradient_threshold:.3f}): ", gradient_threshold)
    return mutation_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers, LENGTH_OF_DENDRITE, learning_rate, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, gradient_threshold


def get_all_settings():
    settings = f"""Current Settings:
    Num_Layers: {Num_Layers}
    LENGTH_OF_DENDRITE: {LENGTH_OF_DENDRITE}
    WEIGHT_MATRIX: {WEIGHT_MATRIX}
    NUMBER_OF_WEIGHTS: {NUMBER_OF_WEIGHTS}
    Mutation Rate Per 10,000 Cycles: {mutation_rate}
    Lower Allele Range: {lower_allele_range}
    Upper Allele Range: {upper_allele_range}
    Weight Change Threshold: {weight_change_threshold} 
    Learning Rate: {learning_rate}
    Bias Range: {Bias_Range}
    Avg Weights per Cell: {Avg_Weights_Cell}
    Weight Decay: {weight_decay}
    Charge Delta: {charge_delta}
    Gradient Clip Range: {gradient_clip_range}
    Training Data Size: {how_much_training_data}
    Start Index: {start_index}
    Display Mode: {display}
    Direction of Charge Flow: {direction_of_charge_flow}
    Prune Logic: {prune_logic}
    """
    return settings

# Add this function to set up the 3D view
def setup_3d_view():
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

# Add this function to render the 3D network

def render_3d_network():
    global rotation_x, rotation_y, zoom
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, zoom)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)
    
    #with cells_lock:
    # Draw neurons
    glPointSize(10)
    glBegin(GL_POINTS)
    for layer in range(Num_Layers):
        z = layer * 2
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if cells[x, y, layer] is not None:
                    charge_intensity = cells[x, y, layer].charge
                    charge_intensity_c = min(int(abs(charge_intensity) * 255), 255)
                    color = (charge_intensity_c/255, 0, 0)
                    glColor3f(*color)
                    glVertex3f(x - WIDTH/2, y - HEIGHT/2, z - Num_Layers)
    glEnd()


    # Draw connections
   
    glLineWidth(1)  # Thin lines for connections
    glBegin(GL_LINES)
    for layer in range(1, Num_Layers):  # Start from layer 1, as layer 0 doesn't have inputs
        z = layer * 2
        prev_z = (layer - 1) * 2
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if cells[x, y, layer] is not None:
                    if layer == Num_Layers - 1:
                        # For the last layer, draw a single straight connection
                        if cells[x, y, layer - 1] is not None:
                            glColor3f(0, 1, 0)  # Green for the direct connection
                            glVertex3f(x - WIDTH/2, y - HEIGHT/2, z - Num_Layers)
                            glVertex3f(x - WIDTH/2, y - HEIGHT/2, prev_z - Num_Layers)
                    else:
                        # For other layers, draw dendritic connections
                        reach = LENGTH_OF_DENDRITE
                        for dx in range(-reach, reach + 1):
                            for dy in range(-reach, reach + 1):
                                if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT:
                                    if cells[x + dx, y + dy, layer - 1] is not None:
                                        weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
                                        if weight_index < len(cells[x, y, layer].weights):
                                            weight = cells[x, y, layer].weights[weight_index]
                                            # Color code based on weight
                                            if weight > 0:
                                                glColor3f(0, min(weight, 1), 0)  # Green for positive weights
                                            else:
                                                glColor3f(min(-weight, 1), 0, 0)  # Red for negative weights
                                            glVertex3f(x - WIDTH/2, y - HEIGHT/2, z - Num_Layers)
                                            glVertex3f(x + dx - WIDTH/2, y + dy - HEIGHT/2, prev_z - Num_Layers)
    glEnd()

def render_3d_backprop(current_layer, current_pos):
    try:
        global rotation_x, rotation_y, zoom
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        
        # Draw all neurons
        glPointSize(10)
        glBegin(GL_POINTS)
        try:
            for layer in range(Num_Layers):
                z = layer * 2
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        if cells[x, y, layer] is not None:
                            try:
                                if layer == current_layer and (x,y) == current_pos:
                                    # Current neuron: yellow
                                    glColor3f(1.0, 1.0, 0.0)
                                elif layer == current_layer + 1:
                                    # Layer above (where error comes from): purple with error intensity
                                    error_intensity = min(max(abs(cells[x, y, layer].error), 0.0), 1.0)
                                    glColor3f(error_intensity, 0, error_intensity)  # Purple for error signal
                                elif layer == current_layer - 1:
                                    # Layer below (where gradients flow): orange with gradient intensity
                                    gradient_intensity = min(max(abs(cells[x, y, layer].gradient), 0.0), 1.0)
                                    glColor3f(1.0, gradient_intensity, 0.0)  # Orange for gradients
                                else:
                                    # Other layers: dim blue
                                    glColor3f(0.2, 0.2, 0.4)
                                
                                # Modify the color based on charge intensity
                                charge_intensity = min(abs(cells[x, y, layer].charge), 1.0)
                                # Get current color and blend with charge intensity
                                r, g, b = 0.2 + 0.8*charge_intensity, 0.2 + 0.8*charge_intensity, 0.4
                                glColor3f(r, g, b)
                                glVertex3f(float(x - WIDTH/2), float(y - HEIGHT/2), float(z - Num_Layers))
                            except Exception as e:
                                print(f"Error drawing neuron at {x},{y},{layer}: {e}")
                                continue
        finally:
            glEnd()

        # Draw connections for current neuron
        x, y = current_pos
        if cells[x, y, current_layer] is not None:
            current_cell = cells[x, y, current_layer]
            z = current_layer * 2
            reach = LENGTH_OF_DENDRITE
            
            glLineWidth(2)
            glBegin(GL_LINES)
            try:
                # Draw connections to layer above (error signals)
                if current_layer < Num_Layers - 1:
                    for dx in range(-reach, reach + 1):
                        for dy in range(-reach, reach + 1):
                            try:
                                next_x, next_y = x + dx, y + dy
                                if 0 <= next_x < WIDTH and 0 <= next_y < HEIGHT:
                                    next_cell = cells[next_x, next_y, current_layer + 1]
                                    if next_cell is not None:
                                        weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
                                        if weight_index < len(next_cell.weights):
                                            weight = next_cell.weights[weight_index]
                                            error = next_cell.error
                                            
                                            # Visualize both weight and error influence
                                            weight_intensity = min(max(abs(weight), 0.0), 1.0)
                                            error_intensity = min(max(abs(error), 0.0), 1.0)
                                            
                                            # Brighter lines indicate stronger error signals
                                            if weight > 0:
                                                glColor3f(0, weight_intensity, error_intensity)  # Green-blue for positive weights
                                            else:
                                                glColor3f(weight_intensity, 0, error_intensity)  # Red-blue for negative weights
                                            
                                            glVertex3f(float(x - WIDTH/2), float(y - HEIGHT/2), float(z - Num_Layers))
                                            glVertex3f(float(next_x - WIDTH/2), float(next_y - HEIGHT/2), float((current_layer + 1) * 2 - Num_Layers))
                            except Exception as e:
                                print(f"Error drawing connection at dx={dx}, dy={dy}: {e}")
                                continue

                # Draw gradient flow indicators
                try:
                    if current_layer > 0:
                        # Remove glLineWidth change and just use color intensity
                        gradient_intensity = min(max(abs(current_cell.gradient), 0.0), 1.0)
                        glColor3f(1.0, 0.5 * gradient_intensity, 0.0)  # Orange with varying intensity
                        
                        # Draw small arrows indicating gradient flow direction
                        arrow_length = 0.5
                        glVertex3f(float(x - WIDTH/2), float(y - HEIGHT/2), float(z - Num_Layers))
                        glVertex3f(float(x - WIDTH/2), float(y - HEIGHT/2), float(z - Num_Layers - arrow_length))
                except Exception as e:
                    print(f"Error drawing gradient flow: {e}")
            finally:
                glEnd()

    except Exception as e:
        print(f"Error in render_3d_backprop: {e}")

def render_3d_networkNOCONNECTIONS():
    global rotation_x, rotation_y, zoom
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, zoom)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)

    glPointSize(5)  # Set the size of points
    glBegin(GL_POINTS)
    for layer in range(Num_Layers):
        z = layer * 2  # Spread out layers more
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if cells[x, y, layer] is not None:
                    glColor3fv([c/255 for c in cells[x, y, layer].colors[0][:3]])
                    glVertex3f(x - WIDTH/2, y - HEIGHT/2, z - Num_Layers)
    glEnd()

    pygame.display.flip()


# Add this function to set default values
def set_default_values():
    
    mutation_rate = 10
    lower_allele_range = 2
    upper_allele_range = 15
    weight_change_threshold = 0.005
    Num_Layers = 8
    LENGTH_OF_DENDRITE = 1
    learning_rate = 0.01
    Bias_Range = 0.01
    Avg_Weights_Cell = 5
    weight_decay = 1e-6
    charge_delta = 0.001 
    gradient_threshold = 0.0000001
    WEIGHT_MATRIX = 2*LENGTH_OF_DENDRITE + 1
    NUMBER_OF_WEIGHTS = WEIGHT_MATRIX*WEIGHT_MATRIX
    return mutation_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers, LENGTH_OF_DENDRITE, learning_rate, Bias_Range, Avg_Weights_Cell, weight_decay, charge_delta, WEIGHT_MATRIX, NUMBER_OF_WEIGHTS, gradient_threshold

pygame.init()
screen = pygame.display.set_mode((EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("JMR's Game of Life with Genetics & Nerual Network")
font = pygame.font.SysFont(None, 24)
font_small = pygame.font.SysFont(None, 20)
font_directory = pygame.font.SysFont(None, 16)

# Create surfaces

bottom_caption_surface = pygame.Surface((EXTENDED_WINDOW_WIDTH, WINDOW_EXTENSION))
help_surface = pygame.Surface((HELP_PANEL_WIDTH-2, WINDOW_HEIGHT))

help_surface = help_surface.convert()
bottom_caption_surface = bottom_caption_surface.convert()

# Create a subsurface that is the original size minus 100 pixels from the height
subsurface_rect = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
growthsurface = screen.subsurface(subsurface_rect)
growthsurface = growthsurface.convert()
growthsurface.fill(WHITE)

# Initialize cells with all None - cut in WIDTH and HEIGH in 4th since you now have 4 x 4 array
cells = np.full((WIDTH, HEIGHT, ARRAY_LAYERS), None, dtype=object) # change to array size. NEED for screen to work on inputs or errors,

# Dictionary to store cell types and their frequencies
cell_types = {}
phenotype_cell_types = {}
max_charge_diff = {}

# Initialize a list to keep track of the top 10 weight changes
top_weight_changes = []

# Mouse-button-up event tracking (to prevent continuous drawing while holding the button)
mouse_up = True

# Initialize side panel text
side_panel_text = []

# Dictionary of help screens
jmr_defs, jmr_defs2, conways_defs, how_network_works, forward_pass, how_backprop_works, how_backprop_works2, controls = get_defs()
help_screen = {"jmr_defs": jmr_defs, "jmr_defs2": jmr_defs2, "conways_defs": conways_defs, "how_network_works": how_network_works, "forward_pass": forward_pass, "how_backprop_works": how_backprop_works, "how_backprop_works2": how_backprop_works2, "controls": controls}

print(jmr_defs)
print(conways_defs)
print (how_network_works)

# Display the help screen
help_text = help_screen["controls"]
help_surface.fill(LIGHT_GRAY)
render_help_text(help_surface, help_text, font_small, BLACK, 10, 10, 20)
screen.blit(help_surface, (MAIN_SURFACE_WIDTH+3, 0))

# Draw the grid lines on the main surface
growthsurface.fill(WHITE)
draw_grid()
screen.blit(growthsurface, (0, 0))
pygame.display.flip()

# Call the function to get the input values and set up your run
mutation_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers, LENGTH_OF_DENDRITE, learning_rate, Bias_Range,Avg_Weights_Cell, weight_decay, charge_delta, WEIGHT_MATRIX, NUMBER_OF_WEIGHTS, gradient_threshold = set_default_values()
print(mutation_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers, LENGTH_OF_DENDRITE, learning_rate, Bias_Range,Avg_Weights_Cell, weight_decay, charge_delta, WEIGHT_MATRIX, NUMBER_OF_WEIGHTS, gradient_threshold)

print (f"numer of weights: {NUMBER_OF_WEIGHTS}, weight matrix: {WEIGHT_MATRIX}, length of dendrite: {LENGTH_OF_DENDRITE}, NUM_LAYERS: {Num_Layers}")

# Flag to indicate whether the simulation has started
display_updating = True
timing = False
running = False
prune = False
gradient_prune = False
training_mode = False
andromida_mode = False
charge_change_protection = True # remove as switch to protect/prune just always on, just make it super high to not protect
back_prop = False
training_data_loaded = False
not_saved_yet = True
simulating = True #Controls main Loop

prune_logic = "OR"  # initial value
display = "proteins"
direction_of_charge_flow = "+++++>>>>>"

epochs = 1
batch_size = 1 # may reset charge between batches

epsilon = 1e-8 # Small value to prevent division by zero but not too small. tryint go figure out why sometimes it is not reset.
bingo_count = 0
max_bingo_count = 0
total_cells = 0
total_loss, total_predictions, running_avg_loss = 0, 0, 0
points = []
gradient_clip_range = 1 # 
start_index = 0
how_much_training_data = 20
total_weights = 0
total_weights_list = np.zeros(how_much_training_data)
training_cycles = 0
current_index = 0

# Add these global variables
show_3d_view = False
rotation_angle = 0
# Add these global variables
rotation_x = 0
rotation_y = 0
zoom = -15

display_set = 0

# Add this to your global variables
AUTONOMOUS_NETWORK_GENES = False

# Simulation loop
start_time = time.time()  # Record the start time

while simulating:

     # Autosave perfect networks - add once per training set or model loaded:)
    if max_bingo_count == how_much_training_data and not_saved_yet: # auto save for perfect networks
        save_file("-perfect")
        not_saved_yet = False    
    for event in pygame.event.get():

        # Add this to toggle help display
        if event.type == pygame.KEYDOWN and event.key == pygame.K_h and not show_3d_view:
            #cycle through the help-screens using the dictionary keys
            #get the keys from the dictionary
            help_keys = list(help_screen.keys())
            #increment the index
            current_index += 1
            #if the index is greater than the number of keys, reset to 0
            if current_index >= len(help_keys):
                current_index = 0
            #get the new help screen
            help = help_keys[current_index]
            #set the help text to the new screen
            help_text = help_screen[help]
            help_surface.fill(LIGHT_GRAY)
            render_help_text(help_surface, help_text, font_small, BLACK, 10, 10, 20)
            screen.blit(help_surface, (MAIN_SURFACE_WIDTH+3, 0)) 

        # Add this to your event handling loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_u:
                AUTONOMOUS_NETWORK_GENES = not AUTONOMOUS_NETWORK_GENES
                print(f"Autonomous network genes mode: {AUTONOMOUS_NETWORK_GENES}")
                print_to_side_panel(f"Autonomous network genes mode: {AUTONOMOUS_NETWORK_GENES}")

        # Press SPACE to start and stop the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = not running
                print ("Running=", running)
                
        # Press P to start and stop Prunning of all not changing charge by threshold
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                prune = not prune
                print ("Prune=", prune)    

        # Press O to start and stop Gradient Pruning
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                gradient_prune = not gradient_prune
                print ("Gradient Prune=", gradient_prune)    

        # Press = to toggle prune_logic from OR to AND
        if event.type  == pygame.KEYDOWN:
            if event.key == pygame.K_EQUALS: # Press = to toggle prune_logic
                if prune_logic == "OR":
                    prune_logic = "AND"
                else:
                    prune_logic = "OR"
                print ("Prune Logic=", prune_logic)

        # Press C to  Charge Change Protection      
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                #charge_change_protection = not charge_change_protection
                #print ("Charge_change_protection=", charge_change_protection)    

                if charge_change_protection: # make it always true
                    old_delta = charge_delta
                    old_gradient_threshold = gradient_threshold
                    try:
                        charge_delta = get_user_input_float(f"Enter the charge delta to protect or prune a cell in protect/prune mode ({charge_delta}): ", charge_delta) 
                        gradient_threshold = get_user_input_float(f"Enter the gradient threshold to protect or prune a cell in protect/prune mode ({gradient_threshold}): ", gradient_threshold)  
                        print_to_side_panel(f"Charge delta: {charge_delta}, Gradient threshold: {gradient_threshold}")  
                    
                    except Exception:
                        charge_delta = old_delta
                        gradient_threshold = old_gradient_threshold 
                        print_to_side_panel("Invalid entry. Using old delta value: ", charge_delta)
                        print_to_side_panel("Invalid entry. Using old gradient threshold value: ", gradient_threshold)

        # Press D to Toggle Display
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                display_updating = not display_updating
                print ("Display_updating=", display_updating)           
                
        # Close window to quit
        if event.type == pygame.QUIT:
            try:
                confirm_quit = input("Are you sure you want to quit? (y/n): ")
                if confirm_quit.lower() == 'y':
                    simulating = False
                    continue
                else:
                    print("Quit cancelled.")
            except Exception as e:
                print("An error occurred: ", e)    

        # Press N to Nuke all cells
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                try:
                    confirm_quit = pygame_input("Are you sure you want to Nuke all cells? (y/n): ")
                    
                    if confirm_quit.lower() == 'y':
                        for i in range(1, Num_Layers-1):
                            cells[:,:,i] = None
                            not_saved_yet = True
                            max_bingo_count = 0
                            bingo_count = 0
                            
                            changed_cells_forward = set()
                            total_loss, total_predictions, running_avg_loss = 0, 0, 0
                            training_cycles = 0
                            points = []
                            print_to_side_panel("All cells nuked.")
                    else:
                        print("Nuclear option cancelled.")
                        print_to_side_panel("Nuclear option cancelled.")
                except Exception as e:
                    print("An error occurred: ",e)   
                    print_to_side_panel(f"An error occurred: {e}")

        # Press 'I' to change the Iq or learning rate
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                old_learning_rate = learning_rate
                try:
                    learning_rate = get_user_input_float(f"Enter the learning rate (.01 to .001 default: {old_learning_rate:.4f}): ", old_learning_rate)
                    print_to_side_panel(f"Learning rate updated to: {learning_rate:.4f}")
                except ValueError:
                    print_to_side_panel(f"Invalid input. Reverting to old learning rate: {old_learning_rate:.4f}")
                    learning_rate = old_learning_rate  
       
        #Press E Enter Paramater Resets # just resets the global variables no change to cells
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                old_LENGTH_OF_DENDRITE = LENGTH_OF_DENDRITE
                # Call the function to get the input values and set up your run
                mutation_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers, LENGTH_OF_DENDRITE, learning_rate, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, gradient_threshold = get_input_values()
                
                # Update WEIGHT_MATRIX and NUMBER_OF_WEIGHTS
                new_WEIGHT_MATRIX = 2*LENGTH_OF_DENDRITE + 1
                new_NUMBER_OF_WEIGHTS = new_WEIGHT_MATRIX*new_WEIGHT_MATRIX

                if new_NUMBER_OF_WEIGHTS != NUMBER_OF_WEIGHTS:
                    print(f"Updating connections: old={NUMBER_OF_WEIGHTS}, new={new_NUMBER_OF_WEIGHTS}")
                    print_to_side_panel(f"Updating connections: old={NUMBER_OF_WEIGHTS}, new={new_NUMBER_OF_WEIGHTS}")
                    for layer in range(1, Num_Layers-1):
                        for x, y in np.ndindex(cells.shape[:2]):
                            if cells[x, y, layer] is not None:
                                cells[x, y, layer].remap_weights(LENGTH_OF_DENDRITE)
                    
                    WEIGHT_MATRIX = new_WEIGHT_MATRIX
                    NUMBER_OF_WEIGHTS = new_NUMBER_OF_WEIGHTS

                print(f"Updated parameters: LENGTH_OF_DENDRITE={LENGTH_OF_DENDRITE}, WEIGHT_MATRIX={WEIGHT_MATRIX}, NUMBER_OF_WEIGHTS={NUMBER_OF_WEIGHTS}")
                print_to_side_panel(f"Updated parameters: LENGTH_OF_DENDRITE={LENGTH_OF_DENDRITE}, WEIGHT_MATRIX={WEIGHT_MATRIX}, NUMBER_OF_WEIGHTS={NUMBER_OF_WEIGHTS}")   

        #Press X Resets number of weights, weights, biases, and charge delta to values and reset genes :() in future should just reset proteins
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                for z in range(1, Num_Layers-1):
                    for x, y in np.ndindex(cells.shape[:2]):
                        if cells[x, y, z] is not None:
                            #initialize the network genes based on the global variables or genes based on if autonomous network genes is on
                            cells[x, y, z].initalize_network_genes(NUMBER_OF_WEIGHTS, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate)
                            cells[x, y, z].color_genes()
                            cells[x, y, z].initialize_network_proteins()
                            cells[x, y, z].color_proteins()
                if not AUTONOMOUS_NETWORK_GENES:
                    print(f"Number of weights, bias range, average weights, charge delta: {NUMBER_OF_WEIGHTS} {Bias_Range} {Avg_Weights_Cell} {charge_delta}")
                    print_to_side_panel(f"Number of weights, bias range, average weights, charge delta: {NUMBER_OF_WEIGHTS} {Bias_Range} {Avg_Weights_Cell} {charge_delta}")
                else:
                    print(f"Autonomous network genes mode: ON, weights, biases, average weights, charge delta set based on genes")
                    print_to_side_panel("Autonomous network genes mode: ON, weights, biases, average weights, charge delta set based on genes")

                not_saved_yet = True
                max_bingo_count = 0
                bingo_count = 0
                total_loss, total_predictions, running_avg_loss = 0, 0, 0
                training_cycles = 0
                points = []

        # Press 'S' to Save the previously saved layers 0 to NUMLAYER-1 for  training
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                save_file("") 

        # Press 'L' to Load the previously saved layers 0 to NUMLAYER-1 - for  training
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l and not show_3d_view:
                load_file()
                
        # Press 'M' to Load the MNEST training data for layers 0 and NUM_LAYERS-1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m and not show_3d_view:
                load_training_data_main()

        # Press 'F' for Forward_propogation - direction is either forward or reverse
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                direction_of_charge_flow = "+++++>>>>>"
                print ("Direction_of_charge_flow=", direction_of_charge_flow )
        
        # Press 'R' for Reverse_Forward_propogation - this sends charge backwards to see connections it does NOT back-propogate or ajust weights
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                direction_of_charge_flow = "<<<<<-----"
                print ("Direction_of_charge_flow=", direction_of_charge_flow )
              
        # Press 'B' to toggle Back_propogation - this adjusts weights based on error only makes sense after forward propogation    
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                back_prop = not back_prop
                print ("Back_prop= ", back_prop)
                
        # Press 'T' to toggle Training Mode - This runs matched pairs of data into layers 0 and 15
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t and training_data_loaded == True:
                training_mode = not training_mode
                print ("training_mode = ", training_mode)     
        
        #Press 'A' to toggle Andromida - same as running so saved for later use
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                andromida_mode = not andromida_mode
                print ("andromida_mode = ", andromida_mode)     
                #print_to_side_panel("andromida_mode = ", andromida_mode)
                
        #Press 'W' to reset all gradient changes 
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                reset_all_gradient_changes()
                print("All gradient changes reset")
                
        #Press 'G' to toggle between Genes and Proteins display modes
        if event.type == pygame.KEYDOWN and not show_3d_view:
            if event.key == pygame.K_g:
                if display == "genes":
                    display = "proteins"
                else:
                    display = "genes"
                print(f"display = {display}")
                print_to_side_panel(f"display = {display}")
        
        # Press 'V' to cycle through different display sets
        if event.type == pygame.KEYDOWN and event.key == pygame.K_v and not show_3d_view:
            display_set = (display_set + 1) % 3  # Cycle through 3 display sets
            
            if display_set == 0:
                # 1st set: All settings
                settings = get_all_settings()
                help_surface.fill(WHITE)
                render_help_text(help_surface, settings, font_small, BLACK, 10, 10, 20)
                screen.blit(help_surface, (MAIN_SURFACE_WIDTH+3, 0))
            elif display_set == 1:
                # 2nd set: Averages
                display_averages()
            else:
                # 3rd set: Statistics
                cell_types = update_cell_types(cells)
                phenotype_cell_types = update_phenotype_cell_types(cells)
                display_statistics(cell_types)
                display_phenotype_statistics(phenotype_cell_types)
                display_max_charge_diff(5)
            
            pygame.display.flip()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_3:  # Press '3' to toggle 3D view
                show_3d_view = not show_3d_view
                if show_3d_view:
                    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
                    setup_3d_view()
                else:
                    pygame.display.set_mode((EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_4:  # Press '4' to toggle backprop view
                show_3d_view = True
                show_backprop_view = not show_backprop_view  # Add this as a global variable
                if show_3d_view:
                    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
                    setup_3d_view()
                else:
                    pygame.display.set_mode((EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))


        if show_3d_view:
            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button
                    rotation_y += event.rel[0]
                    rotation_x += event.rel[1]
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    zoom += 1
                elif event.button == 5:  # Mouse wheel down
                    zoom -= 1
   
        # Check for mouse-button-up
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_up = True

        # Mouse down places a Cell!             
        if event.type == pygame.MOUSEBUTTONDOWN and not show_3d_view:
            x, y = event.pos
            if y < WINDOW_HEIGHT and x < WINDOW_WIDTH: #Have window extension, don't want to mess up calculations, so only allow in main window
                cell_x, cell_y, layer = convert_x_y_to_index (x,y)
                
                # Check if it's a right-click (or secondary click on Mac)
                if event.button == 3 or (event.button == 1 and pygame.key.get_mods() & pygame.KMOD_CTRL):
                    if cells[cell_x, cell_y, layer] is not None:
                        # Display cell information
                        cell_types = update_cell_types(cells) 
                        count_pos, total_cells, phenotype_cell_types = update_phenotype_cell_types(cells) 
                        print_to_side_panel(f"Cells {total_cells} | Positive {count_pos} | Fraction {count_pos/(total_cells+epsilon):.2f} /n Weights {total_weights_list[0]} | Weight/Cell {total_weights_list[0]/(total_cells+epsilon):.2f}")
                        
                        print_to_side_panel(str(cells[cell_x, cell_y, layer]))
   
                    else:
                        print_to_side_panel("No cell at this location")
                else:
                    # Left-click functionality for cell creation/deletion
                    cells[cell_x, cell_y, layer] = Cell(layer, cell_x, cell_y, NUMBER_OF_WEIGHTS, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate) if cells[cell_x, cell_y, layer] is None else None
                    mouse_up = False

        # Mouse down MAKES a Cell while holding mouse down and moving!
        if event.type == pygame.MOUSEMOTION and not show_3d_view:
            if not mouse_up:
                x, y = event.pos
                if y < WINDOW_HEIGHT and x < WINDOW_WIDTH: #Have window extension, don't want to mess up calculations, so only allow in main window
                    cell_x, cell_y, layer = convert_x_y_to_index (x,y)
                    if cells[cell_x, cell_y, layer] is None: cells[cell_x, cell_y, layer] = Cell(layer, cell_x, cell_y, NUMBER_OF_WEIGHTS, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate) 

        # In your main event loop, add a new key to toggle the stats display
        if event.type == pygame.KEYDOWN and not show_3d_view:
            if event.key == pygame.K_v:  # 'V' for View stats
                show_training_stats = not show_training_stats

    if running: 
        update_cells()

    if training_mode:
        total_loss = 0 #Reset each epoch
        total_predictions = 0 # right now this is just the number of training data but with epochs or batches it will chnage
        training_cycles += 1
        try:    
            training(learning_rate, LENGTH_OF_DENDRITE, weight_decay)
        except Exception as e:
            print(f"An error occurred: {e}")
            print_to_side_panel(f"An error occurred in training: {e}")
        bottom_caption_surface.fill(BLACK)
        
        # In your main training loop, update and potentially display the stats
        try:
            if training_cycles % stats_update_frequency == 0:
                if show_training_stats:
                    update_training_stats()
                    display_training_stats()
        except Exception as e:
            print(f"An error occurred: {e}")
            print_to_side_panel(f"An error occurred getting stats: {e}")

    if display_updating:
        if show_3d_view:
            if show_backprop_view:
                pass # we show i in backprop not in main loop 
            else:
                render_3d_network()
            pygame.display.flip()
            
        else:
            growthsurface.fill(WHITE)
            draw_cells()
            draw_grid()
            screen.blit(growthsurface, (0, 0))

    if not show_3d_view:
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time include sceen update!!
            
        if not training_mode: bottom_caption_surface.fill(BLACK)
        prune_color = GREEN if running else WHITE
        if prune: prune_color = RED 

        # In your main game loop, update the bottom caption to include this information
        text_surface = font.render(f"Running = {running} | Andromida = {andromida_mode} | Prune = {prune}: {prune_logic} | Charge = {charge_change_protection}: {charge_delta:.2e}, Gradient = {gradient_prune}: {gradient_threshold:.2e}", True, prune_color)
        text_surface1 = font.render(f"Training = {training_mode}: {learning_rate:.4f} | {direction_of_charge_flow} | Back_prop = {back_prop} | Auto Genes = {AUTONOMOUS_NETWORK_GENES} | Learning Rate: {learning_rate:.4f} | Cycles: {training_cycles}", True, WHITE)   
        text_surface2 = font.render(f"Elapsed: {elapsed_time:.2f} | Training_data: {how_much_training_data} | Loss: {running_avg_loss:.4f} | Correct: {bingo_count} | Max Correct: {max_bingo_count}", True, WHITE) 
        
        bottom_caption_surface.fill(BLACK)
        bottom_caption_surface.blit(text_surface, (10, 10))
        bottom_caption_surface.blit(text_surface1, (10, 40))
        bottom_caption_surface.blit(text_surface2, (10, 70))
        if training_mode:
            prediction_plot() # Average loss per training_cycle.

        screen.blit(bottom_caption_surface, (0, WINDOW_HEIGHT))
        pygame.display.flip()

        start_time = time.time()  # Record the start time did at end since want to include time to update the information on screen