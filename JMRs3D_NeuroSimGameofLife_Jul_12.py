#JMRs Genetic Game of Neural Network LIFE!! updated with NMIST data July 11
#fully working forward and reverse flow, now need to jsut fix backprop!
#a few enviromental items like prunning.
#can add more cell autonmous items, so we select for neurons that instead of changing over training set, have some major information content. like change 2 twice or 5 tims over training set?pp
#Added an “environmental” variable.  If you don’t change charge by a threshold in a session (training set), you die!
#You can watch the network wire from layer 1 to 14 (0 and 15 are training) this way over time.
#Each neuron is connected by dendrites with weights to the 9 cells above (I used dendrites not axons, since I wanted to have the weights that affect a cell
#IN that cell. Each cell is shown here by a 3 x 3 with colors for different genes, or proteins, I reserved red for charge (a protein) so you a see it clearly
#respond to the learning set (I can also toggle to see all 9 weights for each cell).
#Once I get connections wired to layer 14 then I will start back propagation to tune that network.
#Later, I may prune cells depending on error, or some other attribute that selects they are not tuning well.

import pygame
import numpy as np
import random
import copy
import pickle
import os
import math
import tkinter as tk
from tkinter import filedialog
import datetime
#import pygame_gui

input_path_training ='/Users/jonathanrothberg/MNIST_as_cells_training_full_in_out'
#input_path_training = '/Users/jonathanrothberg/MNIST_as_cells_training'  #None type outside of charged cells.
input_path_test = '/Users/jonathanrothberg/MNIST_as_cells_test'

#WINDOW_WIDTH = 1344
#WINDOW_HEIGHT = 1344
WINDOW_WIDTH = 1008
WINDOW_HEIGHT = 1008
CELL_SIZE = 9
FPS = 100

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

COLORS = [INDIGO, ORANGE, BLUE, YELLOW, GREEN, RED, VIOLET, PINK ] # Reserve GREEN for weights, and Red for charge 

WIDTH = WINDOW_WIDTH // CELL_SIZE
HEIGHT = WINDOW_HEIGHT // CELL_SIZE

WIDTH = WIDTH // 4  # making a 4 x 4 to represent the 16 layers
HEIGHT = HEIGHT //4  # Each grid is now 1 / 4th. Otherwise it goes off the visible screen

NUM_LAYERS = 16

my_defs = """The functionality of each gene:

Overcrowding Tolerance (OT): This gene determines the maximum number of neighbors a cell can have before it dies due to overcrowding.
For example, if OT = 3, the cell will die if it has more than 3 neighbors.

Isolation Tolerance (IT): This gene determines the minimum number of neighbors a cell needs to survive.
For example, if IT = 2, a cell with fewer than 2 neighbors will die.

Birth Threshold (BT): This gene determines the number of neighbors an empty cell needs to produce a new cell. If an empty cell has exactly this many neighbors, a new cell will be born.
For example, if BT = 3, a new cell will be born in an empty cell that has exactly 3 neighbors. And inherit the genes by recombination between two parents ONE of whom gave it the BT = 3 allele.

Mutation Rate (MR): This gene determines the chance of a cell's genes mutating when it is reproduced.
It is a fraction, a higher value meaning a higher chance of mutation.
For example, if MR = 1, there is a 1% chance (If rate was selected per 100) each gene can mutate when a new cell is born.
A mutation in the BT gene gives a new cell the chance to be BORN even if neither parents had a BT gene allowing it to be born in that position.
Not until the next cycle will a cell be checked for OT & IT.

Mutation (somatic) here means that any gene's value has chance to randomly change to any value between 0 and 7 (or whatever range is selected by the user).

It's important to note that in JMR's modified version of the Game of Life, each cell can have different values for these genes, leading to diverse behaviors in the simulation.
Cells inherit these genes from their parent cells (random selectin of two parents), with a chance of (germline) mutation based on their inherited Mutation Rate.

Each square from top left:	Red, Green,				 .colors (only the first 4 are used in display right now)
                            Blue, Yellow
                            Red, Green               .charge, .error

Brightness allele value:	OT IT
                            BT MR
Phenotypes value:			Charge Error 

                            Overcrowding Tolerance (OT): .genes[0]  neighbors <= OT 
                            Isolation Tolerance (IT)   : .genes[1]  neighbors >= IT 
                            
                            Birth Threshold (BT)       : .genes[2]         neighbors == BT for birth
                            Mutation Rate (MR)         : .genes[2]
                            Charge Gene (CH)           : .genes[4]
                            Weight Gene (WG)           : .genes[5]
                            Error Gene (ER)            : .genes[6]
                            Bias Gene (BI)             : .genes[7]
                            
                            OT >=  neighbors >= IT 
                            OT > IT"""

conways_defs = """The original Conway's Game of Life has a fixed set of rules that apply to all cells equally:

Overcrowding Tolerance (OT): Any live cell with more than three (3) live neighbors dies, as if by overcrowding.
Isolation Tolerance (IT)   : Any live cell with fewer than two (2) live neighbors dies, as if by isolation/lonelyness).

Birth Threshold (BT): Any cell location with exactly three (3) live neighbors becomes a live cell, as if by reproduction (a new cell is born).
There is no concept of a Gene or Mutation Rate (MR) in the original game, as rules are static and don't change."
                            
                            Overcrowding Tolerance (OT): .genes[0]  neighbors <= 3 OT 
                            Isolation Tolerance (IT)   : .genes[1]  neighbors >= 2 IT 
                            Birth Threshold (BT)       : .genes[2]  neighbors == 3 BT
                            
                            OT 3 >=  neighbors >=  2 IT"""

how_network_works = """The weights represent the influence that a given neuron (in this case, a cell) in one layer has
on the neurons in the subsequent layer. By adjusting these weights, we allow the network to learn over time.
Loaded data introduced into layer 0.

During Training, adjustments made to weights throughout the network, based on error signals determined by the difference between
the actual outputs in layer 14 and the desired outputs in layer 15 and so on for each subsequence layer.
The training process involves adjusting weights to minimize the error between the network's output and this desired output.

This is a form of supervised learning, where the network is guided towards the correct output by feedback provided
in the form of a desired output."""

controls = """Space - Toggle simulation, Q - Quit, S - Save 0 & 15, L - Load 0 & 15, O - Output all layers, I - Input all layers, N - load NMEST,
F - Forward Prop, R - Reverse Forward Prop, B - Toggle Back Prop, T - Toggle Training Mode,  G - Gene Display, W - Weight display """

def draw_grid():
    # Draw vertical grid lines
    for x in range(0, WINDOW_WIDTH, WINDOW_WIDTH // 4):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT), 2)
    # Draw horizontal grid lines
    for y in range(0, WINDOW_HEIGHT, WINDOW_HEIGHT // 4):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y), 2)
 
 
def update_cell_types(cells):
    global cell_types
    cell_types = {}
    for layer in range(NUM_LAYERS):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cell_type = tuple(cells[x, y, layer].genes)
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
 
 
def update_phenotype_cell_types(cells):
    global phenotype_cell_types
    phenotype_cell_types = {}
    
    epsilon = 1e-8  # Small constant for epsilon smoothing
    
    # Collect all cell properties in separate lists for overall standard deviation calculation
    all_charges = []
    all_biases = []
    all_errors = []
    all_weights = []
    
    count_pos = 0  # Count of cells included in the calculation
    total_cells = 0  # Total number of cells that are not None
     
    for layer in range(NUM_LAYERS):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                total_cells += 1  # Increment total_cells
                # Exclude cells with all weights being 0 or epsilon, and cells with error being 0 or epsilon
                if not all(weight in [0, epsilon] for weight in cells[x, y, layer].weights): # and cells[x, y, layer].error not in [0, epsilon]:
                    all_charges.append(cells[x, y, layer].charge)
                    all_biases.append(cells[x, y, layer].bias)
                    all_errors.append(cells[x, y, layer].error)
                    all_weights.append(np.mean(cells[x, y, layer].weights))  # Average weight per cell
                    count_pos += 1  # Increase the count

    # Calculate mean and std for each property across all cells
    charge_mean, charge_std = (np.mean(all_charges), np.std(all_charges)) if all_charges else (0, 0)
    bias_mean, bias_std = (np.mean(all_biases), np.std(all_biases)) if all_biases else (0, 0)
    error_mean, error_std = (np.mean(all_errors), np.std(all_errors)) if all_errors else (0, 0)
    weights_mean, weights_std = (np.mean(all_weights), np.std(all_weights)) if all_weights else (0, 0)

    # Loop again to assign phenotypes based on the overall properties' distributions
    for layer in range(NUM_LAYERS):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                weights_avg = np.mean(cells[x, y, layer].weights)

                phenotype = (
                    'charge:' + ('+' if cells[x, y, layer].charge > charge_mean else '-') + str(int(abs((cells[x, y, layer].charge - charge_mean) / (charge_std + epsilon)))),
                    'bias:' + ('+' if cells[x, y, layer].bias > bias_mean else '-') + str(int(abs((cells[x, y, layer].bias - bias_mean) / (bias_std + epsilon)))),
                    'error:' + ('+' if cells[x, y, layer].error > error_mean else '-') + str(int(abs((cells[x, y, layer].error - error_mean) / (error_std + epsilon)))),
                    'weights:' + ('+' if weights_avg > weights_mean else '-') + str(int(abs((weights_avg - weights_mean) / (weights_std + epsilon))))
                )

                if phenotype not in phenotype_cell_types:
                    phenotype_cell_types[phenotype] = []
                phenotype_cell_types[phenotype].append(cells[x, y, layer])
    #print ("Number of cells with non zero wieghts & error, total cells", count_pos, total_cells)

def display_statistics():
    sorted_types = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]
    font = pygame.font.SysFont(None, 12)
    statistics = ["OT IT BT MR CH WG ER BI: Overcrowding Isolation Birth Mutation Charge Weight Error Bias:"] + [f"Type: {cell_type} | Count: {count}" for cell_type, count in sorted_types]
    
    for i, stat in enumerate(statistics):
        text = font.render(stat, True, BLACK)
        print (stat)
        #screen.blit(text, (10, 20 + i * 15))
        #pygame.display.update()


def display_phenotype_statistics():
    sorted_types = sorted([(phenotype, len(cells), np.mean([cell.charge for cell in cells]), np.mean([cell.bias for cell in cells]), np.mean([cell.error for cell in cells]), np.mean([np.mean(cell.weights) for cell in cells])) for phenotype, cells in phenotype_cell_types.items()], key=lambda x: x[1], reverse=True)[:5]

    font = pygame.font.SysFont(None, 12)
    statistics = ["Charge Bias Error Weights: Phenotype | Count | Avg Charge | Avg Bias | Avg Error | Avg Weights:"] + [f"{phenotype} | Count: {count} | Avg Charge: {avg_charge:.4f} | Avg Bias: {avg_bias:.4f} | Avg Error: {avg_error:.4f} | Avg Weights: {avg_weights:.4f}" for phenotype, count, avg_charge, avg_bias, avg_error, avg_weights in sorted_types]
    
    for i, stat in enumerate(statistics):
        text = font.render(stat, True, BLACK)
        print (stat)
        #screen.blit(text, (10, 20 + i * 15))
        #pygame.display.update()


def prediction_to_actual():
    global cells, bingo_count
    # Extract the one-hot encoded label from the middle of layer 14
    one_hot_label_guess = np.array([cell.charge if cell is not None else 0 for cell in cells[14, 9:19, 14]]) 
    # Extract the one-hot encoded label from the middle of layer 15
    one_hot_label_actual = np.array([cell.charge if cell is not None else 0 for cell in cells[14, 9:19, 15]]) 

    # Get the position of the '1' in the one-hot encoded labels
    pos_guess = np.argmax(one_hot_label_guess)
    pos_actual = np.argmax(one_hot_label_actual)

    text14 = font.render (str(pos_guess), True, BLACK)
    text15 = font.render (str(pos_actual), True, BLACK)
    screen.blit(text14, (950, 1050))
    screen.blit(text15, (1050, 1050))

    # Check if the positions match
    if pos_guess == pos_actual: 
        bingo_count += 1  # Increment the bingo count
        text = font.render("That is a Bingo!", True, RED)
        screen.blit(text, (1100, 1100))
        
        if bingo_count == how_much_training_data:  # Check if bingo count is 2
            text = font.render("Solved!", True, RED)
            screen.blit(text, (1100, 1100))
    else:
        bingo_count = 0  # Reset the bingo count if the positions don't match


class Cell:
#Genes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, *CH={self.genes[4]}, *WG={self.genes[5]}, *ER={self.genes[6], *BI={self.genes[7] },
#Proteins:                                                                            #charge={self.charge}, weights={self.weights}, error={self.error},bias ={self.bias} )"
    def __init__(self, layer, genes=None):
        
        self.layer = layer 
        if genes is None:
            self.genes = [random.randint(lower_allele_range, upper_allele_range) for _ in range(8)]  # Generate 8 random alleles for the eight genes - currently using only first 4
            if self.genes[0] < self.genes[1]:  # If gene[0] is less than gene[1], swap them
                self.genes[0], self.genes[1] = self.genes[1], self.genes[0]
            
            #if self.layer == 1:
            #    self.genes[0] += 14 # add 9 to the overcrowding tollerane so it can survice on layer 1 when full 28 x 28.
            #    self.genes[2] += 14 #add 9 so it can have babies even though all layers above are cells
        
        else:
            self.genes = genes
        #self.colors = [tuple(int(gene * color_component // 28) for color_component in color) for gene, color in zip(self.genes, COLORS)]
        self.colors = [tuple(min(int(gene * color_component // upper_allele_range), 255) for color_component in color) for gene, color in zip(self.genes, COLORS)] #

        # Assign the charge based on the layer - eventually will assign based on an "input mask"
        self.charge = 0 

        # Assign random weights - this is tuned these based on some back propagation value from the end "output"
        self.weights = [(random.uniform(-.5, .5)) for _ in range(9)]  # Generate 9 random weights for the cell
 
        self.bias =  random.uniform(-.5, .5)
       
        self.error = epsilon
    
    def __str__(self):
        return f"Cell:\nlayer={self.layer}\nGenes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, CH={self.genes[4]}, WG={self.genes[5]}, ER={self.genes[6]}, BI={self.genes[7]} \ncharge={self.charge}, \nweights={self.weights}, \nerror={self.error}, bias ={self.bias})"
    
    
def draw_cells():
    global errors
    if display == "genes":
        for layer in range(NUM_LAYERS):
            for (x, y) in np.ndindex(cells.shape[:2]):
                if cells[x, y, layer] is not None:
                    for i in range(9):  # Change from 6 to 9
                        if i < 4:  # First 4 colors represent the 4 genes
                            color = cells[x, y, layer].colors[i]
                            color = (max(0, min(color[0], 255)), max(0, min(color[1], 255)), max(0, min(color[2], 255)))
                        elif i == 4:  # Fifth color represents the charge
                            charge_intensity = cells[x, y, layer].charge  # Assumes charge is an int between 0 and 1

                            if np.isinf(charge_intensity) or charge_intensity > 10 or charge_intensity < -10:
                                #print(f'Charge intensity out of bounds at layer {layer}, position {(x, y)}, value {charge_intensity}')
                                charge_intensity = np.clip(charge_intensity, -10, 10)  # Clip the value between -100 and 100
                                cells[x, y, layer].charge = charge_intensity

                            charge_intensity = min(int(abs(charge_intensity) * 255), 255)
                            color = (charge_intensity, 0, 0)  # Red color
                        elif i == 5:  # Sixth color represents the absolute error signal
                            error_intensity = cells[x, y, layer].error  # Assumes error is a float

                            if np.isinf(error_intensity) or error_intensity > 4 or error_intensity < -4:
                                #print(f'Error intensity is out of bounds at layer {layer}, position {(x, y)}, value {error_intensity}')
                                error_intensity = np.clip(error_intensity, -4, 4)  # Clip the value between -1 and 1
                                cells[x, y, layer].error = error_intensity

                            error_intensity = min(int(abs(error_intensity) * 255), 255) # update to do colors around the distribution of error 
                            color = (0, 0, error_intensity)  # Blue color
                        elif i == 6:  # Seventh color
                            bias = cells[x, y, layer].bias  # Assumes error is a float

                            if np.isinf(bias) or bias > 4 or bias < -4:
                                #print(f'Bias is out of bounds at layer {layer}, position {(x, y)}, value {error_intensity}')
                                bias = error_intensity = np.clip(bias, -4, 4)  # Set it to the maximum value, or handle it in any other appropriate way
                                cells[x, y, layer].bias = bias
            
                            bias = min(int(abs(bias) * 255), 255) # update to do colors around the distribution of error 
                            color = (bias, 0, bias)  # Magenta
                        elif i == 7:  # Eighth color
                            meanweights = np.mean(cells[x, y, layer].weights)
                            meanweights = min(int(abs(meanweights) * 255), 255)
                            color = (0, meanweights, 0)  # Green color 
                        elif i == 8:  # do gradiant charge * error
                            color = color # Assign color based on your requirements
                            # color = (255, 0, 255)  # Magenta color for example - should be same as last color if no error

                        pygame.draw.rect(
                            screen,
                            color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3,
                                CELL_SIZE // 3
                            )
                        )
                else:
                    pygame.draw.rect(
                        screen,
                        WHITE,
                        pygame.Rect(
                            x * CELL_SIZE + (layer % 4) * (WINDOW_WIDTH // 4),
                            y * CELL_SIZE + (layer // 4) * (WINDOW_HEIGHT // 4),
                            CELL_SIZE,
                            CELL_SIZE
                        )
                    )    
    else:
        for layer in range(NUM_LAYERS):
            for (x, y) in np.ndindex(cells.shape[:2]):
                if cells[x, y, layer] is not None:
                    for i in range(9):  # Change from 6 to 9
                        weight = cells[x, y, layer].weights[i]  # Assumes error is a float

                        if np.isinf(weight) or weight > 10 or weight < -10:
                            print(f'Weight is out of bounds at layer {layer}, position {(x, y)}, value {weight}')
                            weight = random.random() # Set it to the maximum value, or handle it in any other appropriate way
                            cells[x, y, layer].weights[i] = weight 
                        
                        weight = min(int(abs(weight) * 255), 255) # update to do colors around the distribution of error 
                        color = (0, weight, 0)  # Green color

                        pygame.draw.rect(
                            screen,
                            color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3,
                                CELL_SIZE // 3
                            )
                        )
                else:
                    pygame.draw.rect(
                        screen,
                        WHITE,
                        pygame.Rect(
                            x * CELL_SIZE + (layer % 4) * (WINDOW_WIDTH // 4),
                            y * CELL_SIZE + (layer // 4) * (WINDOW_HEIGHT // 4),
                            CELL_SIZE,
                            CELL_SIZE
                        )
                    )       

def update_cells():
    global cells, changed_cells, changed_cells_forward, changed_cells_reverse
    epsilon = 1e-8
    # Create a deep copy of the cells array, which will store the updated state of the cells
    new_cells = copy.deepcopy(cells)

    if mode == "future":
        start_layer = 1 # Since it is range starts at 1, leave layer 0 alone
        stop_layer = NUM_LAYERS - 1 
    else:
        start_layer = 1 
        stop_layer = NUM_LAYERS-1   
    
    for layer in range(start_layer, stop_layer):

        # Loop over all positions in the 2D grid within each layer
        for (x, y) in np.ndindex(cells.shape[:2]):

            if prune == True:
                if (x, y, layer) not in changed_cells_forward and (x, y, layer) not in changed_cells_reverse:
                    new_cells[x, y, layer] = None
            
            if andromida_mode == True:
                # Define the offsets for neighboring positions in 3D space
                if layer == 1:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [0, 1] if dx != 0 or dy != 0 or dz != 0]
                elif layer == 14:
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

                # If the current cell is alive
                if cells[x, y, layer] is not None:
                    # Potentially mutate the cell's genes based on its somatic mutation rate
                    cells[x, y, layer].genes = [gene if random.randint(1, som_mut_rate) > cells[x, y, layer].genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in cells[x, y, layer].genes]

                # If the current cell is Empty and there are living neighbors
                elif cells[x, y, layer] is None and alive_neighbors:
                    # Get potential parents from alive neighbors
                    potential_parents = alive_neighbors

                    # If there are at least two potential parents
                    if len(potential_parents) >= 2:
                        # Randomly choose two parents
                        parent1, parent2 = random.sample(potential_parents, 2)

                        # Combine and potentially mutate the parents' genes to create the new cell's genes
                        new_genes = [random.choice([parent1.genes[i], parent2.genes[i]]) for i in range(8)]
                        new_genes = [gene if random.randint(1, germ_mut_rate) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]

                        # If the number of alive neighbors is exactly equal to the birth threshold of the new cell, create a new cell
                        if num_alive == new_genes[2]:
                            new_cells[x, y, layer] = Cell(layer, new_genes)
                    
                    # If there is only one potential parent
                    elif potential_parents:
                        # Choose the single parent
                        parent1 = random.choice(potential_parents)

                        # Copy and potentially mutate the parent's genes to create the new cell's genes
                        new_genes = parent1.genes
                        new_genes = [gene if random.randint(1, germ_mut_rate) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]

                        # If the number of alive neighbors is exactly equal to the birth threshold of the new cell, create a new cell
                        if num_alive == new_genes[2]:
                            new_cells[x, y, layer] = Cell(layer, new_genes)
                    
                #NEW June 13, protect cells that are connected in network   
                if cells[x, y, layer] is not None:
                    # Exclude cells from the list of cells that had their charge changes from genetic selection
                    if (x, y, layer) not in changed_cells_forward and (x, y, layer) not in changed_cells_reverse:    
                        # Kill the cell if it is overcrowded or isolated based on genes
                        if (num_alive <= cells[x, y, layer].genes[1] or num_alive >= cells[x, y, layer].genes[0]):
                            new_cells[x, y, layer] = None             
                            
    # Replace the old cells array with the updated cells array
    cells = new_cells


# Load training data for layer 0 and 15
def load_layers(image_index): #works nicely and shows as you load the dataset.
    global cells  # Accessing the global variable cells
    file_path = os.path.join(input_path_training, f'simulation_state_layers_0_and_15_image_{image_index}.pkl')
    with open(file_path, 'rb') as f:
        cells[:,:,0], cells[:,:,15] = pickle.load(f)
        
    
# Function to load the training data
def load_training_data():
    global training_data_layer_0, training_data_layer_15, cells  # Define these as global variables
    
    training_data_layer_0 = []  # Initialize as empty list
    training_data_layer_15 = []  # Initialize as empty list
    print ("Loading training data from: ",input_path_training )
    for k in range(how_much_training_data):  # For the first 1000 images
        load_layers(k)
        training_data_layer_0.append(copy.deepcopy(cells[:,:,0]))  # Append data to list no just pickle reference!
        training_data_layer_15.append(copy.deepcopy(cells[:,:,15]))  # Append data to list # Append data to list
        draw_cells()
        draw_grid()
        pygame.display.update()
        prediction_to_actual()


def training():
    global cells, changed_cells_forward, changed_cells_reverse
    # Define your delta value here
    changed_cells = set()
    changed_cells = set()
    set_size = how_much_training_data

    # Initialize old_charge_1_to14 as a dictionary
    old_charge_1_to14 = {k: [] for k in range(1, 15)}

    # Initialize a dictionary to keep track of maximum charge difference for each cell
    max_charge_diff = {}

    for i in range(set_size):
        cells[:,:,0] = training_data_layer_0[i]
        cells[:,:,15] = training_data_layer_15[i]

        draw_cells()
        draw_grid()
        pygame.display.update()

        # Save old charges for all layers from 1 to 14
        for k in range(1, 15):
            for cell in cells[:,:,k].flatten():
                if cell is not None:
                    old_charge_1_to14[k].append(cell.charge)

        train_network()

        draw_cells()
        draw_grid()

        # Calculate charge difference for each cell and update max_charge_diff
        for k in range(1, 15):
            for cell in cells[:,:,k].flatten():
                if cell is not None:
                    charge_difference = abs(cell.charge - old_charge_1_to14[k].pop(0))
                    if cell not in max_charge_diff or charge_difference > max_charge_diff[cell]:
                        max_charge_diff[cell] = charge_difference

    
 
    # Add cells to changed_cells if their maximum charge difference is greater than delta
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            for k in range(1, 15):  # Adjusted range to check layers 1 to 14 inclusive
                cell = cells[i, j, k]
                if cell is not None:
                    if cell in max_charge_diff and max_charge_diff[cell] > delta:
                        changed_cells.add((i, j, k))  # Add a tuple of the coordinates instead of the cell
       
    if direction_of_charge_flow == "forward_flow":
        changed_cells_forward = changed_cells
    if direction_of_charge_flow == "reverse_flow":
        changed_cells_reverse = changed_cells              
  
def get_input_values():
    prompts = [
        {"name": "germ_mut_rate", "prompt": "Enter the Germline Mutation rates per", "default": 1000, "type": int},
        {"name": "som_mut_rate", "prompt": "Enter the Somatic Mutation rates up per", "default": 1000, "type": int},
        {"name": "lower_allele_range", "prompt": "Enter the lower value for alleles", "default": 2, "type": int},
        {"name": "upper_allele_range", "prompt": "Enter the upper value for alleles", "default": 15, "type": int},
        {"name": "how_much_training_data", "prompt": "Enter training set size", "default": 15, "type": int},
        {"name": "learning_rate", "prompt": "Enter the learning rate", "default": .01, "type": float},
        {"name": "delta", "prompt": "Enter the charge change threshold needed to protect a cell from removal", "default": .025, "type": float},
        {"name": "weight_change_threshold", "prompt": "Enter weight_change_threshold to report on", "default": .005, "type": float},
        {"name": "bingo_count", "prompt": "Enter the number of times you need to be correct in a row for bingo", "default": 2, "type": int},
        {"name": "error_clip", "prompt": "Enter gradient_clip_range", "default": 10000, "type": int},
        {"name": "max_gradient_norm", "prompt": "Enter max_gradient_norm", "default": 10000, "type": int}
    ]

    user_inputs = {}

    for prompt in prompts:
        user_input = input(f"{prompt['prompt']} ({prompt['default']}): ")
        if user_input == "":
            user_inputs[prompt["name"]] = prompt["default"]
        else:
            user_inputs[prompt["name"]] = prompt["type"](user_input)

    return user_inputs

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("JMR's Game of Life with Genetics & Nerual Network")
font = pygame.font.SysFont(None, 24)

# Initialize cells with all None - cut in WIDTH and HEIGH in 4th since you now have 4 x 4 array
cells = np.full((WIDTH, HEIGHT, NUM_LAYERS), None, dtype=object)

# Dictionary to store cell types and their frequencies
cell_types = {}
phenotype_cell_types = {}

# Mouse-button-up event tracking (to prevent continuous drawing while holding the button)
mouse_up = True

print(my_defs)
print(conways_defs)
print (how_network_works)
print (controls) 

# Call the function to get the input values and set up your run
globals().update(get_input_values())

screen.fill(WHITE)  

# Create a set to store cells that had their charge changed
changed_cells_reverse = set()
changed_cells_forward = set()

# Dictionary to store cell types and their frequencies
cell_types = {}
phenotype_cell_types = {}

# Flag to indicate whether the simulation has started
running = False
prune = False # Kills an cells changed_cells_forward or changed_cells_reverse
training_mode = False
andromida_mode = False # updates cells based on rules
back_prop = False
training_data_loaded = False

simulating = True # main loop

mode = "future"
display = "genes"
direction_of_charge_flow = "forward_flow"

epochs = 1
epsilon = 1e-8

errors = np.zeros_like(cells, dtype=float)  # to store error signals

# Game loop
while simulating:
    for event in pygame.event.get():
        
        # Press SPACE to start and stop the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = not running
                print ("running = ", running)
                
        # Press P to start and stop prunning of all not changing charge
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                prune = not prune
                print ("prune = ", prune)        
                
        # Press Q to end the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                simulating = False
                
        # Press V to change variable settings
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                globals().update(get_input_values())
                
        # Press 'S' to save layers 0 and 15 of the current state of the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                with open('simulation_state_layer_1.pkl', 'wb') as f:
                    pickle.dump(cells[:,:,0], f)  # assuming layer 1 corresponds to index 0
                with open('simulation_state_layer_16.pkl', 'wb') as f:
                    pickle.dump(cells[:,:,15], f)  # assuming layer 16 corresponds to index 15
                print("Layers 0 and 15 saved!")
       
        # Press 'L' to input the previously saved layers 0 and 15 - for doing training
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                if os.path.exists('simulation_state_layer_1.pkl') and os.path.exists('simulation_state_layer_16.pkl'):
                    with open('simulation_state_layer_1.pkl', 'rb') as f:
                        cells[:,:,0] = pickle.load(f)
                    with open('simulation_state_layer_16.pkl', 'rb') as f:
                        cells[:,:,15] = pickle.load(f)
                    print("Layers 0 and 15 loaded!")
                    #mode = "training" # Will only update layers 1 to 14, using 0 and 15 for training otherwise they would get messed up before you can start training
                else:
                    print("No saved layers 0 and 15 found!")
        
          # Press 'O' to output the current state of the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                # Create the directory if it doesn't exist
                if not os.path.exists('simulation_runs'):
                    os.makedirs('simulation_runs')

                # Get the current date and time
                now = datetime.datetime.now()
                filename = 'simulation_state_{}_{}.pkl'.format(now.strftime('%Y%m%d'), now.strftime('%H%M'))

                with open(os.path.join('simulation_runs', filename), 'wb') as f:
                    pickle.dump(cells, f)
                print("Simulation state saved layers 0 to 15!")


        # Press 'I' to Input the previously saved state of the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                # List all .pkl files in the directory
                files = [f for f in os.listdir('simulation_runs') if f.endswith('.pkl')]
                
                # If there are no .pkl files, print a message
                if not files:
                    print("No saved simulation state found!")
                    break

                # Print all .pkl files and let the user choose one
                for i, filename in enumerate(files):
                    print(f"{i+1}: {filename}")
                file_num = input("Enter the number of the file you want to load: ")
                
                # If the input is empty, break the loop
                if not file_num:
                    break

                file_num = int(file_num) - 1

                # Load the chosen file
                with open(os.path.join('simulation_runs', files[file_num]), 'rb') as f:
                    cells = pickle.load(f)
                print("Simulation state loaded layers 0 to 15!")

        
        # Press 'N' to load the NNIST training data
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                print ("Loading training data")
                load_training_data()
                training_data_loaded = True
    
        # Press 'F' for Forward_propogation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                direction_of_charge_flow = "forward_flow"
                print ("direction_of_charge_flow =", direction_of_charge_flow )
        
        # Press 'R' for Reverse_Forward_propogation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                direction_of_charge_flow = "reverse_flow"
                print ("direction_of_charge_flow =", direction_of_charge_flow )
              
        # Press 'B' for Back_propogation on
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                back_prop = not back_prop
                print ("Back_prop = ", back_prop)
                
          
        # Press 'T' to toggle Training Mode
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t and training_data_loaded == True:
                training_mode = not training_mode
                print ("training_mode = ", training_mode)     

        
        #Press 'A' for Andromida Growth Mode
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                andromida_mode = not andromida_mode
                print ("andromida_mode = ", andromida_mode)     
                
        #Press 'W' for Display Modes visualize weights
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                display = "weights"
                print ("display =", display)
    
        #Press 'G' for Display Modes visualize genes and proteins        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                display = "genes"
                print ("display =", display)       
        
        # Mouse down MAKES a Cell!             
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            # Determine which of the 16 screen segments (layer) mouse is in
            layer_x = x // (WINDOW_WIDTH // 4)
            layer_y = y // (WINDOW_HEIGHT // 4)
            layer = layer_x + layer_y * 4

            # Adjust x and y for the layer before calculating cell position
            adjusted_x = x - layer_x * (WINDOW_WIDTH // 4)
            adjusted_y = y - layer_y * (WINDOW_HEIGHT // 4)

            cell_x = min(adjusted_x // CELL_SIZE, WIDTH - 1)
            cell_y = min(adjusted_y // CELL_SIZE, HEIGHT - 1)

            cells[cell_x, cell_y, layer] = Cell(layer) if cells[cell_x, cell_y, layer] is None else None
            mouse_up = False

        # Check for mouse-button-up
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_up = True

        # Mouse down MAKES a Cell while holding mouse down and moving!
        if event.type == pygame.MOUSEMOTION:
            if not mouse_up:
                x, y = event.pos
                # Determine which of the 16 screen segments (layer) mouse is in
                layer_x = x // (WINDOW_WIDTH // 4)
                layer_y = y // (WINDOW_HEIGHT // 4)
                layer = layer_x + layer_y * 4
                
                # Adjust x and y for the layer before calculating cell position
                adjusted_x = x - layer_x * (WINDOW_WIDTH // 4)
                adjusted_y = y - layer_y * (WINDOW_HEIGHT // 4)
                
                cell_x = min(adjusted_x // CELL_SIZE, WIDTH - 1)
                cell_y = min(adjusted_y // CELL_SIZE, HEIGHT - 1)
                
                cells[cell_x, cell_y, layer] = Cell(layer) if cells[cell_x, cell_y, layer] is None else None

        # Display Cell information & Statistics on mousemotion over cell
        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            # Determine which of the 16 screen segments (layer) mouse is in
            layer_x = x // (WINDOW_WIDTH // 4)
            layer_y = y // (WINDOW_HEIGHT // 4)
            layer = layer_x + layer_y * 4

            # Adjust x and y for the layer before calculating cell position
            adjusted_x = x - layer_x * (WINDOW_WIDTH // 4)
            adjusted_y = y - layer_y * (WINDOW_HEIGHT // 4)
            
            cell_x = min(adjusted_x // CELL_SIZE, WIDTH - 1)
            cell_y = min(adjusted_y // CELL_SIZE, HEIGHT - 1)
            
            update_cell_types(cells) #Can move to 
            update_phenotype_cell_types(cells) # can move to calculate ony when we need this data
            
            if cells[cell_x, cell_y, layer]:
                avg_error = np.mean([cells[x, y, z].error for z in range(NUM_LAYERS) for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None])
                avg_error_absolute = np.mean([abs(cells[x, y, z].error) for z in range(NUM_LAYERS) for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None]) 
                print()
                print(f"Average signed error for the network: {avg_error}")
                print(f"Average absolute error for the network: {avg_error_absolute}")
                print ("cell_x, cell_y, layer: ", cell_x, cell_y, layer)
                print(cells[cell_x, cell_y, layer]) # Special function on the STR def for that class Cell
                display_statistics()
                print ("\n")
                display_phenotype_statistics()
                print (f"running: {running}, training_mode: {training_mode}, direction_of_charge_flow: {direction_of_charge_flow}, back_prop: {back_prop}, prune: {prune}, andromida_mode: {andromida_mode}")
                
    if running: 
        update_cells()
    
    if training_mode:
        training()
        
    draw_cells()
    draw_grid()
    pygame.display.update()
    
     