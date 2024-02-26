#JMRs Genetic Game of Neural Network LIFE!! updated with NMIST data July 11
#Fully working forward and reverse flow, fully working backprop Sept 21st
#Sept 21 variable number of layers... and first success of 9 out of 15 test cases with 4 layers.
#Moved mutation outside of Andromedia Sept 25
#3x 20/20 and 60/60 on MNEST 98 out of 100 on 8 layer network Sept 26
# 100/100 on MNEST 8 layer 49 weights per cell Sept 30, 64% on next 100 
#Added info screen Sept 28
# source .venv/bin/activate when using visual studio code   
# added fashion MNIST data set path for mac Feb 22 2024
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

#input_path_training = filedialog.askdirectory(title="Select input directory for training data")  #TK crashes with pygames on mac
#input_path_training_Digits ='/data/MNIST_5000_0_15_Cells'
#input_path_training_Fashion ='/data/Fashion_MNIST_5000'
input_path_training_Digits ='/Users/jonathanrothberg/MNIST_1000_0_15_Cells'
input_path_training_Fashion ='/Users/jonathanrothberg/mnist_fashion'
fashion = """0 T-shirt/top | 1 Trouser | 2 Pullover | 3 Dress | 4 Coat | 5 Sandal | 6 Shirt | 7 Sneaker | 8 Bag | 9 Ankle boot"""

#WINDOW_WIDTH = 1344 # for cell size of 12
#WINDOW_HEIGHT = 1344
WINDOW_WIDTH = 1008
WINDOW_HEIGHT = 1008
WINDOW_EXTENSION = 100
EXTENDED_WINDOW_HEIGHT = WINDOW_HEIGHT + WINDOW_EXTENSION

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

COLORS = [INDIGO, ORANGE, PINK, YELLOW, GREEN, RED, VIOLET, BLUE ] # Same length as number of genes 

WIDTH = WINDOW_WIDTH // CELL_SIZE
HEIGHT = WINDOW_HEIGHT // CELL_SIZE

WIDTH = WIDTH // 4  # making a 4 x 4 to represent the 16 layers #28 x 28
HEIGHT = HEIGHT //4  # Each grid is now 1 / 4th. Otherwise it goes off the visible screen

UPPER_ALLELE_LIMIT = 28

ARRAY_LAYERS = 16

jmr_defs = """
The function of each gene:

Overcrowding Tolerance (OT): This gene determines the maximum number of neighbors a cell can have before it dies due to overcrowding.
For example, if OT = 3, the cell will die if it has more than 3 neighbors.

Isolation Tolerance (IT): This gene determines the minimum number of neighbors a cell needs to survive.
For example, if IT = 2, a cell with fewer than 2 neighbors will die.

Birth Threshold (BT): This gene determines the number of neighbors an empty grid space needs to produce a new cell. If an empty space has exactly this many neighbors, a new cell will be born.
For example, if BT = 3, a new cell will be born in an empty space that has exactly 3 neighbors. And inherit the genes by recombination between two parents ONE of whom could give it the BT = 3 allele.

Mutation Rate (MR): This gene determines the chance of a cell's genes mutating when it reproduces.
It is a fraction of a user selected denominator, a higher value meaning a higher chance of mutation.
For example, if MR = 1, there is a 1% chance (If rate was selected per 100) each gene can mutate at the time a new cell is born.

A "germ-line" mutation in the BT gene gives a new cell the chance to be BORN even if neither parents had a BT gene value allowing it to be born in that position.
Not until the next cycle will a cell be checked for OT & IT to determine cycle.  So many cells are born that die in the next cycle.

Mutation (somatic) here means that any gene's value has chance to randomly change to any value between 0 and 7 (or whatever range is selected by the user).

Cells inherit these genes from their parent cells (random independent selection of two parents if multiple are available, and then independent assortment of genes), 
with a chance of (germline) mutation based on their inherited Mutation Rate each cycle.

The world is made up of 3D grid of potential cells. Each cell has a set of genes that determine its behavior. Each cells is displayed as an informative 3 x 3 arrangment colored blocks.			
                                           
Brightness allele/phenotype value:	OT      IT      BT
                                    MR      Charge  Error
                                    Bias    Weight  Gradient		

    Overcrowding Tolerance (OT): .genes[0]  neighbors <= OT 
    Isolation Tolerance (IT)   : .genes[1]  neighbors >= IT 
    
    Birth Threshold (BT)       : .genes[2]  neighbors == BT for birth
    Mutation Rate (MR)         : .genes[2]
    
    OT >=  neighbors >= IT 
    OT > IT
    -- Future Genes --
    Weight Gene (WG)           : .genes[4]
    Bias Gene (BI)             : .genes[5]
    Error Gene (ER)            : .genes[6]
    Charge Gene (CG)           : .genes[7]"""

conways_defs = """
The original Conway's Game of Life has a fixed set of rules that apply to all cells equally:

Overcrowding Tolerance (OT): Any live cell with more than three (3) live neighbors dies, as if by overcrowding.
Isolation Tolerance (IT)   : Any live cell with fewer than two (2) live neighbors dies, as if by isolation/lonelyness).

Birth Threshold (BT): Any cell location with exactly three (3) live neighbors becomes a live cell, as if by reproduction (a new cell is born).
There is no concept of a Gene or Mutation Rate (MR) in the original game, as rules are static and don't change."
                            
    Overcrowding Tolerance (OT): .genes[0]  neighbors <= 3 OT 
    Isolation Tolerance (IT)   : .genes[1]  neighbors >= 2 IT 
    Birth Threshold (BT)       : .genes[2]  neighbors == 3 BT
    
    OT 3 >=  neighbors >=  2 IT"""

how_network_works = """
The weights represent the influence that a given neuron (in this case, a cell) in one layer has
on the neurons in the subsequent layer. By adjusting these weights, we allow the network to learn over time.
Loaded data introduced into layer 0.

During Training, adjustments made to weights throughout the network, based on error signals determined by the difference between
the actual outputs in layer 14 and the desired outputs in layer 15 (This is NUM_LAYERS-2 and NUM_LAYERS-1) and so on for each subsequence layer.
The training process involves adjusting weights to minimize the error between the network's output and this desired output.

This is a form of supervised learning, where the network is guided towards the correct output by feedback provided
in the form of a desired output."""

forward_pass ="""
Forward pass is the process of moving the input data through the network layer by layer:
Layer N-2:       Layer N-1:       Layer N:

B11 B12 B13      C11 C12 C13      W11 W12 W13
B21 B22 B23  --> C21 C22 C23  --> W21 W22 W23
B31 B32 B33      C31 C32 C33      W31 W32 W33

1. Each cell in Layer N-2 (Bij) calculates its charge based on its inputs and passes it to the corresponding cells in Layer N-1 (Cij) using the weights stored in Cij.

2. Each cell in Layer N-1 (Cij) calculates its charge based on the received charges and its weights, then passes it to the corresponding cells in Layer N (Wij) 
using the weights stored in Wij.

3. Each cell in Layer N (Wij) calculates its final charge based on the received charges and its weights."""

how_backprop_works = """
The weights of the current cell are updated based on the error of the current cell and the charge of the cells in the layer above. 

The `get_upper_layer_cells` function is used in the `update_weights_and_bias` function to get the cells from the layer above the current cell. 
These cells are used to calculate the gradient for updating the weights of the current cell.

The `get_layer_below_cells` function is used in the `compute_error_signal_other_layers` function to get the cells from the layer below the current cell. 
These cells are used to calculate the error signal for the current cell.

Here is a simple text-based figure to illustrate the concept:
Layer N-1 (Above)     Layer N (Current)     Layer N+1 (Below)

  Cell A                 Cell X                 Cell 1
  Cell B                 Cell Y                 Cell 2
  Cell C                 Cell Z                 Cell 3

In this figure, Cells A, B, and C are in the layer above the current layer. Cells X, Y, and Z are in the current layer. 
Cells 1, 2, and 3 are in the layer below the current layer.

When updating the weights of Cell X during backpropagation, we use the charges of Cells A, B, and C (from the layer above) and the error of Cell X. 
This is done in the `update_weights_and_bias` function.

When calculating the error signal for Cell X, we use the errors and weights of Cells 1, 2, and 3 (from the layer below) and the derivative of the activation function of Cell X. 
This is done in the `compute_error_signal_other_layers` function.

Regarding the index, (`weight_index`) in the `update_weights_and_bias` function. 
The index is calculated based on the relative position (dx, dy) of the cell in the layer above to the current cell. 
The `reversed_index` is used in the `compute_error_signal_other_layers` function to access the weights of the cells in the layer below."""


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
            num_layers = int(input("Enter number of layers: "))
            num_weights = int(input("Enter number of weights: "))
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

    # Create icon for each file .png
    icon  = create_icon(pkl_filename, 3)
    pygame.image.save(icon, icon_filename)
    print(f"Icon saved to {icon_filename}!")

    # Save variables to txt file .txt
    with open(txt_filename, 'w') as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"Cell Settings:\n")
        f.write(f"germ_mut_rate: {germ_mut_rate}\n")
        f.write(f"som_mut_rate: {som_mut_rate}\n")
        f.write(f"lower_allele_range: {lower_allele_range}\n")
        f.write(f"upper_allele_range: {upper_allele_range}\n")
        f.write(f"Simulation Settings:\n")
        f.write(f"NUM_LAYERS: {Num_Layers}\n")
        f.write(f"LENGTH_OF_AXON: {LENGTH_OF_AXON}\n")
        f.write(f"Bias_Range: {Bias_Range}\n")
        f.write(f"Weight Range based on estimate Avg weights/cell: {Avg_Weights_Cell}\n")
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
                       

def load_file():
    global cells, Num_Layers, NUMBER_OF_WEIGHTS, LENGTH_OF_AXON, WEIGHT_MATRIX, not_saved_yet, bingo_count, max_bingo_count, total_loss, total_predictions, running_avg_loss, training_cycles, changed_cells_reverse, changed_cells_forward, points  
    file_dir = "./saved_states/"
    # Check if directory exists, if not, create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_list = [f for f in os.listdir(file_dir) if f.startswith("sim") and f.endswith(".pkl")]
    
    if len(file_list) == 0:
        print("No saved simulation states found!")
    else:
        growthsurface.fill(WHITE)
        bottom_caption_surface.fill(BLACK)
        print("Hit Return to select a file to load:")
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
                    real_i = i +1 + page*16
                    # Create icon for each file
                    icon  = create_icon(os.path.join(file_dir, file_name), cell_size)
                    # Display the icon
                    screen.blit(icon, ((i % 4) * 252*cell_size + 75, (i // 4) * 252*cell_size + 50))

                    # Parse the file name to get the NUM_LAYERS and NUMBER_OF_WEIGHTS
                    NUM_LAY[real_i], num_wei[real_i] = parse_file_name(file_name)
                    # Render the file name
                    text = font_directory.render(f"{i+1+page*16}. {file_name}", True, (0, 0, 0))
                    text2 = font_directory.render(f"Layers: {NUM_LAY[real_i]} | Weights {num_wei[real_i]}", True, (0, 0, 0))
                    # Display the file name
                    screen.blit(text, ((i % 4) * 252*cell_size + 10, (i // 4) * 252*cell_size + 112*cell_size + 95))
                    screen.blit(text2, ((i % 4) * 252*cell_size + 10, (i // 4) * 252*cell_size + 112*cell_size + 120))
                
                text_surface = font.render(f"Please HIT Return to Select File in Console", True, WHITE) 
                bottom_caption_surface.blit(text_surface, (50, 10))
                screen.blit(bottom_caption_surface, (0, EXTENDED_WINDOW_HEIGHT - WINDOW_EXTENSION))
                pygame.display.flip()
            new_page = False

            for event2 in pygame.event.get():     
                if event2.type == pygame.KEYDOWN:
                    if event2.key == pygame.K_RETURN:
                        try:
                            selection = int(input("Enter the file number: "))
                            if selection < 1 or selection > len(file_list):
                                raise ValueError
                            file_path = os.path.join(file_dir, file_list[selection-1])
                            with open(file_path, 'rb') as f:
                                cells = pickle.load(f)

                            # Just until all legacy saved data needs to be rotated and flipped
                            legacy = input("Is this a legacy file? (y/n): ")
                            if legacy == "y":
                                cells = np.flip(np.rot90(cells, 3), 1)

                        except Exception as e:
                            print("An error occurred: " + str(e))
                            no_file_selected = False
                            return
                        else:
                            try:   
                                print ("Loading saved layers")
                                #Set the number of layers and number of weights
                                Num_Layers = NUM_LAY[selection]
                                NUMBER_OF_WEIGHTS = num_wei[selection]
                                WEIGHT_MATRIX = int(math.sqrt(NUMBER_OF_WEIGHTS))
                                LENGTH_OF_AXON = int((WEIGHT_MATRIX - 1) / 2)
                                
                            except Exception as e:
                                print("An error occurred: " + str(e))
                                no_file_selected = False
                                return
                            else:
                                print(f"Simulation state loaded from {file_path} | NUM_LAYER: {Num_Layers} | Number of weights per cell {NUMBER_OF_WEIGHTS} | Length of Axon {LENGTH_OF_AXON} |  weight_matrix {WEIGHT_MATRIX}")
                                not_saved_yet = True
                                max_bingo_count = 0
                                bingo_count = 0
                                total_loss, total_predictions, running_avg_loss = 0, 0, 0
                                training_cycles = 0
                                no_file_selected = False
                                changed_cells_reverse = set()
                                changed_cells_forward = set()
                                points = []
                                return
                            
                    elif event2.key == pygame.K_DOWN:
                        page = min(page + 1, len(file_list) // 16)
                        new_page = True
                    elif event2.key == pygame.K_UP:
                        page = max(page - 1, 0)
                        new_page = True
            clock.tick(60)

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

    for k in range(start_index, start_index + how_much_training_data):  # For the first 1000 images
        try:
            load_layers(input_path_training, k)
            training_data_layer_0.append(copy.deepcopy(cells[:,:,0]))  # Append data to list not just pickle reference!
            training_data_NUM_LAYER_MINUS_1.append(copy.deepcopy(cells[:,:,Num_Layers -1]))  # Append data to list not just pickle reference!   
            growthsurface.fill(WHITE)
            draw_cells()
            draw_grid()
            pygame.display.flip()

        except Exception as e:
            print(f"Error occurred while loading data at index {k}: {e}")
            continue

# Control Loop for loading training data
def load_training_data_main():
    global training_data_loaded, how_much_training_data, start_index, total_weights_list, not_saved_yet, max_bingo_count, bingo_count, total_loss, total_predictions, running_avg_loss, training_cycles, changed_cells_reverse, changed_cells_forward, points
    default_training_data = how_much_training_data
    default_start_index = start_index
    try:     
        which_data_set = input ("Enter which data set to load (MNIST, M for MNIST or F for Fashion MNIST): ")
        if which_data_set.lower() == "f":
            training_data = input_path_training_Fashion
            print (fashion)
        else:
            training_data = input_path_training_Digits
        how_much_training_data = int(input("Enter training set size (20, 1 to 1000): ") or 20)
        start_index = int(input("Start index (0, 0 to 999): ") or 0)
    except ValueError:
        print("Invalid input. No New Data loaded")
    else:
        # Ensure start_index is within the valid range
        if start_index + how_much_training_data > 5000:
            how_much_training_data = default_training_data
            start_index = default_start_index
            print("Total can't exceed 5000. Returning to defaults", how_much_training_data, start_index)
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
                changed_cells_reverse = set()
                changed_cells_forward = set()
                points = []
            except Exception as e:
                print("An error occurred while loading the training data: ", + str(e))


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
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT), 2)
    # Draw horizontal grid lines
    for y in range(0, WINDOW_HEIGHT, WINDOW_HEIGHT // 4):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y), 2)
 
 
def update_cell_types(cells):  
    start_layer = 1
    stop_layer = Num_Layers-1
    cell_types = {}
    number_of_genes_to_use = 3 # Only consider the first 3 genes when creating the cell_type tuple
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
    
    count_pos = 0  # Count of cells included in the calculation
    total_cells = 0  # Total number of cells that are not None
    start_layer = 1
    stop_layer = Num_Layers-1
    for layer in range(start_layer, stop_layer):
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
    for layer in range(start_layer, stop_layer):
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
    return (count_pos, total_cells, phenotype_cell_types)


def display_statistics(cell_types):
    sorted_types = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]
    statistics = ["[OT IT BT] CH WG ER BI: Overcrowding Isolation Birth"] + [f"Type: {cell_type} | Count: {count}" for cell_type, count in sorted_types]
    for i, stat in enumerate(statistics):
        print (stat)
       

def display_phenotype_statistics(phenotype_cell_types):
    sorted_types = sorted([(phenotype, len(cells), np.mean([cell.charge for cell in cells]), np.mean([cell.bias for cell in cells]), np.mean([cell.error for cell in cells]), np.mean([np.mean(cell.weights) for cell in cells])) for phenotype, cells in phenotype_cell_types.items()], key=lambda x: x[1], reverse=True)[:10]
    statistics = ["Phenotype: Charge Bias Error Weights| Count | Avg Charge | Avg Bias | Avg Error | Avg Weights:"] + [f"{phenotype} | Count: {count:4} | Avg Charge: {avg_charge:.4f} | Avg Bias: {avg_bias:+.4f} | Avg Error: {avg_error:+.4e} | Avg Weights: {avg_weights:.4f}" for phenotype, count, avg_charge, avg_bias, avg_error, avg_weights in sorted_types]
    for i, stat in enumerate(statistics):
        print (stat)


def display_max_charge_diff(N=5):
    # Sort the max_charge_diff dictionary by values in descending order and take the first N
    top_N_charge_diff = sorted(max_charge_diff.items(), key=lambda x: max(x[1]) - min(x[1]), reverse=True)[:N] 
    for cell, charge_diff_list in top_N_charge_diff:
        # Calculate the difference between the maximum and minimum charge in the list for this cell
        charge_diff = max(charge_diff_list) - min(charge_diff_list)
        # Get the coordinates of the cell
        coordinates = np.where(cells == cell)
        if coordinates[0].size > 0:  # Check if coordinates is not empty
            x, y, layer2 = coordinates[0][0], coordinates[1][0], coordinates[2][0]
            print(f"{x}, {y}, {layer2} Charge difference: {charge_diff:.2f} | ", end='')


def display_averages():
    try:
        avg_error = np.zeros(Num_Layers)
        avg_error_absolute = np.zeros(Num_Layers)
        avg_charge = np.zeros(Num_Layers)
        avg_charge_absolute = np.zeros(Num_Layers)
        avg_gradient_absolute = np.zeros(Num_Layers)

        avg_bias = np.zeros(Num_Layers)
        avg_bias_absolute = np.zeros(Num_Layers)
        avg_weights = np.zeros(Num_Layers)
        avg_weights_absolute = np.zeros(Num_Layers)
        
        avg_error[0] = np.mean([cells[x, y, z].error for z in range(1, Num_Layers-1) for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None and cells[x, y, z].error != epsilon])
        avg_error_absolute[0] = np.mean([abs(cells[x, y, z].error) for z in range(1, Num_Layers-1) for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None and cells[x, y, z].error != epsilon]) 
        avg_gradient_absolute[0] = np.mean([abs(cells[x, y, z].error * cells[x,y,z].charge) for z in range(1, Num_Layers-1) for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None and cells[x, y, z].error != epsilon]) 
        
        
        for z in range(1, Num_Layers-1):
            cells_in_layer = [cells[x, y, z] for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None and cells[x, y, z].error != epsilon]
            avg_error[z] = np.mean([cell.error for cell in cells_in_layer])
            avg_error_absolute[z] = np.mean([abs(cell.error) for cell in cells_in_layer])
            
            avg_charge[z] = np.mean([cell.charge for cell in cells_in_layer])
            avg_charge_absolute[z] = np.mean([abs(cell.charge) for cell in cells_in_layer])
            
            avg_gradient_absolute[z] = np.mean([abs(cell.error * cell.charge) for cell in cells_in_layer])

            avg_bias[z] = np.mean([cell.bias for cell in cells_in_layer])
            avg_bias_absolute[z] = np.mean([abs(cell.bias) for cell in cells_in_layer])
            
            all_weights = [weight for cell in cells_in_layer for weight in cell.weights]
            avg_weights[z] = np.mean(all_weights)
            avg_weights_absolute[z] = np.mean([abs(weight) for weight in all_weights])

        count_error_epsilon = np.sum([1 for z in range(1, Num_Layers-1) for x, y in np.ndindex(cells.shape[:2]) if cells[x, y, z] is not None and cells[x, y, z].error == epsilon])
        '''for z in range(1, actual_layers-1):
            for x, y in np.ndindex(cells.shape[:2]):
                if cells[x, y, z] is not None and cells[x, y, z].error == epsilon:
                    print(f"Coordinates: ({x}, {y}, {z}), Cell: {cells[x, y, z]}", end="r")'''

        print(f"Predictions: {total_predictions} | Average Loss: {running_avg_loss:.4e}")
        print(f"avg_gradient: {avg_gradient_absolute[0]:.4e} | Average signed error: {avg_error[0]:.4e} | Average abs error: {avg_error_absolute[0]:.4e} | Number of cells with error = epsilon: {count_error_epsilon} ")
        for z in range(1, Num_Layers-1):
            print(f"Layer {z}: gradient: {avg_gradient_absolute[z]:.4e} | error: {avg_error[z]:.4e}, abs : {avg_error_absolute[z]:.4e}|  charge: {avg_charge[z]:.4f}, abs: {avg_charge_absolute[z]:.4f} |  bias: {avg_bias[z]:.4f},  absolute : {avg_bias_absolute[z]:.4f} |weights: {avg_weights[z]:.4f},  abs : {avg_weights_absolute[z]:.4f} ")
    except Exception as e:
        print("Error in averages", e)


def prediction_plot():
    global points
    offset = 756
    # Scale the y-coordinate
    y = WINDOW_EXTENSION - (running_avg_loss / 10) * WINDOW_EXTENSION

    # Add the point to the list
    points.append((len(points)+offset, y))

    # If there are more than offset points, remove the first one
    if len(points) > 250:
        points.pop(0)
        # Adjust the x values of the points to create a scrolling effect
        points = [(x-1, y) for x, y in points]

    # Draw the points on the bottom_caption_surface
    for point in points:
        pygame.draw.circle(bottom_caption_surface, BLUE, point, 1)


def prediction_to_actual():
    global cells, bingo_count, max_bingo_count, total_loss, total_predictions, running_avg_loss
    # Extract the one-hot encoded label from the middle of last layer (layer 14 default)
    one_hot_label_guess = np.array([cell.charge if cell is not None else 0 for cell in cells[9:19, 14,  Num_Layers-2]]) 
    # Extract the one-hot encoded label from the middle of training data layer (layer 15 default)
    one_hot_label_actual = np.array([cell.charge if cell is not None else 0 for cell in cells[9:19, 14,  Num_Layers-1 ]]) 

    # Calculate the loss using categorical cross-entropy
    if np.sum(one_hot_label_actual) > 0 and np.sum(one_hot_label_guess) > 0:
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
        
                                                                       
class Cell:
    def __init__(self, layer, weights_per_cell_possible, genes=None):
        epsilon = 1e-8
        self.layer = layer 
        #Genes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, *WG={self.genes[4]}, *BI={self.genes[5]}, *ER={self.genes[6], *CH={self.genes[7]}   
        if genes is None: # make up 8 genes, we only use the first four as of sept 22 2023
            self.genes = np.random.randint(lower_allele_range, upper_allele_range, size=8)
            if self.genes[0] < self.genes[1]:  # If gene[0] is less than gene[1], swap them you alawys want Overcrowding Tolerance (OT) to be greater than Isolation Tolerance (IT)
                self.genes[0], self.genes[1] = self.genes[1], self.genes[0]
            
            #This next group not used yet, but will be used to make more cell autonomous and less dependent on global variables
            #Genes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, *WG={self.genes[4]}, *BI={self.genes[5]}, *AW={self.genes[6], *TE={self.genes[7]}  
            #self.gene[4] = weights_per_cell_possible # set gene 4 to average weights per cell this determines the weight distribution.
            #self.gene[5] = Bias_Range # set gene 5 to bias range this determines the bias distribution.
            #self.gene[6] = Avg_Weights_Cell # 
            #self.gene[7] = training_cycles # set gene 7 to epsilon this determines the error distribution.

        else: # genes are passed from parent cell (Future proteins imprinted on child cell)
            self.genes = genes
        
        #Assign random weights - these are tuned  based on some back propagation value from "output" NUM_LAYER 
        self.weights = np.random.randn(weights_per_cell_possible) / np.sqrt(Avg_Weights_Cell) # Good model to use for generating weights, Avg_Weights_Cell is expected, not calculated
        self.bias =  np.random.uniform(Bias_Range) # updated to ONLY be positive Sept 27 and reduced range
        self.charge = 0 # Charge is the sum of the weights * the charge of the connected cell + bias
        self.colors = [tuple(min(int(gene * 255 // 15), 255) for color_component in color) for gene, color in zip(self.genes, COLORS)]  #for speed, simce could calculate each time by genotype
        self.error = epsilon # Small not zero so no divison by zero issues. 
    
    def __str__(self):
        weights = [f'{w:.4f}' for w in self.weights]
        bias = f'{self.bias:.4f}'
        error = f'{self.error:.4e}'
        charge = f'{self.charge:.4f}'
        return f"Neuron: layer={self.layer} Genes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, CH={self.genes[4]},WG={self.genes[5]}, ER={self.genes[6]}, BI={self.genes[7]} \ncharge={charge}, error={error}, bias={bias} \nweights={weights}, "
    

def draw_cells(): # removed clipping from drawing :) Sept 22 note
    epsilon = 1e-8
    gradiant_c = 1
    error_intensity_c = 1
    if display == "genes":
        for layer in range(Num_Layers):
            for (x, y) in np.ndindex(cells.shape[:2]):
                if cells[x, y, layer] is not None:
                    for i in range(9):  # 3 x 3 grid of genes, charge, error, bias, weights, gradiant
                        if i < 4:  # First 4 colors represent the 4 genes
                            color = cells[x, y, layer].colors[i]
                            #color = (max(0, min(color[0], 255)), max(0, min(color[1], 255)), max(0, min(color[2], 255))) #error from bias bug not here                        
                        elif i == 4:  # Charge: Fifth position RED represents charge
                            charge_intensity = cells[x, y, layer].charge  # Assumes charge is an int between 0 and 1
                            charge_intensity_c = min(int(abs(charge_intensity) * 25500), 255)
                            color = (charge_intensity_c, 0, 0)  # Red color                       
                        elif i == 5 and not np.isnan(cells[x, y, layer].error):  #Error: Sixth color represents the absolute error signal
                            error_intensity = cells[x, y, layer].error  # Assumes error is a float
                            error_intensity_c = max(0, min(int(np.log(abs(error_intensity+epsilon) * 55)), 255)) # update to do colors around the distribution of error 
                            color = (0, 0, error_intensity_c)  # Blue color                        
                        elif i == 6:  # Bias: Seventh color represents bias
                            bias = cells[x, y, layer].bias  # Assumes error is a float        
                            bias_c = min(int(abs(bias) * 10000), 255) # update to do colors around the distribution of error 
                            color = (bias_c, 0, bias_c)  # Magenta                        
                        elif i == 7:  # Weights: Eighth color average weights
                            meanweights = np.mean(np.abs(cells[x, y, layer].weights))
                            meanweights_c = min(int(abs(meanweights) * 1000), 255)
                            color = (0, meanweights_c, 0)  # Green color                       
                        elif i == 8 and not np.isnan(charge_intensity * error_intensity):  # Gradiant: Ninth color charge * error signals
                            gradient = charge_intensity * error_intensity
                            gradient = np.clip(gradient, -gradient_clip_range, gradient_clip_range) # recalculating gradiant so it is not clipped since we id not clip charge or error
                            gradient_c = max(0, min(int(np.log(abs(gradient+epsilon) * 55)), 255))
                            color = (0, 0, gradient_c)  # Blue color
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
        for layer in range(Num_Layers):
            for (x, y) in np.ndindex(cells.shape[:2]):
                if cells[x, y, layer] is not None:
                    for i in range(9):  
                        weight = cells[x, y, layer].weights[i]  # Assumes error is a float
                        
                        weight_c = min(int(abs(weight) * 255), 255) 
                        color = (0, weight_c, 0)  # Green color

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


#This is were cells are born, die, and mutate. It has two swithes around pruning - actively killing cells or protecting based  on charge_change lists
def update_cells():  # This is where cells are created, mutated, and killed based on Prune, and Charge as well as genes
    global cells, changed_cells_forward, changed_cells_reverse
    start_layer = 1 
    stop_layer = Num_Layers-1   
    for layer in range(start_layer, stop_layer):
        # Loop over all positions in the 2D grid within each layer
        for (x, y) in np.ndindex(cells.shape[:2]):

            if prune == True: # Pruning happens before potential for cell birth
                if prune_logic == "AND":
                    if not ((x, y, layer) in changed_cells_forward and (x, y, layer) in changed_cells_reverse):
                        cells[x, y, layer] = None
                elif prune_logic == "OR":
                    if not ((x, y, layer) in changed_cells_forward or (x, y, layer) in changed_cells_reverse):
                        cells[x, y, layer] = None

            # If the current cell is alive somatic mutation possible # this now happens even when not in andromida mode as long as its running.
            if cells[x, y, layer] is not None:
                # Mutate the cell's genes based on its somatic mutation rate
                cells[x, y, layer].genes = [gene if random.randint(1, som_mut_rate) > cells[x, y, layer].genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in cells[x, y, layer].genes]
            
            if andromida_mode == True:
                # Define the offsets for neighboring positions in 3D space THESE Are physical neigbhers NOT axon connections. 
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
                        new_genes = [random.choice([parent1.genes[i], parent2.genes[i]]) for i in range(8)]
                        new_genes = [gene if random.randint(1, germ_mut_rate) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]
                        #moms_weights = parent1.weights
                        #moms_bias = parent1.bias
                        # If the number of alive neighbors is exactly equal to the birth threshold of the new cell, create a new cell
                        if num_alive == new_genes[2]:
                            cells[x, y, layer] = Cell(layer, NUMBER_OF_WEIGHTS, new_genes)
                    
                    # If there is only one potential parent - asexual reproduction and germ-line mutatino can occur
                    elif potential_parents:
                        # Choose the single parent
                        parent1 = potential_parents[0]  # Choose the first parent
                        # Copy and potentially mutate the parent's genes to create the new cell's genes
                        new_genes = parent1.genes
                        new_genes = [gene if random.randint(1, germ_mut_rate) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]
                        #moms_weights = parent1.weights
                        #moms_bias = parent1.bias
                        # If the number of alive neighbors is exactly equal to the birth threshold of the new cell, create a new cell
                        if num_alive == new_genes[2]:
                            cells[x, y, layer] = Cell(layer, NUMBER_OF_WEIGHTS, new_genes)
                    
                #NEW June 13, protect cells that are connected in network and have charge_change  
                if cells[x, y, layer] is not None:
                    # Exclude cells from the list of cells that had their charge changes from genetic selection
                    if charge_change_protection == True:
                        if not ((x, y, layer) in changed_cells_forward or (x, y, layer) in changed_cells_reverse):  #This should be property of the cell! 
                            # Kill the cell if it is overcrowded or isolated based on genes
                            if (num_alive <= cells[x, y, layer].genes[1] or num_alive >= cells[x, y, layer].genes[0]):
                                cells[x, y, layer] = None             
                    else:
                        # Kill the cell if it is overcrowded or isolated based on genes
                        if (num_alive <= cells[x, y, layer].genes[1] or num_alive >= cells[x, y, layer].genes[0]):
                            cells[x, y, layer] = None           


def training(learning_rate,reach):
    global changed_cells_forward, changed_cells_reverse, bingo_count, max_charge_diff,cells, total_weights_list, total_weights # just added cells to global seeing if that caused issue with episilon 
    #delta  in charge change needed protect the cell from being pruned or protected
    changed_cells = set()
    bingo_count = 0  # Reset the bingo count before each iteration of training or testing data
    set_size = how_much_training_data
    epochs = 1  # Number of epochs to train for

    # Initialize a dictionary to keep track of maximum charge difference for each cell
    max_charge_diff = {}

    for i in range(set_size):
        cells[:,:,0] = training_data_layer_0[i]
        cells[:,:,Num_Layers-1] = training_data_NUM_LAYER_MINUS_1[i]
        total_weights = 0
        train_network(epochs, learning_rate,reach)
        total_weights_list [i] = total_weights

        if display_updating: # if display_updating is true then display the cells is this one used no pygame.display.update()?
           growthsurface.fill(WHITE)
           draw_cells()
           draw_grid()
           pygame.display.flip()

        # Loop over all layers in the 3D grid, except the first and last one
        for k in range(1, Num_Layers-1):
            # Loop over all cells in the current layer
            for cell in cells[:,:,k].flatten():
                # Check if the cell is not empty and its charge is not zero
                if cell is not None: #and cell.charge != 0:   for reverse max and min are the same so this is not needed
                    # If the cell is not already in the max_charge_diff dictionary, add it with its charge as the value
                    if cell not in max_charge_diff:
                        max_charge_diff[cell] = [cell.charge]  # Store charges in a list
                    else:
                        # If the cell is already in the dictionary, append the value p
                        max_charge_diff[cell].append(cell.charge)  # Add to list regardless of delta

        
    #Add cells to changed_cells if their maximum charge difference is greater than delta - kept as seperate loop so delta not checked for each example in training loopa
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            for k in range(1, Num_Layers-1):  
                # Get the current cell
                cell = cells[i, j, k]
                # Check if the cell is not empty
                if cell is not None:
                    # If the cell is in the max_charge_diff dictionary
                    if cell in max_charge_diff:
                        # Calculate the difference between the maximum and minimum charge in the list for this cell
                        charge_diff = max(max_charge_diff[cell]) - min(max_charge_diff[cell]) 
                        
                        if abs(charge_diff) > charge_delta:
                            # Add the coordinates of the cell to the set of changed cells
                            changed_cells.add((i, j, k))  # Add a tuple of the coordinates
    

    if direction_of_charge_flow == "+++++>>>>>":
        changed_cells_forward = changed_cells
    if direction_of_charge_flow == "<<<<<-----":
        changed_cells_reverse = changed_cells                  
                     
                     
def train_network(epochs, learning_rate,reach):
    for epoch in range(epochs):  # Not implemented yet
        
        if direction_of_charge_flow == "+++++>>>>>":
            forward_propagation(reach)  # Regular forward propogation
            if back_prop:
                back_propagation(learning_rate,reach)  # Compute the back propagation
            prediction_to_actual()
        if direction_of_charge_flow == "<<<<<-----":
            reverse_forward_propagation(reach)  # Compute the reverse_forward propagation
        

def reverse_forward_propagation(reach):
    global cells  #modifying the `cells` array, not assigning a new array to it, so you don't actually need the global. 
    # Iterating from the last hidden layer, not the answer layer (layer 15)
    charge = 0
    for layer in range(Num_Layers - 2, 0, -1):
        # For each cell in the 2D grid of each layer
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:  # If the cell at this position and layer is not None
                if layer == Num_Layers - 2:  # If the current layer is the second to last layer (layer 14)
                    if cells[x, y, Num_Layers - 1] is not None:
                        cells[x,y,layer].charge  = cells[x, y, Num_Layers - 1].charge
                else:
                    # Compute the total charge for the current cell
                    charge = compute_total_charge_reverse(x, y, layer, reach)
                    # Replace sigmoid function with ReLU function
                    cells[x, y, layer].charge = relu(charge)
                    #cells[x, y, layer].charge = sigmoid(charge)


def compute_total_charge_reverse(x, y, layer, reach): #used in reverse_forward-propagation to see how the network is connected, using the output of training data
    
    lower_layer_cells = get_layer_below_cells(x, y, layer, reach)
    # Compute the total charge for the current cell, based on the charges from the lower layer (+1) cells and the weights from the lower level  cell
    charge = 0
    charge -= cells[x, y, layer].bias  # I think only for reverse flow you need to subtract the bias and do it first
    for dx, dy, cell in lower_layer_cells:
        # Forward mapping
        weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
        # Reverse mapping
        reversed_index = NUMBER_OF_WEIGHTS - 1 - weight_index
        charge += cell.charge * cell.weights[reversed_index] 
    return charge


def relu(x):  # use for forward
    return np.maximum(0, x)

def relu_derivative(x):  #use for backprop 2x the np.where!
    return np.greater(x, 0).astype(int)

def sigmoid(x): #use for forward 4x faster without the clip but not sure if that is ok
    # Prevent overflow.
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))

def get_upper_layer_cells(x, y, layer, reach):
    upper_layer_cells = []
    for dx in range(-reach, reach + 1):  # range from -reach to reach
        for dy in range(-reach, reach + 1):  # range from -reach to reach
            if 0 <= x + dx < cells.shape[0] and 0 <= y + dy < cells.shape[1] and cells[x + dx, y + dy, layer - 1] is not None:
                upper_layer_cells.append((dx, dy, cells[x + dx, y + dy, layer - 1]))
    return upper_layer_cells

def get_layer_below_cells(x, y, layer, reach): 
    
    layer_below_cells = []
    for dx in range(-reach, reach + 1):  # range from -reach to reach
        for dy in range(-reach, reach + 1):  # range from -reach to reach
            if 0 <= x + dx < cells.shape[0] and 0 <= y + dy < cells.shape[1] and cells[x + dx, y + dy, layer + 1] is not None:
                layer_below_cells.append((dx, dy, cells[x + dx, y + dy, layer + 1]))
    return layer_below_cells


def forward_propagation(reach):
    global cells  # Accessing the global variable cells
    stop_layer = Num_Layers - 1  # leave layer 15 alone since it is what we solve for range stops at 14!! 
    charge = 0
    # Iterating from the first hidden layer, not the input layer (layer 0)
    for layer in range(1, stop_layer):
        # For each cell in the 2D grid of each layer
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:  # If the cell at this position and layer is not None
                # Compute the total charge for the current cell
                charge = compute_total_charge(x, y, layer,reach)
                charge += cells[x, y, layer].bias
                # Replace sigmoid function with ReLU function
                cells[x, y, layer].charge = relu(charge) # this works to get 100 out of 100 for MNEST
                #cells[x, y, layer].charge = sigmoid(charge)


def compute_total_charge(x, y, layer, reach): #used in forward prop THIS IS CORRECT 
    
    upper_layer_cells = get_upper_layer_cells(x, y, layer, reach)
    # Compute the total charge for the current cell, based on the charges from the upper layer cells and the weights from the current cell
    charge = 0
    for dx, dy, cell in upper_layer_cells:
        weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach) # Makes sense for forward you use your own weights 
        charge += cell.charge * cells[x, y, layer].weights[weight_index]
    return charge


def back_propagation(learning_rate,reach):
    global cells
    for layer in range(Num_Layers - 2, 0, -1):  # Iterate through each layer in reverse order (starting from 14) changes 0 to -1 to see if fix espisolon it did not
        # For each cell in the 2D grid of each layer
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:  # If the cell at this position and layer is not None
                error_signal = compute_error_signal(x, y, layer,reach)
                cells[x, y, layer].error = error_signal
                update_weights_and_bias(x, y, layer, learning_rate, reach)  # Update the current layer


def compute_error_signal(x, y, layer,reach):
    error_signal = epsilon # since if you return when no assigment it gives error. don't think needed here.
    if layer == Num_Layers - 2:  
        if cells[x, y, Num_Layers - 1] is not None:   
            desired_output = cells[x, y, Num_Layers - 1].charge
            error_signal = (cells[x, y, layer].charge - desired_output) * relu_derivative(cells[x, y, layer].charge) # my errror is my charge - desired output * relu of my charge - removed the 2x
    else:
        error_signal = compute_error_signal_other_layers(x, y, layer, reach)
    return error_signal


def compute_error_signal_other_layers(x, y, layer, reach):
    layer_below_cells = get_layer_below_cells(x, y, layer, reach)
    error_signal = epsilon # if no layer below return epsilon
    for dx, dy, cell in layer_below_cells:
        # Forward mapping
        weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
        # Reverse index to identify the weights of the cells below you flow into
        reversed_index = NUMBER_OF_WEIGHTS - 1 - weight_index
        error_signal += cell.error * cell.weights[reversed_index] * relu_derivative(cells[x, y, layer].charge) # my error is my error + error below * weight i'm connected to below * relu of my charge
    return error_signal


def update_weights_and_bias(x, y, layer, learning_rate, reach): 
    global cells, total_weights  #update the weights and bias for the current layer
    upper_layer_cells = get_upper_layer_cells(x, y, layer,reach)  # Get the cells from the layer above
    total_weights += len(upper_layer_cells)
    for dx, dy, cell in upper_layer_cells:
        gradient = cells[x, y, layer].error * cell.charge  # my gradient is my error * the charge of the cell above me
        gradient = np.clip(gradient, -gradient_clip_range, gradient_clip_range)
        weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach) # Calculate the index of the weight based on the dx and dy values on how I am connected to above
        cells[x, y, layer].weights[weight_index] -= learning_rate * gradient # I update my weights to above based on my error * the charge of the cell connected by that weight above me (gradient)
    cells[x, y, layer].bias -= learning_rate * cells[x, y, layer].error


'''def debug_backprop(x,y,layer,i,gradiant):
    #print(f'x: {x}, y: {y}, layer: {layer}, weight_index: {weight_index}, before: {before}, after: {after}, gradient: {gradient}', end='\r')
    grad = min(int(abs(gradiant) * 2000), 255)   # is 2000  a good number to see the changes
    color = (0, 0, grad)  # Blue color
    pygame.draw.rect(
                            screen,
                            color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3,
                                CELL_SIZE // 3
                            )
                        ) '''


def get_user_input(prompt, default_value):
    try:
        user_input = input(prompt)
        if user_input == "":
            return default_value
        else:
            return int(user_input)
    except:
        print("Invalid input")
        return default_value

def get_user_input_float(prompt, default_value):
    try:
        user_input = input(prompt)
        if user_input == "":
            return default_value
        else:
            return float(user_input)
    except:
        print("Invalid input")
        return default_value

def get_input_values():
    NUM_LAYERS = get_user_input("Enter the number of layers (8, Range 4 to 16): ", 8)
    if NUM_LAYERS <3 or NUM_LAYERS > 16: NUM_LAYERS = 8
    LENGTH_OF_AXON = get_user_input("Enter the length of the axon (1, Range 1 to 3): ", 1)
    germ_mut_rate = get_user_input("Enter the Germline Mutation rates per (1000): ", 1000)
    som_mut_rate = get_user_input("Enter the Somatic Mutation rates up per (1000): ", 1000)
    lower_allele_range = get_user_input("Enter the lower value for alleles (2): ", 2)
    upper_allele_range = get_user_input("Enter the upper value for alleles (15): ", 15)
    weight_change_threshold = get_user_input_float("Enter the weight change threshold for reporting(0.005): ", 0.005)
    Avg_Weights_Cell = get_user_input("Enter estimate number neurons per cell to set weight distribution (5, 5 to 50 - set to 1 to build network by charge quickly): ", 5)
    BIAS_RANGE = get_user_input_float("Enter the bias randome distributon positive range (.01, Range 0 to .1): ", .01)
    learning_rate = get_user_input_float("Enter the learning rate (0.01, Range .01 to .001): ", 0.01)
    delta = get_user_input_float("Enter the charge delta to protect or prune a cell in protect/prune mode (0.01): ", 0.01)
    return germ_mut_rate, som_mut_rate, lower_allele_range, upper_allele_range, weight_change_threshold, NUM_LAYERS, LENGTH_OF_AXON,learning_rate, BIAS_RANGE,Avg_Weights_Cell, delta

def get_updated_input_values():
    NUM_LAYERS = get_user_input("Enter the number of layers (8, Range 4 to 16): ", 8)
    germ_mut_rate = get_user_input("Enter the Germline Mutation rates per (1000): ", 1000)
    som_mut_rate = get_user_input("Enter the Somatic Mutation rates up per (1000): ", 1000)
    lower_allele_range = get_user_input("Enter the lower value for alleles (2): ", 2)
    upper_allele_range = get_user_input("Enter the upper value for alleles (15): ", 15)
    Avg_Weights_Cell = get_user_input("Enter estimate number neurons per cell to set weight distribution (5, 5 to 50 - set to 1 to build network by charge quickly): ", 5)
    BIAS_RANGE = get_user_input_float("Enter the bias randome distributon positive range (.01, Range 0 to .1): ", .01)
    learning_rate = get_user_input_float("Enter the learning rate (0.01, Range .01 to .0001): ", 0.01)
    delta = get_user_input_float("Enter the charge delta to protect or prune a cell in protect/prune mode (0.01): ", 0.01)
    return germ_mut_rate, som_mut_rate, lower_allele_range, upper_allele_range, weight_change_threshold, NUM_LAYERS,learning_rate, BIAS_RANGE,Avg_Weights_Cell, delta

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("JMR's Game of Life with Genetics & Nerual Network")
font = pygame.font.SysFont(None, 24)
font_directory = pygame.font.SysFont(None, 16)

# Create a surface for the bottom caption
bottom_caption_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_EXTENSION)) #Default is black

# Create a subsurface that is the original size minus 100 pixels from the height
subsurface_rect = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
growthsurface = screen.subsurface(subsurface_rect)
growthsurface.fill(WHITE)

# Initialize cells with all None - cut in WIDTH and HEIGH in 4th since you now have 4 x 4 array
cells = np.full((WIDTH, HEIGHT, ARRAY_LAYERS), None, dtype=object) # change to array size. NEED for screen to work on inputs or errors,

# Create a set to store cells that had their charge changed
changed_cells_reverse = set()
changed_cells_forward = set()

# Dictionary to store cell types and their frequencies
cell_types = {}
phenotype_cell_types = {}
max_charge_diff = {}

# Initialize a list to keep track of the top 10 weight changes
top_weight_changes = []

# Mouse-button-up event tracking (to prevent continuous drawing while holding the button)
mouse_up = True

print(jmr_defs)
print(conways_defs)
print (how_network_works)

# Call the function to get the input values and set up your run
germ_mut_rate, som_mut_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers, LENGTH_OF_AXON, learning_rate, Bias_Range,Avg_Weights_Cell, charge_delta = get_input_values()

#cells = np.full((WIDTH, HEIGHT, NUM_LAYERS), None, dtype=object)  # crashes the cell placement, so just make big array and use the layers you use.

WEIGHT_MATRIX = 2*LENGTH_OF_AXON + 1 # Length of the side of the square of cells that axons can connect to
NUMBER_OF_WEIGHTS = WEIGHT_MATRIX*WEIGHT_MATRIX  # Number of weights for each cell

print (f"numer of weights: {NUMBER_OF_WEIGHTS}, weight matrix: {WEIGHT_MATRIX}, length of axon: {LENGTH_OF_AXON}, NUM_LAYERS: {Num_Layers}")

# Flag to indicate whether the simulation has started
display_updating = True
timing = False
running = False
prune = False
training_mode = False
andromida_mode = False
charge_change_protection = True
back_prop = False
training_data_loaded = False
not_saved_yet = True
simulating = True #Controls main Loop

prune_logic = "OR"  # initial value
display = "genes"
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

controls = """
    Mouse Down/Mouse Drag - Places a Cell                 Mouse Over - Get Cell Info & Statistics.            Close Window - Quit.
    Space - Toggle Running: {running} while running:
        A - Andromida_mode to allow normal growth rules: {andromida_mode}  
        C - Toggle Charge Change Protection: {charge_change_protection, delta :.4f}
        P - Toggle Prune: {prune} 
        + - Toggle Prune Logic: {prune_logic}
        
    T - Toggle Training: {training_mode}                                  
        I - Set Learning Rate: {learning_rate :.4f}
        F - Forward Prop, or R - Reverse Forward Prop: {direction_of_charge_flow}
        B - Toggle Back Prop: {back_prop}                                  
    
    G - Gene Display or W - Weight display: {display}   X - Reset weights and biases to random values
    H - Help    L - Load layers     S - Save layers     M - load MNEST training data    E - Enter Paramaters  N - Nuke all cells"""

print (controls) 
# Simulation loop
start_time = time.time()  # Record the start time
while simulating:

     # Autosave perfect networks - add once per training set or model loaded:)
    if max_bingo_count == how_much_training_data and not_saved_yet: # auto save for perfect networks
        save_file("-perfect")
        not_saved_yet = False
    
    for event in pygame.event.get():
        # Press H for help
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                print ("\n")
                print (controls) 

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

        # Press = to toggle prune_logic from OR to AND
        if event.type  == pygame.KEYDOWN:
            if event.key == pygame.K_EQUALS: # Press = to toggle prune_logic
                if prune_logic == "OR":
                    prune_logic = "AND"
                else:
                    prune_logic = "OR"
                print ("Prune Logic=", prune_logic)

        # Press C to start and stop Charge Change Protection      
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                charge_change_protection = not charge_change_protection
                print ("Charge_change_protection=", charge_change_protection)    

                if charge_change_protection:
                    old_delta = charge_delta
                    try:
                        charge_delta = float(input (f"Enter the charge delta to protect or prune a cell in protect/prune mode ({charge_delta}): ") or charge_delta)
                    except Exception:
                        charge_delta = old_delta
                        print("Invalid entry. Using old delta value: ", charge_delta)

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
                    confirm_quit = input("Are you sure you want to Nuke all cells? (y/n): ")
                    if confirm_quit.lower() == 'y':
                        for i in range(1, Num_Layers-1):
                            cells[:,:,i] = None
                            not_saved_yet = True
                            max_bingo_count = 0
                            bingo_count = 0
                            changed_cells_reverse = set()
                            changed_cells_forward = set()
                            total_loss, total_predictions, running_avg_loss = 0, 0, 0
                            training_cycles = 0
                            points = []
                    else:
                        print("Nuclear option cancelled.")
                except Exception as e:
                    print("An error occurred: ", e)   

        # Press 'I' to change the Iq or learning rate
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                old_learning_rate = learning_rate
                try:
                    learning_rate = float(input(f"Enter the learning rate (.01 to .001 default: {old_learning_rate}): ") or old_learning_rate)
                except Exception:
                    print("Invalid input. Reverting to old learning rate.", old_learning_rate)
                    learning_rate = old_learning_rate     
       
        #Press E Enter Paramater Resets 
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                # Call the function to get the input values and set up your run
                germ_mut_rate, som_mut_rate, lower_allele_range, upper_allele_range, weight_change_threshold, Num_Layers,learning_rate, Bias_Range,Avg_Weights_Cell, charge_delta = get_updated_input_values()

        #Press X Resets weights and biases to random values
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                for z in range(1, Num_Layers-1):
                    for x, y in np.ndindex(cells.shape[:2]):
                        if cells[x, y, z] is not None:
                            #cells[x, y, z].weights = [(np.random.normal(0,Weight_Range)) for _ in range(NUMBER_OF_WEIGHTS)]
                            cells[x, y, z].weights = np.random.randn(NUMBER_OF_WEIGHTS) / np.sqrt(Avg_Weights_Cell)
                            cells[x, y, z].bias = np.random.uniform(0,Bias_Range)
                            cells[x, y, z].charge = 0 # random charge between experiment since we keep charge ! compare to using zeros.
                            cells[x, y, z].error = 10e-8
                print ("Weights and biases reset to random values, in range of ", Avg_Weights_Cell, " and ", Bias_Range)
                not_saved_yet = True
                max_bingo_count = 0
                bingo_count = 0
                changed_cells_reverse = set()
                changed_cells_forward = set()
                total_loss, total_predictions, running_avg_loss = 0, 0, 0
                training_cycles = 0
                points = []

        # Press 'S' to Save the previously saved layers 0 to NUMLAYER-1 for  training
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                save_file("") 

        # Press 'L' to Load the previously saved layers 0 to NUMLAYER-1 - for  training
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                load_file()
                
        # Press 'M' to Load the MNEST training data for layers 0 and NUM_LAYERS-1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
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
                
        #Press 'W' for Display Modes Visualize Weights
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                display = "weights"
                print ("display =", display)
    
        #Press 'G' for Display Modes visualize Genes and proteins        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                display = "genes"
                print ("display =", display)       
        
        # Check for mouse-button-up
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_up = True

        # Mouse down places a Cell!             
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if y < WINDOW_HEIGHT: #Have window extension, don't want to mess up calculations, so only allow in main window
                cell_x, cell_y, layer = convert_x_y_to_index (x,y)
                cells[cell_x, cell_y, layer] = Cell(layer, NUMBER_OF_WEIGHTS) if cells[cell_x, cell_y, layer] is None else None
                mouse_up = False

        # Mouse down MAKES a Cell while holding mouse down and moving!
        if event.type == pygame.MOUSEMOTION:
            if not mouse_up:
                x, y = event.pos
                if y < WINDOW_HEIGHT: #Have window extension, don't want to mess up calculations, so only allow in main window
                    cell_x, cell_y, layer = convert_x_y_to_index (x,y)
                    if cells[cell_x, cell_y, layer] is None: cells[cell_x, cell_y, layer] = Cell(layer, NUMBER_OF_WEIGHTS ) 

        # Display Cell information & Statistics on MOUSEMOTION over cell
        if event.type == pygame.MOUSEMOTION:
            if mouse_up:
                x, y = event.pos
                if y < WINDOW_HEIGHT: #Have window extension, don't want to mess up calculations, so only allow in main window
                    cell_x, cell_y, layer = convert_x_y_to_index (x,y)
                    
                    if cells[cell_x, cell_y, layer]:
                        cell_types = update_cell_types(cells) 
                        count_pos, total_cells, phenotype_cell_types = update_phenotype_cell_types(cells) 
                        print (f"Number of cell {total_cells} | Positive cells {count_pos} | Fraction positive {count_pos/(total_cells+epsilon):.2f} | Number of weights {total_weights_list[0]} | Weight/Cell {total_weights_list[0]/(total_cells+epsilon):.2f}")
                        print ("\n")
                        print ("cell_x, cell_y, layer: ", cell_x, cell_y, layer)
                        print(cells[cell_x, cell_y, layer]) # Special function on the STR def for that class Cell
                        print ("\n")
                        display_averages() 
                        print ("\n")
                        display_statistics(cell_types)
                        print ("\n")
                        display_phenotype_statistics(phenotype_cell_types)
                        print ("\n")
                        display_max_charge_diff()
                        print ("\n")
                        print (f" Weight Range based on Average neuron selection: {Avg_Weights_Cell} | Bias_Range (random): {Bias_Range} ")
    if running: 
        update_cells()
    
    if training_mode:
        total_loss = 0 #Reset each epoch
        total_predictions = 0 # right now this is just the number of training data but with epochs or batches it will chnage
        training_cycles += 1
        training(learning_rate, LENGTH_OF_AXON)
        bottom_caption_surface.fill(BLACK)
        prediction_plot() # Average loss per training_cycle.

    if display_updating:
        growthsurface.fill(WHITE)
        draw_cells()
        draw_grid()

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time include sceen update!!
    
    if not training_mode: bottom_caption_surface.fill(BLACK)
    prune_color = GREEN if running else WHITE
    if prune: prune_color = RED 
    text_surface = font.render(f"Running = {running} | Prune = {prune}: {prune_logic} | Andromida = {andromida_mode} | Charge_protection = {charge_change_protection}: {charge_delta:.2e}", True, prune_color)
    text_surface1 = font.render(f"Training = {training_mode}: {learning_rate:.4f} | {direction_of_charge_flow} | Back_prop = {back_prop} | Loss: {running_avg_loss:.4f} | Cycles: {training_cycles}", True, WHITE)   
    text_surface2 = font.render(f"Elapsed: {elapsed_time:.2f} | Training_data: {how_much_training_data} | Correct: {bingo_count} | Max Correct: {max_bingo_count}", True, WHITE) 
    
    # Blit the text onto the bottom caption surface
    bottom_caption_surface.blit(text_surface, (10, 10))
    bottom_caption_surface.blit(text_surface1, (10, 40))
    bottom_caption_surface.blit(text_surface2, (10, 70))
    screen.blit(bottom_caption_surface, (0, EXTENDED_WINDOW_HEIGHT - WINDOW_EXTENSION))
    pygame.display.flip()
    start_time = time.time()  # Record the start time did at end since want to include time to update the information on screen