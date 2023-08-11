#JMR mnist to cells!!! Jul 14th 2023
#Zero full 28 x 28, 15 is 1 x 10
import os
from tensorflow.keras.datasets import mnist
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

output_path_training = '/Users/jonathanrothberg/MNIST_as_cells_training_full_in_out'
#output_path_test = '/Users/jonathanrothberg/MNIST_as_cells_test'

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
COLORS = [RED, GREEN, BLUE, YELLOW, ORANGE, INDIGO, VIOLET, PINK ]
upper_allele_range = 28

class Cell:
    def __init__(self, layer, genes=None):
        self.layer = layer  
        if genes is None:
            self.genes = [0, 0, 0, 0, 0, 0, 0, 0]  # For training cells
        else:
            self.genes = genes
        #self.colors = [tuple(int(gene * color_component // 28) for color_component in color) for gene, color in zip(self.genes, COLORS)]
        self.colors =  [tuple(min(int(gene * color_component // upper_allele_range), 255) for color_component in color) for gene, color in zip(self.genes, COLORS)]

        # Assign random weights - eventually will tune these based on some back propagation value from the end "output"
        self.weights =  [0 for _ in range(9)]  # Generate 9 random weights for the cell
 
        self.bias = 0 #random.uniform(-1, 1)
       
        self.error = 0.0
        #** Assign the charge based on the layer the NMIST DATA
        self.charge = 1 # 1 if self.layer == 0 or self.layer == 15 else 0
       
    def __str__(self):
        return f"Cell:\nlayer={self.layer}\nGenes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, CH={self.genes[4]}, WG={self.genes[5]}, ER={self.genes[6]}, BI={self.genes[7]} \ncharge={self.charge}, \nweights={self.weights}, \nerror={self.error}, bias ={self.bias})"

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize cells with all None
cells = np.full((28, 28, 16), None, dtype=object)

# Function to create a stable cell with a given charge
def create_stable_cell(layer, charge):  
    cell = Cell(layer)  # Create a cell at the specified layer
    cell.charge = charge  # Set the charge
    cell.weights = [charge for _ in range(9)]
    return cell

fig, axs = plt.subplots(20, 20, figsize=(10, 10)) 

for k in range(200):  # For the first 10 images
    # Create a 28x28 array with zeros
    output_array = np.zeros((28, 28))
    # Convert the label into a one-hot encoded 1x10 array
    one_hot_label = (train_labels[k] == np.arange(10)).astype(int)
    # Place the one-hot encoded array in the middle of the 28x28 array
    output_array[14, 9:19] = one_hot_label
    for i in range(28):
        for j in range(28):
            # Set the charge in layer 0 to match the input layer
            cells[i, j, 0] = create_stable_cell(0, train_images[k, i, j])
            # Set the charge in layer 15 to match the solution
            if i == 14 and 9 <= j < 19:
                cells[i, j, 15] = create_stable_cell(15, output_array[i, j]) 
            else:
                cells[i, j, 15] = None
                
    # Save the current state of layers 0 and 15 to a file
    file_path = os.path.join(output_path_training, f'simulation_state_layers_0_and_15_image_{k}.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump((cells[:,:,0], cells[:,:,15]), f)

    if k < 200:
         # Display the input image
        input_array = np.array([[cell.charge if cell is not None else 0 for cell in row] for row in cells[:,:,0]])
        
        axs[k // 10, k % 10 * 2 ].imshow(input_array, cmap='gray')
        axs[k // 10, k % 10 * 2 ].axis('off')  # Hide axes

        # Display the output image
        output_image = np.array([[cell.charge if cell is not None else 0 for cell in row] for row in cells[:,:,15]])
        
        #output_image = np.array([[cell.charge for cell in row] for row in cells[:,:,15]])
        axs[k // 10, k % 10 * 2 + 1 ].imshow(output_image, cmap='gray')
        axs[k // 10, k % 10 * 2 + 1 ].axis('off')  # Hide axes
    if k == 199: 
        plt.show()
     

    if k % 100 == 0:  # Print progress every 100 images
        print(f"Number of NMIST data pairs turned into cell layers: {k}")
        
        