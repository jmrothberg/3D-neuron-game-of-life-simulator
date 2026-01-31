#Jonathan Marc Rothberg MNIST to cells!!! Jul 14th 2023
#layer 0 is 28 x 28 , 15 is 1 x 10
import os
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib
import datetime
import tensorflow as tf
import ssl # this is needed for the tensorflow to work with the internet
import urllib.request # this is needed for the tensorflow to work with the internet and download fashion_mnist or I got security errors 
from tensorflow.keras.datasets import fashion_mnist 

matplotlib.use('TkAgg') # this is needed for the tkinter to work with matplotlib

# Create an SSL context with verification disabled - otherewise it refuses to download the fashion_mnist data
ssl_context = ssl._create_unverified_context()

# Install the SSL context globally
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

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
       
        self.colors =  [tuple(min(int(gene * color_component // upper_allele_range), 255) for color_component in color) for gene, color in zip(self.genes, COLORS)]

        self.weights =  [0 for _ in range(9)]  # Generate 9 0 weights for the cell
 
        self.bias = 0 
       
        self.error = 0
       
        self.charge = 0 
       
    def __str__(self):
        return f"Cell:\nlayer={self.layer}\nGenes: OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}, CH={self.genes[4]}, WG={self.genes[5]}, ER={self.genes[6]}, BI={self.genes[7]} \ncharge={self.charge}, \nweights={self.weights}, \nerror={self.error}, bias ={self.bias})"


def load_dataset(dataset_name):
    if dataset_name == 1:
        from tensorflow.keras.datasets import fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    elif dataset_name == 2:
        from tensorflow.keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif dataset_name == 3:
        from tensorflow.keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    elif dataset_name == 4:
        from tensorflow.keras.datasets import cifar100
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
    else:
        print("Invalid selection. Please choose a number between 1 and 4.")
        return None

    return (train_images, train_labels), (test_images, test_labels)

# User selects the dataset
print("Select the dataset you want to load:")
print("1: Fashion MNIST")
print("2: MNIST")
print("3: CIFAR-10")
print("4: CIFAR-100")

dataset_name = int(input("Enter the number of the dataset you want to load: "))
(train_images, train_labels), (test_images, test_labels) = load_dataset(dataset_name)

today = datetime.date.today().strftime('%Y-%m-%d')

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Get the number of training pairs to create
num_training_pairs = int(input("Enter the number of training pairs to create: "))

# Get the output directory for the training data using a Tkinter dialog box
root = tk.Tk()
root.withdraw()
output_path_training = filedialog.askdirectory(title="Create output directory for training data")

# Create the output directory if it doesn't exist
if not os.path.exists(output_path_training):
    os.makedirs(output_path_training)

output_figure_dir = os.path.join(output_path_training, "NMIST_CELL_TRAINING_PLOTS")
os.makedirs(output_figure_dir, exist_ok=True)

# Initialize cells with all None
cells = np.full((28, 28, 16), None, dtype=object)

# Function to create a stable cell with a given charge
def create_stable_cell(layer, charge):  
    cell = Cell(layer)  # Create a cell at the specified layer
    cell.charge = charge  # Set the charge
   
    return cell

plt.ion()  # Turn on interactive mode
fig, axs = plt.subplots(20, 20, figsize=(10, 10)) 
k1 = 0
fig_num = 0
for k in range(num_training_pairs ):  # For the first num_training_pairs  images
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

    if k1 < 200:
         # Display the input image
        input_array = np.array([[cell.charge if cell is not None else 0 for cell in row] for row in cells[:,:,0]])
        
        axs[k1 // 10, k1 % 10 * 2 ].imshow(input_array, cmap='gray')
        axs[k1 // 10, k1 % 10 * 2 ].axis('off')  # Hide axes

        # Display the output image
        output_image = np.array([[cell.charge if cell is not None else 0 for cell in row] for row in cells[:,:,15]])
        
        axs[k1 // 10, k1 % 10 * 2 + 1 ].imshow(output_image, cmap='gray')
        axs[k1 // 10, k1 % 10 * 2 + 1 ].axis('off')  # Hide axes
    k1 += 1 
    if k1 == 200: 
        fig_num += 1
        plt.show()
        # Save the current state of the plot to a file
        plt.savefig(os.path.join(output_figure_dir, f'plot_{str(fig_num)}_{today}.png'))
        plt.pause(2)
        k1 = 0
        fig, axs = plt.subplots(20, 20, figsize=(10, 10)) 
    elif k == num_training_pairs - 1:
        plt.ioff()
        for ax in axs.flat:
            ax.axis('off')  # Hide axes for all cells
        plt.show()
        plt.savefig(os.path.join(output_path_training,"NMIST_CELL_TRAINING_FIGURES", f'plot_{fig}_{today}.png'))

    if (k + 1) % 100 == 0:  # Print progress every 100 images
        print(f"Number of NMIST data pairs turned into cell training layer 0 & 15 pairs: {k+1}")
print(f"Final Number of NMIST data pairs turned into cell training layer 0 & 15 pairs: {k+1}")      