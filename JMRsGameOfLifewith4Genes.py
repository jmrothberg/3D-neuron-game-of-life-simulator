#JMR's game of life with Genes and mutations!
#You ONLY makes new cell IF you EQUAL the BT Threshold
#Germline mutation happen at creation.
#Somatic mutations happen each cycle

import pygame
import numpy as np
import random
import copy

# Pygame configurations
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 1200
CELL_SIZE = 14
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [RED, GREEN, BLUE, YELLOW]

# Grid
WIDTH = WINDOW_WIDTH // CELL_SIZE
HEIGHT = WINDOW_HEIGHT // CELL_SIZE

my_defs = """The functionality of each gene:

Overcrowding Tolerance (OT): This gene determines the maximum number of neighbors a cell can have before it dies due to overcrowding. For example, if OT = 3, the cell will die if it has more than 3 neighbors.

Isolation Tolerance (IT): This gene determines the minimum number of neighbors a cell needs to survive. If a cell has fewer neighbors than its IT, it dies from isolation. For example, if IT = 2, a cell with fewer than 2 neighbors will die.

Birth Threshold (BT): This gene determines the number of neighbors an empty cell needs to produce a new cell. If an empty cell has exactly this many neighbors, a new cell will be born.
For example, if BT = 3, a new cell will be born in an empty cell that has exactly 3 neighbors. And inherit the genes by recobination between two parents ONE of whom gave it the BT = 3 allele.

Mutation Rate (MR): This gene determines the chance of a cell's genes mutating when it is reproduced. It is a percentage, with a higher value meaning a higher chance of mutation.
For example, if MR = 1, there is a 1% chance (If rate was selected per 100) each gene will mutate when a new cell is born. A mutation in the BT gene gives a new cell the ability to be BORN
even if neither parents had a BT gene allowing it to be born in that position. Not until the next cycle will a cell be checked for OT & IT.
Mutation here means that the gene's value could randomly change to any value between 0 and 7 (or whatever range is selected by the user).

It's important to note that in JMR's modified version of the Game of Life, each cell can have different values for these genes, leading to diverse behaviors in the simulation.
Cells inherit these genes from their parent cells (the ones they were born from), with a chance of mutation based on the their inherited Mutation Rate.

Each square from top left to is colored Red, Green,
                                        Blue, yellow
The brightness of the square is based on the allele value for each of those genes:
                                        OT IT
                                        BT MR"""

conways_defs = """
The original Conway's Game of Life has a fixed set of rules that apply to all cells equally:

Overcrowding Tolerance (OT): Any live cell with more than three (3) live neighbors dies, as if by overpopulation (it's as if the cell is overcrowded).
Isolation Tolerance (IT): Any live cell with fewer than two(2) live neighbors dies, as if by underpopulation (it's as if the cell is isolated).

Birth Threshold (BT): Any cell location with exactly three (3) live neighbors becomes a live cell, as if by reproduction (a new cell is born).
In other words, in the original game, the Overcrowding Tolerance is 3, Isolation Tolerance is 2, and the Birth Threshold is 3.
There is no concept of mutation rate in the original game, as rules are static and don't change."""

def get_user_input(prompt, default_value):
    user_input = input(prompt)
    if user_input == "":
        return default_value
    else:
        return int(user_input)

def get_input_values():
    germ_mut_rate = get_user_input("Enter the Germline Mutation rates per (10000): ", 10000)
    som_mut_rate = get_user_input("Enter the Somatic Mutation rates up per (1000): ", 1000)
    lower_allele_range = get_user_input("Enter the lower value for alleles (2): ", 2)
    upper_allele_range = get_user_input("Enter the upper value for alleles (3): ", 3)

    return germ_mut_rate, som_mut_rate, lower_allele_range, upper_allele_range

class Cell:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = [random.randint(lower_allele_range, upper_allele_range) for _ in range(4)]  # Generate 4 random alleles of the four genes
        else:
            self.genes = genes
        self.colors = [tuple(int(gene * color_component // 8) for color_component in color) for gene, color in zip(self.genes, COLORS)]

def draw_grid():
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            rectangle = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rectangle, 1)

def draw_cells():
    for (x, y) in np.ndindex(cells.shape):
        if cells[x, y] is not None:
            for i in range(4):
                pygame.draw.rect(screen, cells[x, y].colors[i], pygame.Rect(x*CELL_SIZE + (i%2)*(CELL_SIZE//2), y*CELL_SIZE + (i//2)*(CELL_SIZE//2), CELL_SIZE//2, CELL_SIZE//2))
        else:
            pygame.draw.rect(screen, WHITE, pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def update_cells():
    global cells

    new_cells = copy.deepcopy(cells)

    for (x, y) in np.ndindex(cells.shape):
        neighbors = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]
        alive_neighbors = [cells[(x+dx)%cells.shape[0], (y+dy)%cells.shape[1]] for dx, dy in neighbors if cells[(x+dx)%cells.shape[0], (y+dy)%cells.shape[1]] is not None]
        num_alive = len(alive_neighbors)
        
        if cells[x, y] is not None:
            #mutate cell before checking if it should die
            cells[x, y].genes = [gene if random.randint(1, som_mut_rate) > cells[x, y].genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in cells[x, y].genes]
            
            if (num_alive <= cells[x, y].genes[1] or num_alive >= cells[x, y].genes[0]):
                new_cells[x, y] = None
        
        elif cells[x, y] is None and alive_neighbors:
            potential_parents = alive_neighbors # [cell for cell in alive_neighbors if cell.genes[2] <= num_alive] # if wanted only BT parents
            if len(potential_parents) >= 2:
                parent1, parent2 = random.sample(potential_parents, 2)
                new_genes = [random.choice([parent1.genes[i], parent2.genes[i]]) for i in range(4)]
                new_genes = [gene if random.randint(1, 100) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]
                if num_alive == new_genes[2]:  # Check if BT is met <= Threshold or exact :)
                    new_cells[x, y] = Cell(new_genes)
            elif potential_parents:
                parent1 = random.choice(potential_parents)
                new_genes = parent1.genes
                new_genes = [gene if random.randint(1, germ_mut_rate) > new_genes[3] else random.randint(lower_allele_range, upper_allele_range) for gene in new_genes]
                if num_alive == new_genes[2]:  # Check if BT is met <=
                    new_cells[x, y] = Cell(new_genes)

    cells = new_cells

# Dictionary to store cell types and their frequencies
cell_types = {}

def update_cell_types(cells):
    global cell_types
    cell_types = {}
    
    for (x, y) in np.ndindex(cells.shape):
        if cells[x, y] is not None:
            cell_type = tuple(cells[x, y].genes)
            cell_types[cell_type] = cell_types.get(cell_type, 0) + 1

def display_statistics():
    
    sorted_types = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:10]
    
    font = pygame.font.SysFont(None, 24)
    statistics = ["OT IT BT MR: Overcrowding Isolation Birth Mutation:"] + [f"Type: {cell_type} | Count: {count}" for cell_type, count in sorted_types]
    
    for i, stat in enumerate(statistics):
        
        text = font.render(stat, True, BLACK)
        if running: print (stat)
        screen.blit(text, (10, 20 + i * 30))


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("JMR's Game of Life with Genetics")
# Initialize cells with all None
cells = np.full((WIDTH, HEIGHT), None, dtype=object)

# Flag to indicate whether the simulation has started
running = False

# Mouse-button-up event tracking (to prevent continuous drawing while holding the button)
mouse_up = True

print (my_defs)
print (conways_defs)

# Call the function to get the input values
germ_mut_rate, som_mut_rate, lower_allele_range, upper_allele_range = get_input_values()

simulating = True
# Game loop
screen.fill(WHITE)

while simulating:
    draw_cells()
    #draw_grid()
    display_statistics()
    pygame.display.update()

    if running:
        update_cells()
        update_cell_types(cells)

    for event in pygame.event.get():
        
        # Press SPACE to start and stop the simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = not running
                
        # Press q to end simulation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                simulating = False
                
        # Check for mouse button click to place or remove cells
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            x = min(x // CELL_SIZE, WIDTH - 1)
            y = min(y // CELL_SIZE, HEIGHT - 1)
            cells[x, y] = Cell() if cells[x, y] is None else None
            mouse_up = False

        # Check for mouse-button-up
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_up = True

        # Check for mouse movement with button pressed
        if event.type == pygame.MOUSEMOTION:
            if not mouse_up:
                x, y = event.pos
                x = min(x // CELL_SIZE, WIDTH - 1)
                y = min(y // CELL_SIZE, HEIGHT - 1)      
                cells[x, y] = Cell() if cells[x, y] is None else None
                
        # Display gene information on mouse over
        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            x = min(x // CELL_SIZE, WIDTH - 1)
            y = min(y // CELL_SIZE, HEIGHT - 1)
            if cells[x, y]:
                print("\n")
                print(f"Overcrowding Tolerance: {cells[x][y].genes[0]}")
                print(f"Isolation Tolerance: {cells[x][y].genes[1]}")
                print(f"Birth Threshold: {cells[x][y].genes[2]}")
                print(f"Mutation Rates: {cells[x][y].genes[3]} per {som_mut_rate} & per {germ_mut_rate}")
                
    clock.tick(FPS)

