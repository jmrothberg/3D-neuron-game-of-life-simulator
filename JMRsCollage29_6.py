#JMRCollageMaker no duplicates unless needed
import os
import random
from PIL import Image
# Bypass the DecompressionBombError
Image.MAX_IMAGE_PIXELS = None
# Folder path containing the images
folder_path = "/Users/jonathanrothberg/AIArtC"
folder_out = "/Users/jonathanrothberg/AIArtOut"

# Get a list of all image file names in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith((".jpg", ".jpeg", ".png"))]

# Display the number of pictures in the folder
num_images = len(image_files)
print (f"You need to populate a {folder_path} with the images you want to make a collage from, and havea second folder {folder_out} for the output")
print(f"There are {num_images} pictures in the {folder_path}.")

# Allow the user to select an aspect ratio or enter custom dimensions
print("Please select an aspect ratio or enter custom dimensions:")
print("1. Square (1:1)")
print("2. Landscape (3:2)")
print("3. Portrait (2:3)")
print("4. Custom (up to 3:1 or 1:3)")
print("5. Custom (absolute dimensions)")

# Get the user's choice
choice = int(input("Enter your choice (1-5): "))

# Calculate the dimensions of the final rectangular image based on the chosen aspect ratio or custom dimensions
if choice == 1:
    cols = rows = int(num_images ** 0.5)  # Square aspect ratio
elif choice == 2:
    cols = int((num_images * 2) ** 0.5)  # Landscape aspect ratio (3:2)
    rows = (num_images + cols - 1) // cols
elif choice == 3:
    cols = int((num_images * 3) ** 0.5)  # Portrait aspect ratio (2:3)
    rows = (num_images + cols - 1) // cols
elif choice == 4:
    custom_ratio = input("Enter the custom aspect ratio (e.g., '3:1' or '1:3'): ")
    ratio_parts = custom_ratio.split(":")
    if len(ratio_parts) == 2:
        width_ratio = int(ratio_parts[0])
        height_ratio = int(ratio_parts[1])
        cols = int((num_images * width_ratio) ** 0.5)
        rows = (num_images + cols - 1) // cols
    else:
        print("Invalid custom aspect ratio. Exiting the program.")
        exit()
elif choice == 5:
    cols = int(input("Enter the number of columns: "))
    rows = int(input("Enter the number of rows: "))
else:
    print("Invalid choice. Exiting the program.")
    exit()


# Calculate the number of images needed to fill the desired aspect ratio or custom dimensions
num_required_images = cols * rows

# Check if there are enough images available
if num_required_images > num_images:
    print("Not enough unique images in {folder_path} to fill the desired aspect ratio or custom dimensions.")
    print(f"Available unique images: {num_images}, Required images: {num_required_images}")
    print(f"Will use random duplicates to fill up the remaining space.")

# Select the required number of images randomly from the available images, without repeating until necessary
selected_images = list(image_files)
if num_required_images > num_images:
    extra_images = num_required_images - num_images
    selected_images.extend(random.choices(image_files, k=extra_images))

# Create a blank canvas for the final image
canvas_width = cols * max(Image.open(os.path.join(folder_path, img)).size[0] for img in selected_images)
canvas_height = rows * max(Image.open(os.path.join(folder_path, img)).size[1] for img in selected_images)
canvas = Image.new("RGB", (canvas_width, canvas_height))

# Shuffle the list of selected image files
random.shuffle(selected_images)

# Paste the images onto the canvas
x_offset = 0
y_offset = 0
for image_file in selected_images:
    img = Image.open(os.path.join(folder_path, image_file))
    canvas.paste(img, (x_offset, y_offset))

    x_offset += img.width
    if x_offset >= canvas_width:
        x_offset = 0
        y_offset += img.height

# Display the final image
canvas.show()

# Save the final image with a random number in the file name
random_number = random.randint(1, 10000)
output_filename = f"Pictures_{random_number}.jpg"
output_path = os.path.join(folder_out, output_filename)
canvas.save(output_path)
print(f"The final image has been saved as '{output_filename}' in the '{folder_out}' folder.")
