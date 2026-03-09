import os
import urllib.request
import gzip
import shutil

# Define the URLs for the MNIST dataset
urls = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}

# Define the output directory for the MNIST dataset
output_dir = '/Users/jonathanrothberg/mnist'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download and extract the MNIST dataset files
for name, url in urls.items():
    print(f"Downloading {name}...")
    response = urllib.request.urlopen(url)
    with open(os.path.join(output_dir, f"{name}.gz"), 'wb') as f:
        shutil.copyfileobj(response, f)
    with gzip.open(os.path.join(output_dir, f"{name}.gz"), 'rb') as f_in:
        with open(os.path.join(output_dir, f"{name}"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(os.path.join(output_dir, f"{name}.gz"))

print("MNIST dataset downloaded and extracted successfully!")