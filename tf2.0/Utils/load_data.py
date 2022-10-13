import numpy as np
import random
import requests
import string
import tarfile
import tensorflow as tf

########## Load Numpy Arrays  #########
evens = np.arange(0, 100, step=2, dtype=np.int32)
evens_label = np.zeros(50, dtype=np.int32)
odds = np.arange(1, 100, step=2, dtype=np.int32)
odds_label = np.ones(50, dtype=np.int32)
# Concatenate arrays
features = np.concatenate([evens, odds])
labels = np.concatenate([evens_label, odds_label])

# Load a numpy array using tf data api with `from_tensor_slices`.
data = tf.data.Dataset.from_tensor_slices((features, labels))
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=100)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

for batch_x, batch_y in data.take(5):
    print(batch_x, batch_y)

# Note: If you are planning on calling multiple time,
# you can user the iterator way:
ite_data = iter(data)
for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

###########   Load CSV files   #########
# Download Titanic dataset (in csv format).
d = requests.get("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv")
with open("titanic_dataset.csv", "wb") as f:
    f.write(d.content)

# Load Titanic dataset.
# Original features: survived,pclass,name,sex,age,sibsp,parch,ticket,fare
# Select specific columns: survived,pclass,name,sex,age,fare
column_to_use = [0, 1, 2, 3, 4, 8]
record_defaults = [tf.int32, tf.int32, tf.string, tf.string, tf.float32, tf.float32]

# Load the whole dataset file, and slice each line.
data = tf.data.experimental.CsvDataset("titanic_dataset.csv", record_defaults, header=True, select_cols=column_to_use)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)
# Batch data (aggregate records together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

for survived, pclass, name, sex, age, fare in data.take(1):
    print(survived.numpy())
    print(pclass.numpy())
    print(name.numpy())
    print(sex.numpy())
    print(age.numpy())
    print(fare.numpy())

#############   Load Images   ############
# Download Oxford 17 flowers dataset
d = requests.get("http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
with open("17flowers.tgz", "wb") as f:
    f.write(d.content)
# Extract archive.
with tarfile.open("17flowers.tgz") as t:
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(t)

with open('jpg/dataset.csv', 'w') as f:
    c = 0
    for i in range(1360):
        f.write("jpg/image_%04i.jpg,%i\n" % (i+1, c))
        if (i+1) % 80 == 0:
            c += 1

# Load Images
with open("jpg/dataset.csv") as f:
    dataset_file = f.read().splitlines()

# Load the whole dataset file, and slice each line.
data = tf.data.Dataset.from_tensor_slices(dataset_file)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)

# Load and pre-process images.
def load_image(path):
    # Read image from path.
    image = tf.io.read_file(path)
    # Decode the jpeg image to array [0, 255].
    image = tf.image.decode_jpeg(image)
    # Resize images to a common size of 256x256.
    image = tf.image.resize(image, [256, 256])
    # Rescale values to [-1, 1].
    image = 1. - image / 127.5
    return image
# Decode each line from the dataset file.
def parse_records(line):
    # File is in csv format: "image_path,label_id".
    # TensorFlow requires a default value, but it will never be used.
    image_path, image_label = tf.io.decode_csv(line, ["", 0])
    # Apply the function to load images.
    image = load_image(image_path)
    return image, image_label
# Use 'map' to apply the above functions in parallel.
data = data.map(parse_records, num_parallel_calls=4)

# Batch data (aggregate images-array together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

for batch_x, batch_y in data.take(1):
    print(batch_x, batch_y)

###############   Load data from a Generator  ###########
# Create a dummy generator.
def generate_features():
    # Function to generate a random string.
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters) for m in xrange(length))
    # Return a random string, a random vector, and a random int.
    yield random_string(4), np.random.uniform(size=4), random.randint(0, 10)

# Load a numpy array using tf data api with `from_tensor_slices`.
data = tf.data.Dataset.from_generator(generate_features, output_types=(tf.string, tf.float32, tf.int32))
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=100)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

# Display data.
for batch_str, batch_vector, batch_int in data.take(5):
    print(batch_str, batch_vector, batch_int)
