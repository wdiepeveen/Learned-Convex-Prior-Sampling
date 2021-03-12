import numpy as np
import os
import glob

"""
This script generates the ground truth train and test data for the l1_toy_example
"""

data_dim = 10

size_training_data = 500
size_test_data = 100

# Set seed
np.random.seed(seed=0)

results_dir = os.path.join("..", "data", "l1_toy_data")
prior_dir = "groundtruth"
# fidelity_dir

# Generate training data
training_dir = os.path.join(results_dir, "train")

for i in range(size_training_data):
    # Generate prior data
    sample = np.random.laplace(loc=0, scale=1, size=(data_dim, data_dim))
    filename = 'sample_{}'.format(str(i+1).zfill(4))
    np.save(os.path.join(training_dir, prior_dir, filename + ".npy"), sample)

    # Generate l2 fidelity data


# Generate test data
test_dir = os.path.join(results_dir, "test")

for i in range(size_test_data):
    # Generate prior data
    sample = np.random.laplace(loc=0, scale=1, size=(data_dim, data_dim))
    filename = 'sample_{}'.format(str(i+1).zfill(4))
    np.save(os.path.join(test_dir, prior_dir, filename + ".npy"), sample)

    # Generate l2 fidelity data
