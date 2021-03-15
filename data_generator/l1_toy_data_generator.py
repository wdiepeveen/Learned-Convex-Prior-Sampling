import numpy as np
import os
import glob

"""
This script generates the ground truth train and test data for the l1_toy_example
"""

data_dim = 10

size_training_data = 500
size_test_data = 100

size_sup_training_data = 1000
size_sup_test_data = 20

# Set seed
np.random.seed(seed=0)

results_dir = os.path.join("..", "data", "l1_toy_data")
prior_dir = "groundtruth"
# fidelity_dir

labeled_data_dir = "labeled"

# Generate training data
training_dir = os.path.join(results_dir, "train")

def mye_l1(lambd,x):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.abs(x[i,j]) < lambd:
                z[i,j] = 1/(2*lambd) * x[i,j]**2
            else:
                z[i,j] = np.abs(x[i,j]) - lambd/2

    mye = np.sum(z)

    return mye


for i in range(size_training_data):
    # Generate prior data
    sample = np.random.laplace(loc=0, scale=1, size=(data_dim, data_dim))
    filename = 'sample_{}'.format(str(i+1).zfill(4))
    np.save(os.path.join(training_dir, prior_dir, filename + ".npy"), sample)

    # Generate l2 fidelity data

for i in range(size_sup_training_data):
    # Generate labels
    labeled_sample = np.random.laplace(loc=0, scale=1, size=(data_dim, data_dim))
    label = np.sum(mye_l1(1, labeled_sample))
    filename = 'sample_{}'.format(str(i + 1).zfill(4))
    np.save(os.path.join(training_dir, labeled_data_dir, "data", filename + ".npy"), labeled_sample)
    np.save(os.path.join(training_dir, labeled_data_dir, "labels", filename + ".npy"), label)


# Generate test data
test_dir = os.path.join(results_dir, "test")

for i in range(size_test_data):
    # Generate prior data
    sample = np.random.laplace(loc=0, scale=1, size=(data_dim, data_dim))
    filename = 'sample_{}'.format(str(i+1).zfill(4))
    np.save(os.path.join(test_dir, prior_dir, filename + ".npy"), sample)

    # Generate l2 fidelity data

for i in range(size_sup_test_data):
    # Generate labels
    labeled_sample = np.random.laplace(loc=0, scale=1, size=(data_dim, data_dim))
    label = mye_l1(1, labeled_sample)
    filename = 'sample_{}'.format(str(i + 1).zfill(4))
    np.save(os.path.join(test_dir, labeled_data_dir, "data", filename + ".npy"), labeled_sample)
    np.save(os.path.join(test_dir, labeled_data_dir, "labels", filename + ".npy"), label)

