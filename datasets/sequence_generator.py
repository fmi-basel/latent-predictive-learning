from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
import time
from tqdm import tqdm

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
_PERMUTATIONS_PER_FACTOR = {}

# generate num_vectors random vectors of dim num_dims and length at sqrt(num_dims)
def sample_random_vectors(num_vectors, num_dims):
    """ Samples many vectors of dimension num_dims and length sqrt(num_dims)
    Args:
        num_vectors: number of vectors to sample.
        num_dims: dimension of vectors to sample.
    
    Returns:
        batch: vectors shape [batch_size,num_dims]
    """
    vectors = np.random.normal(size=[num_vectors, num_dims])
    lengths = np.sqrt(np.sum(vectors**2, axis=1))
    vectors = np.sqrt(num_dims) * vectors / lengths[:, np.newaxis]
    # discretize vectors to the closest integer (negative values rounded down, positive values rounded up)
    vectors = np.round(vectors)
    # if any vector is all zeros, then set random coordinate to 1
    vectors[np.sum(vectors**2, axis=1) == 0, np.random.randint(num_dims)] = 1
    return vectors

def sample_one_hot_vectors(num_vectors, num_dims):
    """ Samples many one-hot vectors of dimension num_dims
    Args:
        num_vectors: number of vectors to sample.
        num_dims: dimension of vectors to sample.
    
    Returns:
        batch: vectors shape [batch_size,num_dims]
    """
    non_zero_indices = np.random.randint(num_dims, size=num_vectors)
    vectors = np.eye(num_dims)[non_zero_indices]
    # multiply random sign to each vector
    signs = np.random.choice([-1, 1], size=num_vectors)
    # select one-quarter of the vectors and multiply by 2
    signs[np.random.choice(num_vectors, size=num_vectors//4)] *= 2
    vectors = vectors * signs[:, np.newaxis]
    return vectors


def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
        factors: np array shape [6,batch_size].
                factors[i]=factors[i,:] takes integer values in 
                range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
        indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


def generate_trajectories(num_sequences=64000, batch_size=16):

    current_state = np.zeros([num_sequences, len(_FACTORS_IN_ORDER)], dtype=np.int8)
    trajectories = np.zeros([num_sequences, batch_size+1, len(_FACTORS_IN_ORDER)], dtype=np.int8)

    for k, factor in enumerate(_FACTORS_IN_ORDER):
        current_state[:, k] = np.random.choice(_NUM_VALUES_PER_FACTOR[factor], num_sequences)
    trajectories[:, 0] = current_state

    directions = sample_one_hot_vectors(num_sequences, len(_FACTORS_IN_ORDER)-1)
    directions = np.insert(directions, _FACTORS_IN_ORDER.index('shape'), 0, axis=1)
    
    for i in tqdm(range(1, batch_size+1)):
        for factor in _FACTORS_IN_ORDER:
            if factor == 'floor_hue' or factor == 'wall_hue' or factor == 'object_hue':
                current_state[:, _FACTORS_IN_ORDER.index(factor)] = (current_state[:, _FACTORS_IN_ORDER.index(factor)] + directions[:, _FACTORS_IN_ORDER.index(factor)]) % _NUM_VALUES_PER_FACTOR[factor]
            if factor == 'scale' or factor == 'orientation':
                current_state[:, _FACTORS_IN_ORDER.index(factor)] = np.clip(current_state[:, _FACTORS_IN_ORDER.index(factor)] + directions[:, _FACTORS_IN_ORDER.index(factor)], 0, _NUM_VALUES_PER_FACTOR[factor]-1)
                directions[:, _FACTORS_IN_ORDER.index(factor)] = np.where(current_state[:, _FACTORS_IN_ORDER.index(factor)] == 0, -directions[:, _FACTORS_IN_ORDER.index(factor)], directions[:, _FACTORS_IN_ORDER.index(factor)])
                directions[:, _FACTORS_IN_ORDER.index(factor)] = np.where(current_state[:, _FACTORS_IN_ORDER.index(factor)] == _NUM_VALUES_PER_FACTOR[factor]-1, -directions[:, _FACTORS_IN_ORDER.index(factor)], directions[:, _FACTORS_IN_ORDER.index(factor)])
        trajectories[:, i] = current_state
    
    flattened_trajectories = trajectories.reshape(-1, len(_FACTORS_IN_ORDER)).astype(np.int32)
    # exchange values of each factor according to _PERMUTATIONS_PER_FACTOR
    for i, factor in enumerate(_FACTORS_IN_ORDER):
        flattened_trajectories[:, i] = _PERMUTATIONS_PER_FACTOR[factor][flattened_trajectories[:, i]]
    indexed_trajectories = get_index(flattened_trajectories.T)

    return indexed_trajectories

if __name__ == '__main__':
    data_dir = os.path.expanduser("~/data/datasets/shapes3d")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    shuffle_colors = True

    permutations_file_path = os.path.join(data_dir, 'permutations.npy')
    if shuffle_colors:
        trajectory_file_path = os.path.join(data_dir, 'indexed_trajectories.npy')
        if not os.path.exists(permutations_file_path):
            print('did not find permutations file, creating new shuffling of colors ...')
            for factor in _FACTORS_IN_ORDER:
                if factor == 'floor_hue' or factor == 'wall_hue' or factor == 'object_hue':
                    _PERMUTATIONS_PER_FACTOR[factor] = np.random.permutation(_NUM_VALUES_PER_FACTOR[factor])
                else:
                    _PERMUTATIONS_PER_FACTOR[factor] = np.arange(_NUM_VALUES_PER_FACTOR[factor])
            np.save(permutations_file_path, _PERMUTATIONS_PER_FACTOR)
        else:
            print('found permutations file, loading ...')
            _PERMUTATIONS_PER_FACTOR = np.load(permutations_file_path, allow_pickle=True).item()
    else:
        trajectory_file_path = os.path.join(data_dir, 'indexed_trajectories_no_shuffle.npy')
        print('not shuffling colors')
        _PERMUTATIONS_PER_FACTOR = {factor: np.arange(_NUM_VALUES_PER_FACTOR[factor]) for factor in _FACTORS_IN_ORDER}

    if os.path.exists(trajectory_file_path):
        print('trajectories file {} already exists, please delete it first if you want to regenerate it'.format(trajectory_file_path))
    else:
        print("generating training data...")
        print("destination directory: {}".format(data_dir))
        indexed_trajectories = generate_trajectories(num_sequences=64000, batch_size=16)

        print("saving training data...")
        np.save(trajectory_file_path, indexed_trajectories)
        print('saved trajectories to {}'.format(trajectory_file_path))
