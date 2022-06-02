import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np

from metrics.losses import cosine_push_loss, uniform_loss, kl_divergence_to_standard_gaussian, \
    sliced_wasserstein_distance_to_uniform_hypercube, sliced_wasserstein_distance_to_standard_gaussian, \
    sliced_wasserstein_distance_to_uniform_hypersphere


# TODO: Convert to lightning metrics


def compute_cosine_measures(X, y, diagnostic_mode=False):
    cos_dists = pairwise_cosine_similarity(X - X.mean(dim=0), zero_diagonal=False)

    cross_class_cos_mat = cos_dists
    within_class_cos_mats = []

    exclusion_mask = torch.tril(torch.ones_like(cos_dists, dtype=torch.bool, device=cos_dists.device))
    within_class_mean = 0

    for i in torch.unique(y):
        idxs = torch.nonzero(y == i, as_tuple=False).squeeze()

        for i in idxs:
            exclusion_mask[i, idxs] = True
        cos_dists_of_class = cos_dists[idxs][:, idxs]
        num_samples = cos_dists_of_class.shape[0]

        within_class_cos_mats.append(cos_dists_of_class)

        within_class_mean += cos_dists_of_class[np.triu_indices(num_samples, k=1)].mean()

    within_class_mean /= len(torch.unique(y))

    mask = ~exclusion_mask
    cross_class_mean = (cross_class_cos_mat * mask).sum() / mask.sum()
    cross_class_mean = cross_class_mean

    if not diagnostic_mode:
        return cross_class_mean, within_class_mean
    else:
        return cross_class_cos_mat, within_class_cos_mats, mask


def compute_manifold_measures(X, y, diagnostic_mode=False):
    """
    Function to calculate simple measures of manifold radius and dimension
    Also returns the correlation matrix of the features belonging to different classes
    :param X: (torch.Tensor) Feature matrix (batch should be along the first dimension)
    :param y: (torch.Tensor) Label vector
    :return: (three torch.Tensors)
            class correlation matrix, manifold radii for each class, participation ratio (linear measure of
            manifold dimensionality) for each class
    """
    class_centers = []
    class_radii = []
    class_dims = []

    global_center = X.mean(dim=0)
    global_radius = torch.norm(X - global_center, dim=1).mean()

    for i in torch.unique(y):
        x = X[y == i]

        pr = participation_ratio(x)
        class_center = x.mean(dim=0)
        class_radius = torch.norm(x - class_center, dim=1).mean()

        class_centers.append(class_center)
        class_radii.append(class_radius)
        class_dims.append(pr)

    class_centers = torch.stack(class_centers)
    class_radii = torch.stack(class_radii)
    class_dims = torch.stack(class_dims)
    centroid_spreads = torch.norm(class_centers - global_center, dim=1)

    class_correlation_matrix = correlation(class_centers)

    if not diagnostic_mode:
        class_correlation = class_correlation_matrix[np.triu_indices(class_correlation_matrix.shape[0], k=1)].mean()
        return class_correlation, class_radii.mean()/global_radius, class_dims.mean(), global_radius
    else:
        return class_correlation_matrix, class_radii, class_dims, centroid_spreads, global_radius


def participation_ratio(x):
    """
    Calculates participation ratio of the correlation matrix of the input
    pr = (sum of eigenvalues)^2 / sum of (eigenvalues^2)
    :param x: (torch.tensor) input feature tensor (batch x features)
    :return: (torch.tensor)
             participation ratio (singleton tensor)
    """
    x_c = x - x.mean(dim=0)

    corr_matrix = torch.einsum('ij,kj->ik', x_c, x_c)

    eig_vals, _ = torch.linalg.eig(corr_matrix)
    eig_vals = eig_vals.real

    pr = eig_vals.sum() ** 2 / (eig_vals ** 2).sum()

    return pr


def correlation(X):
    """
    Weird definition of correlation used in Cohen, Chung, Lee and Sampolinsky (2020)
    Divides by the vector norm instead of the standard deviation
    """
    X_c = X - X.mean(dim=1).unsqueeze(-1)
    stds = torch.norm(X_c, dim=1)
    
    # X_c = X
    # stds = torch.norm(X, dim=1)

    corr_mat = torch.einsum('ij,kj->ik', X_c, X_c)/torch.einsum('i,j->ij', stds, stds)

    return corr_mat


def compute_push_metrics(X):
    cosine_push = cosine_push_loss(X)
    uniformity_push = uniform_loss(X)
    kl_push = kl_divergence_to_standard_gaussian(X)
    swd_gaussian_push = sliced_wasserstein_distance_to_standard_gaussian(X)
    swd_hypersphere_push = sliced_wasserstein_distance_to_uniform_hypersphere(X)
    swd_hypercube_push = sliced_wasserstein_distance_to_uniform_hypercube(X)

    return cosine_push, uniformity_push, kl_push, swd_gaussian_push, swd_hypersphere_push, swd_hypercube_push
