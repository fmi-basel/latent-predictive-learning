import torch
import torch.nn.functional as F
import numpy as np

MAX_DIM = 1024


def cosine_pull_loss(a, b, labels=None):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    if labels is None:
        loss = - (a * b).sum(-1).mean()
    else:
        loss = 0
        cossim_a = - torch.einsum('ij,kj->ik', a, a)
        cossim_b = - torch.einsum('ij,kj->ik', b, b)
        for i in torch.unique(labels):
            idxs = torch.nonzero(labels == i, as_tuple=False).squeeze()
            num_samples = len(idxs)
            class_sims_a = cossim_a[idxs][:, idxs]
            class_sims_b = cossim_b[idxs][:, idxs]
            loss += 0.5 * (class_sims_a[np.triu_indices(num_samples, k=1)].mean() +
                           class_sims_b[np.triu_indices(num_samples, k=1)].mean())

    return loss


def cosine_push_loss(a, labels=None):
    a = F.normalize(a, dim=-1)
    div = torch.einsum('ij,kj->ik', a, a) ** 2
    if labels is None:
        mask = torch.ones_like(div, dtype=torch.bool, device=div.device)
        mask.fill_diagonal_(0)
    else:
        exclusion_mask = torch.tril(torch.ones_like(div, dtype=torch.bool, device=div.device))
        for i in torch.unique(labels):
            idxs = torch.nonzero(labels == i, as_tuple=False).squeeze()
            for i in idxs:
                exclusion_mask[i, idxs] = True
        mask = ~exclusion_mask
    loss = (div * mask).sum() / mask.sum()

    return loss


def cosine_push_to_centroid_loss(a):
    a = F.normalize(a, dim=-1)
    centroid = a.detach().mean(dim=0, keepdim=True)
    centroid = F.normalize(centroid, dim=-1)
    div = torch.einsum('ij,kj->ik', a, centroid) ** 2
    loss = div.mean()

    return loss


def uniform_loss(x, t=2):
    """
    Loss function from:
    Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
    <https://arxiv.org/abs/2005.10242>
    Paper authors: Tongzhou Wang, Phillip Isola

    Function copied from project implementation at:
    https://github.com/SsnL/align_uniform
    """
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def kl_divergence_to_standard_gaussian(x, eps=1e-4):
    mu = x.mean(dim=0)
    logvar = torch.log(x.var(dim=0) + eps)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def sliced_kld_to_standard_gaussian(x, eps=1e-4):
    dim = min(x.shape[-1], MAX_DIM)
    rand_w = torch.empty(x.shape[-1], dim).to(x.device)
    torch.nn.init.orthogonal_(rand_w)
    x = torch.mm(x, rand_w)
    return kl_divergence_to_standard_gaussian(x, eps=eps)


def logvar(x, eps=1e-4):
    loss = torch.log(x.var(dim=0) + eps)
    return torch.abs(loss).mean()


def radial_push(x, eps=1e-4):
    centroid = x.detach().mean(dim=0, keepdim=True)
    loss = -1.0/(torch.norm(x - centroid, dim=1, p=1) + eps) + 1.0/(torch.norm(x - centroid, dim=1, p=2) + eps)
    return loss.mean()

"""
The following are loss functions from:
Intriguing Properties of Contrastive Losses
<https://arxiv.org/abs/2011.02803>
Paper authors: Ting Chen, Lala Li

Functions are reimplemented in pytorch from the tensorflow implementation at:
https://github.com/google-research/simclr/tree/master/colabs/intriguing_properties
"""


def get_swd_loss(x, rand_w, prior='normal', stddev=1., hidden_norm=False):
    states = torch.mm(x, rand_w)
    states_t = torch.sort(torch.transpose(states, 0, 1))[0]  # (dim, bsz)

    if prior == 'normal':
        states_prior = stddev * torch.randn_like(x)
    elif prior == 'uniform':
        states_prior = stddev * (2 * torch.rand_like(x) - 1)
    else:
        raise ValueError('Unknown prior {}'.format(prior))

    if hidden_norm:
        states_prior = F.normalize(states_prior, dim=-1)

    states_prior = torch.mm(states_prior, rand_w)
    states_prior_t = torch.sort(torch.transpose(states_prior, 0, 1))[0]  # (dim, bsz)

    return ((states_prior_t - states_t) ** 2).mean()


def sliced_wasserstein_distance_to_standard_gaussian(x):
    dim = min(x.shape[-1], MAX_DIM)
    rand_w = torch.empty(x.shape[-1], dim).to(x.device)
    torch.nn.init.orthogonal_(rand_w)
    loss = get_swd_loss(x, rand_w, prior='normal', stddev=1.0)
    return loss


def sliced_wasserstein_distance_to_uniform_hypersphere(x):
    dim = min(x.shape[-1], MAX_DIM)
    rand_w = torch.empty(x.shape[-1], dim).to(x.device)
    torch.nn.init.orthogonal_(rand_w)
    x = F.normalize(x, dim=-1)
    loss = get_swd_loss(x, rand_w, prior='normal', stddev=1.0, hidden_norm=True)
    return loss


def sliced_wasserstein_distance_to_uniform_hypercube(x):
    dim = min(x.shape[-1], MAX_DIM)
    rand_w = torch.empty(x.shape[-1], dim).to(x.device)
    torch.nn.init.orthogonal_(rand_w)
    loss = get_swd_loss(x, rand_w, prior='uniform', stddev=1.0)
    return loss


def sparse_filtering_loss(x):
    x = F.normalize(x, dim=0)
    x = F.normalize(x, dim=1)

    loss = torch.norm(x, dim=1, p=1).mean()

    return loss
