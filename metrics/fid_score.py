"""
Code for FID metric calculation. 
Based on https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
"""

import os
import glob
import pickle
import math
from os.path import join

import numpy as np
import torch
import cv2
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from models.inception import InceptionV3


def get_activations(image_iter, model, dims=2048):
    model.eval()
    pred_arr = np.empty((image_iter.total_len(), dims))
    pos = 0
    for batch in tqdm(iter(image_iter)):
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        bs = pred.shape[0]
        pred_arr[pos:pos+bs] = pred.flatten(1).cpu().numpy()
        pos += bs
    return pred_arr

def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def load_precomputed_statistics(path_to_precomputed):
    with open(path_to_precomputed, 'w') as f: 
        sm = pickle.load(f)
    return sm

def imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) / 255.

class DatasetIterator:
    def __init__(self, path_to_images, batch_size=64, device='cpu'):
        self.path_to_images = path_to_images
        self.files = os.listdir(path_to_images)
        self.n_images = len(self.files)
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= self.n_images:
            raise StopIteration

        images = np.array([imread(join(self.path_to_images, f))
            for f in self.files[self.counter:self.counter+self.batch_size]]).astype(np.float32)
        images = images.transpose((0, 3, 1, 2))
        batch = torch.from_numpy(images).to(self.device)

        self.counter += self.batch_size
        return batch

    def __len__(self):
        return math.ceil(self.n_images / self.batch_size)

    def total_len(self):
        return self.n_images


def calculate_fid(gen_iter, path_to_precomputed, device='cpu', batch_size=64, dims=2048):
    """
    Calculates the FID score

    Parameters
    ----------
    gen_iter: iterator which yields batch of generated images on every iteration. 
              Must have function 'total_len' which returns total number of images
    path_to_precomputed: path to pickle file with tuple (mean, covariation_matrix) for chosen dataset

    Return
    ------
    fid_value: value of FID score
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(get_activations(gen_iter, model, dims))
    m2, s2 = load_precomputed_statistics(path_to_precomputed)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def precompute_dataset_statistics(path_to_images, path_to_save, device='cpu', batch_size=64, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    diter = DatasetIterator(path_to_images, batch_size, device=device)
    ms = calculate_activation_statistics(get_activations(diter, model, dims))
    with open(path_to_save) as f:
        pickle.dump(ms, f)