#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

def calculate_inception_score(preds: Tensor, splits: int = 1, repititions: int = 1) -> Tuple[float, float]:
    with torch.no_grad():
        stepsize = np.max((int(np.ceil(preds.size(0) / splits)), 2))
        steps = np.arange(0, preds.size(0), stepsize)
        scores = []
        for rep in np.arange(repititions):
            preds_tmp = preds[torch.randperm(preds.size(0), device=preds.device)]
            if len(preds_tmp) < 2:
                continue
            for i in np.arange(len(steps)):
                preds_step = preds_tmp[steps[i]:steps[i] + stepsize]
                step_mean = torch.nanmean(preds_step, 0, keepdim=True)
                kl = preds_step * (torch.log(preds_step) - torch.log(step_mean))
                kl = torch.nanmean(torch.sum(kl, 1))
                scores.append(torch.exp(kl).item())

        return np.nanmean(scores).item(), np.nanstd(scores).item()

def get_predictions(data, model, batch_size=100, device='cuda', n_label=4, num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(data):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(data)

    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)

    pred_arr = np.empty((len(data), n_label))

    start_idx = 0
    
    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[1]

        pred = pred.reshape(pred.shape[0], -1).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return Tensor(pred_arr)