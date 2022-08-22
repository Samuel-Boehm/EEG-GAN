#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from torch.optim import AdamW
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
import torch


def build_model(input_time_length, n_channels, n_classes, cropped=False):

    model = Deep4Net(in_chans=n_channels,
                     n_classes=n_classes,
                     input_window_samples=input_time_length,
                     final_conv_length='auto')

    if cropped:
        final_conv_length = model.final_conv_length
        model = Deep4Net(n_channels, n_classes, input_window_samples=input_time_length,
                         final_conv_length=final_conv_length)

    if cropped:
        to_dense_prediction_model(model)

    optimizer = AdamW(model.parameters(), lr=1 * 0.01, weight_decay=0.5 * 0.001)

    loss = NLLLoss()

    return model, loss, optimizer


def train_model(model, train_dataloader, loss, optimizer, n_epochs=10, cuda=True):
    """Run training loop.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_dataloader : torch.utils.data.Dataloader
        Data loader which will serve examples to the model during training.
    loss :
        Loss function.
    optimizer :
        Optimizer.
    n_epochs : int
        Number of epochs to train the model for.
    cuda : bool
        If True, move X and y to CUDA device.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    """

    correct = 0
    total = 0

    for i in range(n_epochs):
        loss_vals = list()
        for X, y, _ in train_dataloader:
            model.train()
            model.zero_grad()

            # y = y.long()
            if cuda:
                X, y = X.cuda(), y.cuda()
                model = model.cuda()
            else:
                X, y = X.cpu(), y.cpu()
                model = model.cpu()

            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted.cpu() == y.cpu()).sum()

            loss_val = loss(model(X), y)
            loss_vals.append(loss_val.item())

            loss_val.backward()
            optimizer.step()

        accu = 100. * correct / total

        if (i + 1) % 10 == 0 or i == 0:
            print(f'Epoch {i + 1} - mean training loss: {np.mean(loss_vals)} - Acc %: {accu:.2f}')

    return model


def train_completetrials(train_set, test_set, n_classes, n_chans, n_epochs=100, batch_size=60, cuda=True):

    input_time_length = train_set.X.shape[2]
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    model, loss, optimizer = build_model(input_time_length, n_chans, n_classes, cropped=False)

    trained_model = train_model(model=model, train_dataloader=train_dataloader,
                          loss=loss, optimizer=optimizer, n_epochs=n_epochs, cuda=cuda)

    # Calculate Acc on Test set:
    X = torch.tensor(test_set.X).cuda()
    y = torch.tensor(test_set.y).cuda()

    out = trained_model(X)

    _, predicted = torch.max(out.data, 1)
    total = y.size(0)

    correct = (predicted.cpu() == y.cpu()).sum()
    accu = 100. * correct / total
    print(f'Accuracy on testset %: {accu:.3f}')

    return trained_model
