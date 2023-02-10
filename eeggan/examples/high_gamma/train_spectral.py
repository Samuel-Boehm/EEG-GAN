#  Author: Samuel Böhm <samuel-boehm@web.de>

import os
import numpy as np
import torch
from typing import Tuple
from ignite.engine import Events
from ignite.metrics import MetricUsage
from matplotlib import pyplot
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eeggan.cuda import to_cuda, init_cuda
from eeggan.examples.high_gamma.braindecode_hack import IntermediateOutputWrapper
from eeggan.data.dataset import Data
from eeggan.data.preprocess.resample import downsample
from eeggan.examples.high_gamma.make_data import load_dataset, load_deeps4
from eeggan.training.handlers.metrics import WassersteinMetric, InceptionMetric, FrechetMetric, \
    LossMetric,ClassificationMetric, WassersteinMetricPOT, Old_WassersteinMetric
from eeggan.training.handlers.plots import SpectralPlot, DiscriminatorSpectrum
from eeggan.training.progressive.handler import SpectralProgessionHandler
from eeggan.training.trainer.gan_softplus_spectral import SpectralTrainer

def train_spectral(dataset_name: str, dataset_path: str, deep4s_path: str, result_path: str,
          progression_handler: SpectralProgessionHandler, trainer: SpectralTrainer, n_batch: int, lr_d: float, lr_g: float,
          betas: Tuple[float, float], n_epochs_per_stage: int, n_epochs_metrics: int, plot_every_epoch: int,
          orig_fs: float, n_samples: int, tensorboard_writer: SummaryWriter, subject: int = None, 
          pretrained_path: str = None):

    plot_path = os.path.join(result_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    init_cuda()  # activate cuda

    dataset = load_dataset(dataset_name, dataset_path)
    
    # Here we can select a subject:
    if subject:
        dataset.train_data = dataset.train_data.return_subject(subject)
        dataset.test_data = dataset.test_data.return_subject(subject)


    # Pool Train and Test
    dataset.train_data.X = np.concatenate((dataset.train_data.X[:], dataset.test_data.X[:]), axis=0)
    dataset.train_data.y = np.concatenate((dataset.train_data.y[:], dataset.test_data.y[:]), axis=0)
    dataset.train_data.y_onehot = np.concatenate((dataset.train_data.y_onehot[:], dataset.test_data.y_onehot[:]), axis=0)
    
    train_data = dataset.train_data

    del dataset

    discriminator = progression_handler.discriminator
    generator = progression_handler.generator
    spectral_discriminator = progression_handler.spectral_discriminator

    discriminator, generator, spectral_discriminator = to_cuda(discriminator, generator, spectral_discriminator)

        
    # usage to update every epoch and compute once at end of stage
    usage_metrics = MetricUsage(Events.STARTED, Events.EPOCH_COMPLETED(every=n_epochs_per_stage),
                                Events.EPOCH_COMPLETED(every=n_epochs_metrics))

    for stage in range(progression_handler.current_stage, progression_handler.n_stages):

        # If we set a path to pretrained models for each stage, the model will be loaded and used as base
        if pretrained_path:
            stateDict = torch.load(os.path.join(pretrained_path, f'states_stage_{stage}.pt'))
            progression_handler.generator.load_state_dict(stateDict['generator'])
            progression_handler.discriminator.load_state_dict(stateDict['discriminator'])

        # scale data for current stage
        sample_factor = 2 ** (progression_handler.n_stages - stage - 1)
        X_block = downsample(train_data.X, factor=sample_factor, axis=2)

        # Set mask for spectral discriminator stage: 
        trainer.spectral_discriminator.calculate_size_for_block()
        to_cuda(trainer.spectral_discriminator)

        # optimizer
        optim_discriminator = optim.Adam(progression_handler.get_trainable_discriminator_parameters()[0], lr=lr_d,
                                         betas=betas)

        optim_generator = optim.Adam(progression_handler.get_trainable_generator_parameters(), lr=lr_g, betas=betas)

        optim_spectral = optim.Adam(progression_handler.get_trainable_discriminator_parameters()[1] , lr=lr_d,
                                         betas=betas)

        trainer.set_optimizers(optim_discriminator, optim_generator, optim_spectral)

        # modules to save
        to_save = {'discriminator': discriminator, 'generator': generator,
                   'optim_discriminator': optim_discriminator, 'optim_generator': optim_generator}

        # load trained deep4s for stage
        deep4s = load_deeps4(dataset_name, stage, deep4s_path)
        select_modules = ['conv_4', 'softmax']
        deep4s = [to_cuda(IntermediateOutputWrapper(select_modules, deep4)) for deep4 in deep4s]

        # initiate spectral plotter
        spectral_plot = SpectralPlot(pyplot.figure(figsize=(15,7)), plot_path, "spectral_stage_%d_" % stage, X_block.shape[2],
                                     orig_fs / sample_factor, tb_writer = tensorboard_writer)
        spectral_profile_plot = DiscriminatorSpectrum(pyplot.figure(), plot_path, "spectral_profile_%d_" % stage,
                                                    tb_writer = tensorboard_writer)

        event_name = Events.EPOCH_COMPLETED(every=plot_every_epoch)
        spectral_handler = trainer.add_event_handler(event_name, spectral_plot)
        spectral_handler = trainer.add_event_handler(event_name, spectral_profile_plot)

    
        # initiate metrics
        metric_wasserstein = WassersteinMetric(100, np.prod(X_block.shape[1:]).item(), tb_writer=tensorboard_writer)
        metric_inception = InceptionMetric(deep4s, sample_factor, tb_writer=tensorboard_writer)
        # metric_frechet = FrechetMetric(deep4s, sample_factor, tb_writer=tensorboard_writer) -> Does not work becaus of GPU memory overload :( 
        metric_loss = LossMetric(tb_writer=tensorboard_writer)
        metric_classification = ClassificationMetric(deep4s, sample_factor, tb_writer=tensorboard_writer)
        metric_POT_SWD = WassersteinMetricPOT(100, tb_writer=tensorboard_writer)
        metric_old_swd = Old_WassersteinMetric(100, np.prod(X_block.shape[1:]).item(), tb_writer=tensorboard_writer)
        metrics = [metric_wasserstein, metric_POT_SWD, metric_old_swd, metric_inception, metric_loss, metric_classification]
        metric_names = ["wasserstein", "POT_SWD", 'old_swd', "inception", "loss", "classification"]
        trainer.attach_metrics(metrics, metric_names, usage_metrics)

        # wrap into cuda loader
        train_data_tensor: Data[Tensor] = Data(
            *to_cuda(Tensor(X_block), Tensor(train_data.y), Tensor(train_data.y_onehot)))
        train_loader = DataLoader(train_data_tensor, batch_size=n_batch, shuffle=True)
        
        # train stage
        state = trainer.run(train_loader, (stage + 1) * n_epochs_per_stage)
        trainer.remove_event_handler(spectral_plot, event_name)
        trainer.remove_event_handler(spectral_profile_plot, event_name) 

        # save stuff
        torch.save(to_save, os.path.join(result_path, 'modules_stage_%d.pt' % stage))
        torch.save(dict([(name, to_save[name].state_dict()) for name in to_save.keys()]),
                   os.path.join(result_path, 'states_stage_%d.pt' % stage))
        torch.save(trainer.state.metrics, os.path.join(result_path, 'metrics_stage_%d.pt' % stage))


        # advance stage if not last 
        if stage != progression_handler.n_stages - 1:
            progression_handler.advance_stage()

        trainer.detach_metrics(metrics, usage_metrics)
        