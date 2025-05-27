from typing import Tuple

import torch
from scipy import signal
from torch import Tensor

from src.models.gan import GAN


class udGAN(GAN):
    def __init__(*args, **kwargs):
        """
        The udGAN class is a subclass of the GAN class. It generates data with twice the
        sampling rate and then downsamples it by a factor of 2 while applying a lowpass.
        The Idea is that we get rid of the high frequency noise and get a better.
        """
        super().__init__(*args, **kwargs)

    def training_step(self, batch_real: Tuple[Tensor, Tensor], batch_idx: int):
        """
        The training step for the GAN. This is called by the Lightning Trainer framework.
        Here we define a single training step for the GAN. This includes the forward pass,
        the calculation of the loss and the backward pass.

        Steps
        ----------
        1: Generate a batch of fake data
        2: In each epoch train critic
        3: In each n_critic epochs train generator

        """
        X_real, y_real = batch_real

        optim_g, optim_c, optim_spc = self.optimizers()

        # 1: Generate fake batch:
        X_fake, y_fake = self.generator.generate(X_real.shape[0])

        # Lowpass filter generated data and downsample it by factor 2
        X_fake = self.generator.lowpass_filter(X_fake)

        y_fake = y_fake.type_as(y_real)

        # Lowpass filter real data and downsample it by factor 2
        X_real = self.lowpass_filter_and_downsample(
            X_real,
            sfreq=self.sfreq / 2,
            low_pass_freq=self.low_pass_freq,
            downsample_factor=self.downsample_factor,
            filter_order=self.filter_order,
        )

        # 2: Train critic:
        ## optimize time domain critic
        c_loss, gp = self.train_critic(
            X_real, y_real, X_fake, y_fake, self.critic, optim_c
        )
        ## optional: optimize frequency domain critic
        if self.sp_critic:
            spc_loss, _ = self.train_critic(
                X_real, y_real, X_fake, y_fake, self.sp_critic, optim_spc
            )

        # 3: Train generator:
        # If n_critic =! 1 we train the generator only every n_th step
        if (batch_idx + 1) % self.n_epochs_critics == 0:
            self.toggle_optimizer(optim_g)

            # Generate fake data
            X_fake, y_fake = self.generator.generate(X_real.shape[0])

            ## optimize generator
            fx_fake = self.critic(X_fake, y_fake)
            if self.sp_critic:
                fx_spc = self.sp_critic(X_fake, y_fake)
                loss_fd = torch.mean(softplus(-fx_spc))
            else:
                loss_fd = 0
                self.beta = 0

            loss_td = torch.mean(softplus(-fx_fake))

            g_loss = (self.alpha * loss_td + self.beta * loss_fd) / (
                self.alpha + self.beta
            )

            self.manual_backward(g_loss)
            optim_g.step()
            optim_g.zero_grad()
            self.untoggle_optimizer(optim_g)
            self.generator_loss(g_loss)

        # Log
        self.critic_loss(c_loss.item())
        if self.sp_critic:
            self.sp_critic_loss(spc_loss.item())
        self.gp(gp.item())

        self.sliced_wasserstein_distance.update(X_real, X_fake)

        self.log_metrics()

    def lowpass_filter_and_downsample(
        self,
        data: torch.Tensor,
        sfreq: float,
        low_pass_freq: float,
        downsample_factor: int,
        filter_order: int = 5,
    ) -> Tuple[torch.Tensor, float]:
        """
        Applies a Butterworth low-pass filter and then downsamples a batch of EEG data.

        """
        # --- Input Validation ---
        if not isinstance(data, torch.Tensor) or data.ndim != 3:
            raise ValueError(
                "Input 'data' must be a 3D PyTorch Tensor (batch, channels, time_points)."
            )
        if not data.dtype.is_floating_point:
            print(
                f"Warning: Input data tensor has dtype {data.dtype}. Converting to float32 for filtering."
            )
            data = data.float()  # Ensure float for filtering stability

        if not isinstance(downsample_factor, int) or downsample_factor < 1:
            raise ValueError("'downsample_factor' must be an integer >= 1.")

        original_nyquist = sfreq / 2.0
        if low_pass_freq >= original_nyquist:
            raise ValueError(
                f"low_pass_freq ({low_pass_freq} Hz) must be strictly less than "
                f"the original Nyquist frequency ({original_nyquist} Hz)."
            )

        target_nyquist = (sfreq / downsample_factor) / 2.0
        if low_pass_freq >= target_nyquist:
            print(
                f"Warning: low_pass_freq ({low_pass_freq} Hz) is not strictly less than "
                f"the target Nyquist frequency after downsampling ({target_nyquist} Hz). "
                f"This may lead to aliasing artifacts during downsampling. "
                f"Consider setting low_pass_freq < {target_nyquist} Hz."
            )

        # --- Filtering ---
        # Design the Butterworth low-pass filter using second-order sections (SOS) for stability
        normalized_cutoff = low_pass_freq / original_nyquist
        sos = signal.butter(
            filter_order, normalized_cutoff, btype="low", analog=False, output="sos"
        )

        original_device = data.device
        original_dtype = data.dtype
        data_np = data.contiguous().cpu().numpy()

        filtered_data_np = signal.sosfiltfilt(sos, data_np, axis=-1)

        filtered_data = (
            torch.from_numpy(filtered_data_np.copy())
            .to(original_device)
            .to(original_dtype)
        )

        # --- Downsampling ---
        if downsample_factor > 1:
            # Simple slicing along the time dimension
            downsampled_data = filtered_data[:, :, ::downsample_factor]
        else:
            # No downsampling if factor is 1
            downsampled_data = filtered_data

        # --- Calculate new sampling frequency ---
        new_sfreq = sfreq / downsample_factor

        return downsampled_data, new_sfreq
