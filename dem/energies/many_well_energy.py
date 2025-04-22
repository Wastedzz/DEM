from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from bgflow import MultiDoubleWellPotential
from hydra.utils import get_original_cwd
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.data_utils import remove_mean


class Energy():
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        
    @property
    def dim(self):
        return self._dim
    
    def _energy(self, x):
        raise NotImplementedError()
        
    def energy(self, x, temperature=None):
        #assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._energy(x) / temperature
    


class DoubleWellEnergy(Energy):
    def __init__(self, dim=2, a=-0.5, b=-6.0, c=1.0, k=1.0):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c
        self._k = k
    
    def _energy(self, x):
        d = x[..., 0:1]
        v = x[..., 1:]
        e1 = self._a * d + self._b * (d**2) + self._c * (d**4)
        e2 = torch.sum(0.5 * self._k * (v**2), dim=-1, keepdim=True)
        return e1 + e2
   
    @property
    def log_Z(self):
        if self._a == -0.5 and self._b == -6.0 and self._c == 1.0 and self._k == 1.0:
            log_Z_dim0 = np.log(11784.50927)
            log_Z_dim1 = 0.5 * np.log(2 * torch.pi)
            return log_Z_dim0 + log_Z_dim1
        else:
            raise NotImplementedError
        
    def log_prob(self, x):
        if self._a == -0.5 and self._b == -6.0 and self._c == 1.0 and self._k == 1.0:
            return -self.energy(x).squeeze(-1) - self.log_Z
        else:
            raise NotImplementedError


class ManyWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        data_from_efm=True,  # if False, data from EFM
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule=False,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        # self.data_path = get_original_cwd() + "/" + data_path
        # self.data_path_train = get_original_cwd() + "/" + data_path_train
        # self.data_path_val = get_original_cwd() + "/" + data_path_val

        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self.device = 'cpu'

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 100000

        self.double_well = DoubleWellEnergy(dim=2, a=-0.5, b=-6, c=1, k=1)


        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

        self.set_device = False
    
    def to(self, device):
        self.device = device
        if not self.set_device:
            self._test_set = self._test_set.to(device)
            self._val_set = self._val_set.to(device)
            self._train_set = self._train_set.to(device) if self._train_set is not None else None


    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return -self.multi_double_well.energy(samples).squeeze(-1)

    def setup_test_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size :]
            del all_data

        data = torch.tensor(data).to(
            self.device
        )

        return data

    def setup_train_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_train, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][: self.train_set_size]
            del all_data

        data = torch.tensor(data).to(
            self.device
        )

        return data

    def setup_val_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_val, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size - self.val_set_size : -self.test_set_size]
            del all_data

        data = torch.tensor(data).to(
            self.device
        )
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        samples = self.unnormalize(samples)
        samples_fig = self.get_dataset_fig(samples)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if unprioritized_buffer_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

        self.curr_epoch += 1

    def get_dataset_fig(self, samples):
        test_data_smaller = self.sample_test_set(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = -self(samples).detach().detach().cpu()
        energy_test = -self(test_data_smaller).detach().detach().cpu()

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()
        

        try:
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
            buffer.seek(0)

            return PIL.Image.open(buffer)

        except Exception as e:
            fig.canvas.draw()
            return PIL.Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.renderer.buffer_rgba()
            )
            fig.canvas.draw()
            return PIL.Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
