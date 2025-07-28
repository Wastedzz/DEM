from typing import Optional
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch
import dem.energies.gmm_lrds as gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image


class GMM(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        data_normalization_factor = n_mixes / 0.8 if data_normalization_factor is None else data_normalization_factor
        use_gpu = device != "cpu"
        torch.manual_seed(0)  # seed of 0 for GMM problem
        self.gmm = gmm.GMM(
            dim=dimensionality,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )
        self._dimensionality = dimensionality
        self.loc_scaling = loc_scaling
        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.data_path_train = data_path_train

        self.name = "gmm"

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )
        self.set_device = False
    
    def to(self, device):
        self.gmm.to(device)
        self.gmm.device = device
        self.device = device
        if not self.set_device:
            self._test_set = self._test_set.to(device)
            self._val_set = self._val_set.to(device)
            self._train_set = self._train_set.to(device) if self._train_set is not None else None

    def setup_test_set(self):
        test_sample = self.gmm.sample((self.test_set_size,))
        return test_sample
        # return self.gmm.test_set

    def setup_train_set(self):
        if self.data_path_train is None:
            train_samples = self.normalize(self.gmm.sample((self.train_set_size,)))

        else:
            # Assume the samples we are loading from disk are already normalized.
            # This breaks if they are not.

            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu().numpy()
            else:
                data = np.load(self.data_path_train, allow_pickle=True)

            data = torch.tensor(data, device=self.device)

        return train_samples

    def setup_val_set(self):
        val_samples = self.gmm.sample((self.val_set_size,))
        return val_samples

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        return self._dimensionality

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        logger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
        epoch: Optional[int] = 0,
    ) -> None:

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize:
                # Don't unnormalize CFM samples since they're in the
                # unnormalized space
                if latest_samples is not None:
                    latest_samples = self.unnormalize(latest_samples)

                if unprioritized_buffer_samples is not None:
                    unprioritized_buffer_samples = self.unnormalize(unprioritized_buffer_samples)

            if unprioritized_buffer_samples is not None:
                buffer_samples, _, _ = replay_buffer.sample(self.plotting_buffer_sample_size)
                if self.should_unnormalize:
                    buffer_samples = self.unnormalize(buffer_samples)

                samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)
                samples_tensor = ToTensor()(samples_fig)
                try:
                    logger.log_image(f"{prefix}unprioritized_buffer_samples", [samples_fig])
                except:
                    logger.experiment.add_image(f"{prefix}unprioritized_buffer_samples", samples_tensor, global_step=epoch)

            # if cfm_samples is not None:
            #     cfm_samples_fig = self.get_dataset_fig(unprioritized_buffer_samples, cfm_samples)
            #     cfm_samples_tensor = ToTensor()(cfm_samples_fig)
            #     try:
            #         logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_tensor])
            #     except:
            #         logger.experiment.add_image(f"{prefix}cfm_generated_samples", cfm_samples_tensor, global_step=epoch)

            if latest_samples is not None:
                img = self.get_single_dataset_fig(latest_samples, "dem_generated_samples")
                
                img = ToTensor()(img)
                try:
                    logger.log_image(f"{prefix}generated_samples", [img])
                except:
                    logger.experiment.add_image(f"{prefix}generated_samples", img, global_step=epoch)
            plt.close()

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])


    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        plotting_bounds = (-1.4 * self.loc_scaling, 1.4 * self.loc_scaling)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        if self.dimensionality == 2:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=ax,
                n_contour_levels=50,
                grid_width_n_points=200,
            )

        plot_marginal_pair(samples[:2], ax=ax, bounds=plotting_bounds)
        # ax.set_title(f"{name}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        self.gmm.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        plotting_bounds = (-1.4 * self.loc_scaling, 1.4 * self.loc_scaling)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        if self.dimensionality == 2:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[0],
                n_contour_levels=50,
                grid_width_n_points=200,
            )

        # plot dataset samples
        plot_marginal_pair(samples[:,:2], ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples is not None:
            if self.dimensionality == 2:
                plot_contours(
                    self.gmm.log_prob,
                    bounds=plotting_bounds,
                    ax=axs[1],
                    n_contour_levels=50,
                    grid_width_n_points=200,
                )
            # plot generated samples
            plot_marginal_pair(gen_samples[:,:2], ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)
