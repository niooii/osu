import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import osu.dataset as dataset


# make my life a lot easier
class OsuModel(ABC):
    def __init__(
        self, batch_size: int = 64, device: Optional[torch.device] = None, compile: bool = True, **kwargs
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.input_size = len(dataset.INPUT_FEATURES)
        self.compile = compile

        # set by load_data
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        # appended every time _train_epoch is called
        self.training_history = {}

        # Let subclasses initialize their specific models and optimizers
        self._initialize_models(**kwargs)
        self._initialize_optimizers()

        # Move models to device and optionally compile
        self._setup_device_and_compilation()

        # Initialize plotting figure
        plt.ion()  # Enable interactive mode
        self._loss_fig = None

        # Print initialization info
        self._print_initialization_info()

    # Should create all models (nn.Module instances) needed for training and inference
    @abstractmethod
    def _initialize_models(self, **kwargs):
        pass

    # Should create all optimizers. Models are already created before this
    @abstractmethod
    def _initialize_optimizers(self):
        pass

    # most models extract position data
    def _extract_target_data(self, output_data):
        return output_data[:, :, :2] # gets first 2 features (x, y)

    # Train model for one epoch, this will use custom imlpementations
    @abstractmethod
    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        pass

    # Generate output data for a map, with shape (T, features) where T is the time axis, and features
    # are the target (e.g. RNN should generate (x, y) as features, keypress (k1, k2), etc)
    @abstractmethod
    def generate(
        self, beatmap_data: Union[np.ndarray, torch.Tensor], **kwargs
    ) -> np.ndarray:
        pass

    # input_data and output_data are of shape (chunks, T, features), usually referred to around the codebase as 
    # xs and ys when loaded
    def load_data(
        self, input_data: np.ndarray, output_data: np.ndarray, test_size: float = 0.2
    ):

        # most classes will extract the cursor position data 
        target_data = self._extract_target_data(output_data)

        x_train, x_test, y_train, y_test = train_test_split(
            input_data, target_data, test_size=test_size, random_state=42
        )

        train_dataset = TensorDataset(
            torch.FloatTensor(x_train), torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(x_test), torch.FloatTensor(y_test)
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        print(
            f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples"
        )

    def _set_custom_train_status(self, status: str):
        self.train_iterator.set_postfix_str(status)

    # trains the model for a given amount of epochs
    # set save_every to 0 if you don't want to autosave
    def train(self, epochs: int, save_every: int = 5, save_dir: str = ".trained", **kwargs):
        if self.train_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self._set_train_mode()

        loss_str = None

        self.train_iterator = tqdm.tqdm(range(epochs), total=epochs, position=0)

        for epoch in self.train_iterator:
            self.train_iterator.set_description(
                f"[Epoch]: {epoch}" + (f" | {loss_str}" if loss_str is not None else "")
            )

            epoch_losses = self._train_epoch(epoch, epochs, **kwargs)

            for loss_name, loss_value in epoch_losses.items():
                if loss_name not in self.training_history:
                    self.training_history[loss_name] = []
                self.training_history[loss_name].append(loss_value)

            loss_str = ", ".join(
                [f"{name.title()}: {value:.4f}" for name, value in epoch_losses.items()]
            )

            if save_every != 0 and epoch % save_every == 0:
                # am i a bum bro
                short_loss_str = "_".join(
                    [
                        f"{name.title().lower().replace("_loss_", "").replace("_loss", "").replace("loss_", "").replace("loss", "").strip()}_{value:.4f}"
                        for name, value in epoch_losses.items()
                    ]
                )

                self.save(additional_str=short_loss_str, save_dir=save_dir, verbose=False)

            self.plot_losses()

    # plots all the tracked losses
    def plot_losses(self):
        if not self.training_history:
            print("No training history to plot. Train the model first.")
            return

        loss_names = list(self.training_history.keys())
        num_plots = len(loss_names)

        # Create figure on first use
        if self._loss_fig is None:
            figsize = (10, 6) if num_plots == 1 else (5 * min(3, num_plots), 4 * ((num_plots + 2) // 3))
            self._loss_fig = plt.figure(figsize=figsize)
        
        # Clear the figure and replot
        self._loss_fig.clear()

        if num_plots == 1:
            ax = self._loss_fig.add_subplot(111)
            loss_name = loss_names[0]
            ax.plot(
                self.training_history[loss_name],
                label=loss_name.replace("_", " ").title(),
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{self._get_model_name()} Training Loss")
            ax.legend()
        else:
            # Multiple subplots for multiple loss types
            cols = min(3, num_plots)
            rows = (num_plots + cols - 1) // cols

            for i, loss_name in enumerate(loss_names):
                ax = self._loss_fig.add_subplot(rows, cols, i + 1)
                ax.plot(
                    self.training_history[loss_name],
                    label=loss_name.replace("_", " ").title(),
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(loss_name.replace("_", " ").title())
                ax.legend()

        self._loss_fig.tight_layout()
        plt.draw()
        plt.pause(0.001)

    # if path prefix is none, it saves to .trained/.
    def save(
        self,
        save_dir: Optional[str] = None,
        additional_str: Optional[str] = None,
        verbose: bool = True,
    ) -> str:

        model_name_lower = self._get_model_name().lower()
        save_dir = save_dir.rstrip('/') if save_dir is not None else ".trained"

        timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        save_path = f"{save_dir}/{model_name_lower}_{timestamp}{("_" + additional_str) if additional_str is not None else ""}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        state_dict = self._get_state_dict()

        # save both timestamped and most_recent versions
        torch.save(state_dict, save_path)
        torch.save(state_dict, f"{save_dir}/{model_name_lower}_most_recent.pt")

        verbose and print(f"{self._get_model_name()} model saved to {save_dir}.pt")

        return save_path

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, **kwargs):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create instance
        instance = cls(device=device, **kwargs)

        # Load state dict
        checkpoint = torch.load(path, map_location=device)
        instance._load_state_dict(checkpoint)
        instance._set_eval_mode()

        print(f"{cls.__name__} loaded from {path}")
        return instance

    # Helper functions for freezing parameters/entire models


    @staticmethod
    def set_requires_grad(model: nn.Module, requires_grad: bool):
        for p in model.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def freeze_model(model: nn.Module):
        OsuModel.set_requires_grad(model, False)
        model.eval()

    @staticmethod
    def unfreeze_model(model: nn.Module):
        OsuModel.set_requires_grad(model, True)
        model.train()


    # Helper methods that subclasses can override if needed

    def _setup_device_and_compilation(self):
        # Move all nn.Module attributes to device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                attr.to(self.device)

                if self.compile and hasattr(torch, "compile"):
                    compiled_attr = torch.compile(attr)
                    setattr(self, attr_name, compiled_attr)

    def _print_initialization_info(self):
        model_name = self._get_model_name()
        print(f"{model_name} initialized on {self.device}")

        total_params = 0
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                params = sum(p.numel() for p in attr.parameters() if p.requires_grad)
                print(f"{attr_name} parameters: {params}")
                total_params += params

        if total_params > 0:
            print(f"Total parameters: {total_params}")

    def _get_model_name(self) -> str:
        class_name = self.__class__.__name__

        if class_name.startswith("Osu"):
            class_name = class_name[3:]
        if class_name.endswith("Model"):
            class_name = class_name[:-5]
        return class_name

    @abstractmethod
    def _get_state_dict(self) -> dict:
        pass

    @abstractmethod
    def _load_state_dict(self, checkpoint: dict):
        pass

    def _set_eval_mode(self):
        self.training = False
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                attr.eval()

    def _set_train_mode(self):
        self.training = True
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                attr.train()

