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
from osu.rulesets.core import REPLAY_SAMPLING_RATE


# make my life a lot easier
class OsuModel(ABC):
    def __init__(
        self,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        compile: bool = True,
        **kwargs,
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
        return output_data[:, :, :2]  # gets first 2 features (x, y)

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
    def train(
        self, epochs: int, save_every: int = 5, save_dir: str = ".trained", **kwargs
    ):
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

                self.save(
                    additional_str=short_loss_str, save_dir=save_dir, verbose=False
                )

    # plots all the tracked losses
    def plot_losses(self):
        if not self.training_history:
            print("No training history to plot. Train the model first.")
            return

        loss_names = list(self.training_history.keys())
        num_plots = len(loss_names)

        if num_plots == 1:
            plt.figure(figsize=(10, 6))
            loss_name = loss_names[0]
            plt.plot(
                self.training_history[loss_name],
                label=loss_name.replace("_", " ").title(),
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{self._get_model_name()} Training Loss")
            plt.legend()
        else:
            # Multiple subplots for multiple loss types
            cols = min(3, num_plots)
            rows = (num_plots + cols - 1) // cols

            plt.figure(figsize=(5 * cols, 4 * rows))

            for i, loss_name in enumerate(loss_names):
                plt.subplot(rows, cols, i + 1)
                plt.plot(
                    self.training_history[loss_name],
                    label=loss_name.replace("_", " ").title(),
                )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(loss_name.replace("_", " ").title())
                plt.legend()

        plt.tight_layout()
        plt.show()

    # displays metrics on a model's performance without requiring you to watch a replay
    # accepts tensors/arrays of size (num_chunks, chunk_len, features)
    def display_metrics(
        self,
        beatmap_data: Union[np.ndarray, torch.Tensor],
        real_data: Union[np.ndarray, torch.Tensor],
        max_frames: int=1000,
        plot_mode: str="3d",
        samples: int=1
    ):
        samples = min(samples, 15)
        fake_data_list = []
        for _ in range(samples):
            fake_data_list.append(self.generate(beatmap_data=beatmap_data))

        # flatten along the chunks dim
        # (frames, features)
        map_data = beatmap_data.reshape(-1, beatmap_data.shape[2])[:max_frames]
        rplay_data = real_data.reshape(-1, real_data.shape[2])[:max_frames]
        
        fplay_data_list = []
        for fake_data in fake_data_list:
            fplay_data_list.append(fake_data.reshape(-1, fake_data.shape[2])[:max_frames])

        # plot real data against fake data
        rplay_pos = rplay_data[:, :2]
        rplay_keys = rplay_data[:, 2:]

        # time is in seconds
        time = np.linspace(0, rplay_data.shape[0] * REPLAY_SAMPLING_RATE / 1000, rplay_data.shape[0])
        
        # color palette for samples
        colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 
                 'yellow', 'lime', 'indigo', 'teal', 'maroon']
        
        # position plot and key press plot
        if plot_mode == "2d":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(time, rplay_pos[:, 0] + rplay_pos[:, 1], "green", label="Real")
            for i, fplay_data in enumerate(fplay_data_list):
                fplay_pos = fplay_data[:, :2]
                color = colors[i % len(colors)]
                label = "Generated" if i == 0 else None
                ax1.plot(time, fplay_pos[:, 0] + fplay_pos[:, 1], color, label=label, linewidth=0.8, alpha=0.7)
            ax1.set_ylabel("X + Y Position")
            ax1.legend()
            ax1.set_title("Cursor Position")
        else:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(2, 1, 1, projection='3d')
            ax1.plot3D(time, rplay_pos[:, 0], rplay_pos[:, 1], "green", label="Real")
            for i, fplay_data in enumerate(fplay_data_list):
                fplay_pos = fplay_data[:, :2]
                color = colors[i % len(colors)]
                label = "Generated" if i == 0 else None
                ax1.plot3D(time, fplay_pos[:, 0], fplay_pos[:, 1], color, label=label, alpha=0.7)
            ax1.legend()
            ax1.set_title("Cursor Position (3D)")
            ax2 = fig.add_subplot(2, 1, 2)
        
        # plot keypresses
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_ylabel("Keys")
        ax2.set_xlabel("Time (s)")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["K1", "K2"])
        ax2.set_title("Key Presses")
        
        for key_idx in range(min(2, rplay_keys.shape[1])):  # k1 and k2
            key_pressed = (rplay_keys[:, key_idx] > 0.5).astype(int)
            key_diff = np.diff(np.concatenate([[0], key_pressed, [0]]))
            starts = np.where(key_diff == 1)[0]
            ends = np.where(key_diff == -1)[0]
            
            for start, end in zip(starts, ends):
                ax2.axvspan(time[start], time[end-1], ymin=key_idx/2, ymax=(key_idx+1)/2, 
                           color='green', alpha=0.5)
            
            for i, fplay_data in enumerate(fplay_data_list):
                fplay_keys = fplay_data[:, 2:]
                color = colors[i % len(colors)]
                key_pressed = (fplay_keys[:, key_idx] > 0.5).astype(int)
                key_diff = np.diff(np.concatenate([[0], key_pressed, [0]]))
                starts = np.where(key_diff == 1)[0]
                ends = np.where(key_diff == -1)[0]
                
                for start, end in zip(starts, ends):
                    ax2.axvspan(time[start], time[end-1], ymin=key_idx/2, ymax=(key_idx+1)/2, 
                               color=color, alpha=0.3)
        
        plt.tight_layout()

        if samples > 1:
            def mae(p1, p2):
                return np.mean(np.abs(p1 - p2))

            def mse(p1, p2):
                return np.mean((p1 - p2) ** 2) 

            avg_mae = np.average([mae(fake_data_list[i], fake_data_list[i+1]) for i in range(samples-1)])
            avg_mse = np.average([mse(fake_data_list[i], fake_data_list[i+1]) for i in range(samples-1)])

            print(f"MAE: {avg_mae}")
            print(f"MSE: {avg_mse}")
        else:
            print(f"MAE and MSE not available for sample size of {samples}")


    # randomly selects a chunk in the loaded data, then generates play data for it,
    # and plots it against real play data.
    def plot_error_loaded_data(self, max_frames: int = 1000, random_seed: int = 42):

        pass

    # if path prefix is none, it saves to .trained/.
    def save(
        self,
        save_dir: Optional[str] = None,
        additional_str: Optional[str] = None,
        verbose: bool = True,
    ) -> str:

        model_name_lower = self._get_model_name().lower()
        save_dir = save_dir.rstrip("/") if save_dir is not None else ".trained"

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
