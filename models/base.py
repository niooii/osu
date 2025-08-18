import os
import datetime
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import osu.dataset as dataset


class OsuModel(ABC):
    """
    Abstract base class for all the osu! play generation models.
    """
    
    def __init__(self, batch_size: int = 64, device: Optional[torch.device] = None, **kwargs):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.input_size = len(dataset.INPUT_FEATURES)
        
        # set by load_data
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        # Training history - base structure that subclasses can extend
        self.training_history = {}
        
        # Let subclasses initialize their specific models and optimizers
        self._initialize_models(**kwargs)
        self._initialize_optimizers()
        
        # Move models to device and optionally compile
        self._setup_device_and_compilation()
        
        # Print initialization info
        self._print_initialization_info()
    
    @abstractmethod
    def _initialize_models(self, **kwargs):
        """
        Initialize the specific neural network models for this model type.
        
        This method should create all PyTorch nn.Module instances needed
        by the specific model implementation.
        
        Args:
            **kwargs: Model-specific initialization arguments
        """
        pass
    
    @abstractmethod
    def _initialize_optimizers(self):
        """
        Initialize optimizers for the model's neural networks.
        
        This method should create all optimizers needed for training
        the model's neural networks.
        """
        pass
    
    @abstractmethod
    def _extract_target_data(self, output_data: np.ndarray) -> np.ndarray:
        """
        Extract the relevant target data from the full output data.
        
        Different models need different parts of the output data:
        - Position models (VAE, GAN, RNN): Extract x, y coordinates (first 2 features)
        - Key models: Extract key states (last 2 features)
        
        Args:
            output_data: Full output data array with shape [batch, sequence, features]
                        where features typically include [x, y, k1, k2]
        
        Returns:
            Extracted target data for this specific model type
        """
        pass
    
    @abstractmethod
    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        """
        Train the model for one epoch.
        
        This method contains the core training logic specific to each model type.
        It should iterate through the training data and update model parameters.
        
        Args:
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs being trained
            
        Returns:
            Dictionary of loss values for this epoch (e.g., {'total_loss': 0.5, 'recon_loss': 0.3})
        """
        pass
    
    @abstractmethod
    def generate(self, beatmap_data: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Generate output data given beatmap input data.
        
        This method should set the model to evaluation mode, process the input data,
        and generate predictions using the trained model.
        
        Args:
            beatmap_data: Input beatmap features with shape [batch, sequence, features]
            **kwargs: Model-specific generation parameters
            
        Returns:
            Generated output data (positions, keys, etc.) as numpy array
        """
        pass
    
    def load_data(self, input_data: np.ndarray, output_data: np.ndarray, test_size: float = 0.2):
        """
        Load and split data for training and testing.
        
        This method handles the common pattern of:
        1. Extracting relevant target data using _extract_target_data
        2. Performing train/test split
        3. Creating PyTorch DataLoaders
        4. Storing the loaders for use during training
        
        Args:
            input_data: Input beatmap features with shape [samples, sequence, input_features]
            output_data: Output data with shape [samples, sequence, output_features]
            test_size: Fraction of data to use for testing (default: 0.2)
        """
        # Extract relevant target data using subclass-specific logic
        target_data = self._extract_target_data(output_data)
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, target_data, test_size=test_size, random_state=42
        )
        
        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    def train(self, epochs: int, **kwargs):
        """
        Train the model for the specified number of epochs.
        
        This method provides the common training loop structure:
        1. Check that data has been loaded
        2. Loop through epochs
        3. Call subclass-specific _train_epoch method
        4. Track and display training progress
        5. Update training history
        
        Args:
            epochs: Number of epochs to train
            **kwargs: Additional training parameters passed to _train_epoch
        """
        if self.train_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self._set_train_mode()
        
        for epoch in range(epochs):
            # Train one epoch using subclass-specific logic
            epoch_losses = self._train_epoch(epoch, epochs, **kwargs)
            
            # Update training history
            for loss_name, loss_value in epoch_losses.items():
                if loss_name not in self.training_history:
                    self.training_history[loss_name] = []
                self.training_history[loss_name].append(loss_value)
            
            # Print epoch results
            loss_str = ", ".join([f"{name.title()}: {value:.4f}" for name, value in epoch_losses.items()])
            print(f'Epoch {epoch + 1}/{epochs}, {loss_str}')
    
    def plot_losses(self):
        """
        Plot training loss history.
        
        Creates matplotlib plots for all tracked losses in the training history.
        The layout adapts based on the number of different loss types.
        """
        if not self.training_history:
            print("No training history to plot. Train the model first.")
            return
        
        loss_names = list(self.training_history.keys())
        num_plots = len(loss_names)
        
        if num_plots == 1:
            plt.figure(figsize=(10, 6))
            loss_name = loss_names[0]
            plt.plot(self.training_history[loss_name], label=loss_name.replace('_', ' ').title())
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{self._get_model_name()} Training Loss')
            plt.legend()
        else:
            # Multiple subplots for multiple loss types
            cols = min(3, num_plots)
            rows = (num_plots + cols - 1) // cols
            
            plt.figure(figsize=(5 * cols, 4 * rows))
            
            for i, loss_name in enumerate(loss_names):
                plt.subplot(rows, cols, i + 1)
                plt.plot(self.training_history[loss_name], label=loss_name.replace('_', ' ').title())
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(loss_name.replace('_', ' ').title())
                plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path_prefix: Optional[str] = None):
        """
        Save the trained model to disk.
        
        Args:
            path_prefix: Path prefix for saving. If None, generates timestamp-based name
        """
        model_name_lower = self._get_model_name().lower()
        if path_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path_prefix = f'.trained/{model_name_lower}_{timestamp}'
        
        if not os.path.exists('.trained'):
            os.makedirs('.trained')
        
        # Get state dict from subclass
        state_dict = self._get_state_dict()
        
        # Save both timestamped and most_recent versions
        torch.save(state_dict, f'{path_prefix}.pt')
        torch.save(state_dict, f'.trained/{model_name_lower}_most_recent.pt')
        
        print(f"{self._get_model_name()} model saved to {path_prefix}.pt")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, **kwargs):
        """
        Load a trained model from disk.
        
        This method provides the common loading pattern. Subclasses may need
        to override this method if they have special loading requirements.
        
        Args:
            path: Path to the saved model file
            device: Device to load the model on
            **kwargs: Additional arguments for model construction
            
        Returns:
            Loaded model instance
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create instance
        instance = cls(device=device, **kwargs)
        
        # Load state dict
        checkpoint = torch.load(path, map_location=device)
        instance._load_state_dict(checkpoint)
        instance._set_eval_mode()
        
        print(f"{cls.__name__} loaded from {path}")
        return instance
    
    # Helper methods that subclasses can override if needed
    
    def _setup_device_and_compilation(self):
        # Move all nn.Module attributes to device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                attr.to(self.device)
                
                if hasattr(torch, 'compile'):
                    compiled_attr = torch.compile(attr)
                    setattr(self, attr_name, compiled_attr)
    
    def _print_initialization_info(self):
        """Print model initialization information."""
        model_name = self._get_model_name()
        print(f"{model_name} initialized on {self.device}")
        
        # Count parameters for all nn.Module attributes
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
        """Get a human-readable name for this model type."""
        class_name = self.__class__.__name__

        # Remove "Osu" prefix and "Model" suffix
        if class_name.startswith('Osu'):
            class_name = class_name[3:]
        if class_name.endswith('Model'):
            class_name = class_name[:-5]
        return class_name

    @abstractmethod
    def _get_state_dict(self) -> dict:
        """
        Get the state dictionary for saving the model.
        
        Returns:
            Dictionary containing all necessary state for model reconstruction
        """
        pass

    @abstractmethod
    def _load_state_dict(self, checkpoint: dict):
        """
        Load the state dictionary when loading a model.
        
        Args:
            checkpoint: Dictionary containing saved model state
        """
        pass
    
    def _set_eval_mode(self):
        """Set all models to evaluation mode."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                attr.eval()
    
    def _set_train_mode(self):
        """Set all models to training mode."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                attr.train()