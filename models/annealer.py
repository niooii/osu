import math


class Annealer:
    """
    This class is used to anneal values over the course of training (e.g., KL divergence loss in VAEs).
    The annealing follows a specified shape (linear, cosine, or logistic) from a minimum to maximum value.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape='linear', range=(0.0, 1.0), cyclical=False, stay_max_steps=0, start_offset=0, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full annealing weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            range (tuple): Tuple of (min_val, max_val) defining the annealing range. Default is (0.0, 1.0).
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            stay_max_steps (int): If cyclical, number of steps to stay at maximum value before cycling. Default is 0.
            start_offset (int): Number of steps to delay annealing start. Returns start value during offset. Default is 0.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """

        self.current_step = 0

        if shape not in ['linear', 'cosine', 'logistic']:
            raise ValueError("Shape must be one of 'linear', 'cosine', or 'logistic.")
        self.shape = shape

        if not isinstance(range, (tuple, list)) or len(range) != 2:
            raise ValueError("Range must be a tuple or list of two numbers (start_val, end_val).")
        start_val, end_val = range
        if not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)):
            raise ValueError("Range values must be numbers.")
        if start_val == end_val:
            raise ValueError("Range start_val must be different from end_val.")
        self.range = (float(start_val), float(end_val))

        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("Argument total_steps must be an integer greater than 0")
        self.total_steps = total_steps

        if type(cyclical) is not bool:
            raise ValueError("Argument cyclical must be a boolean.")
        self.cyclical = cyclical

        if type(stay_max_steps) is not int or stay_max_steps < 0:
            raise ValueError("Argument stay_max_steps must be a non-negative integer")
        self.stay_max_steps = stay_max_steps

        if type(start_offset) is not int or start_offset < 0:
            raise ValueError("Argument start_offset must be a non-negative integer")
        self.start_offset = start_offset

        if type(disable) is not bool:
            raise ValueError("Argument disable must be a boolean.")
        self.disable = disable

    def __call__(self, value):
        """
        Args:
            value (torch.tensor or float): Value to be annealed
        Returns:
            out (torch.tensor or float): Input value multiplied by the current annealing weight.
        """
        if self.disable:
            return value
        out = value * self._slope()
        return out

    def step(self):
        total_cycle_steps = self.total_steps + self.stay_max_steps
        effective_step = self.current_step - self.start_offset
        if effective_step < total_cycle_steps:
            self.current_step += 1
        if self.cyclical and effective_step >= total_cycle_steps:
            self.current_step = self.start_offset
        return

    def set_cyclical(self, value):
        if not isinstance(value, bool):
            raise ValueError("Argument to cyclical method must be a boolean.")
        self.cyclical = value
        return

    def set_range(self, range):
        """
        Set the annealing range.
        
        Parameters:
            range (tuple): Tuple of (min_val, max_val) defining the annealing range.
        """
        if not isinstance(range, (tuple, list)) or len(range) != 2:
            raise ValueError("Range must be a tuple or list of two numbers (start_val, end_val).")
        start_val, end_val = range
        if not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)):
            raise ValueError("Range values must be numbers.")
        if start_val == end_val:
            raise ValueError("Range start_val must be different from end_val.")
        self.range = (float(start_val), float(end_val))
        return

    def current(self):
        """
        Get the current annealing multiplier/weight for the current step.
        
        Returns:
            float: The current annealing weight in the specified range.
        """
        return self._slope()


    def _slope(self):
        # If we're in the offset period, return start value
        if self.current_step < self.start_offset:
            return self.range[0]

        effective_step = self.current_step - self.start_offset

        # If we're in the stay-at-max period, return max value
        if effective_step >= self.total_steps:
            y = 1.0
        else:
            # Normal annealing calculation
            if self.shape == 'linear':
                y = (effective_step / self.total_steps)
            elif self.shape == 'cosine':
                y = (math.cos(math.pi * (effective_step / self.total_steps - 1)) + 1) / 2
            elif self.shape == 'logistic':
                exponent = ((self.total_steps / 2) - effective_step)
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1.0

        y = self._scale_to_range(y)
        return y

    def _scale_to_range(self, y):
        """Scale normalized value (0-1) to the specified range"""
        start_val, end_val = self.range
        y_out = y * (end_val - start_val) + start_val
        return y_out