"""Optimizer class."""


class Optimizer:
    """Optimization method."""

    """The optimizer method runs through a whole Monte Carlo loop
    for each gradient descent iteration. Update of the variational
    parameter is done within the run_vmc file."""

    def __init__(self, learning_rate):
        """Instance of class."""
        self.learning_rate = learning_rate

    def gradient_descent(self, a, b, W, d_energy_a, d_energy_b,
                         d_energy_W):
        """Orinary gradient descent."""

        new_a = a - self.learning_rate*d_energy_a
        new_b = b - self.learning_rate*d_energy_b
        new_W = W - self.learning_rate*d_energy_W

        return new_a, new_b, new_W

    def gradient_descent_Barzilai_Borwein(self, a, b, W, d_energy_a,
                                          d_energy_b, d_energy_W):
        """Gradient descent with Barzilai Borwein formula"""
        """for updating learning rate"""

        gamma_a = self.learning_rate
        gamma_b = self.learning_rate
        gamma_W = self.learning_rate
        new_a = a - gamma_a*d_energy_a
        new_b = b - gamma_b*d_energy_b
        new_W = W - gamma_W*d_energy_W

        return new_a, new_b, new_W
