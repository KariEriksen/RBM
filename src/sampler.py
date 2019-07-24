"""Sampler class."""
import numpy as np


class Sampler:
    """Calculate variables regarding energy of given system."""

    def __init__(self, gamma, numerical_step, system):
        """Instance of class."""
        self.gamma = gamma
        self.step = numerical_step
        self.s = system

    def local_energy(self, positions):
        """Return the local energy."""

        return 0

    def local_energy_times_wf(self, positions):
        """Return local energy times the derivative of wave equation."""

        return 0

    def probability(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""

        return 0

    def drift_force(self, positions):
        """Return drift force."""
        # position_forward = positions + self.step
        position_forward = np.array(positions) + self.step
        wf_forward = self.s.wavefunction(position_forward)
        wf_current = self.s.wavefunction(positions)
        derivativ = (wf_forward - wf_current)/self.step

        return derivativ

    def greens_function(self, positions, new_positions_importance, delta_t):
        """Calculate Greens function."""
        # greens_function = 0.0
        """
        D = 0.0
        F_old = self.drift_force(positions)
        F_new = self.drift_force(new_positions_importance)
        greens_function = (0.5*(F_old + F_new) * (0.5 * (positions -
                           new_positions_importance)) +
                           D*delta_t*(F_old - F_new))
        """
        greens_function = 0.0
        # greens_function = np.exp(greens_function)

        return greens_function
