"""Hamiltonian class."""
import numpy as np
import math


class Hamiltonian:
    """Calculate variables regarding energy of given system."""

    def __init__(self, gamma, omega, numerical_step, system):
        """Instance of class."""
        self.gamma = gamma
        self.omega = omega
        self.omega2 = omega*omega
        self.step = numerical_step
        self.s = system

    def local_energy(self, positions):
        """Return the local energy."""

        Xi = 0.0
        fd, sd = self.derivative_wavefunction
        interaction = self.interaction
        for i in range(self.s.M):
            Xi += self.positions[i]
        local_energy = 0.5*(-fd*fd + sd + self.omega2*Xi) + interaction

        return local_energy

    def interaction(self, positions):
        """Return the interaction between particles"""

        return 0

    def local_energy_times_wf(self, positions):
        """Return local energy times the derivative of wave equation."""

        energy = self.local_energy(positions)
        energy_times_wf_a = self.s.derivative_wavefunction(positions)*energy
        energy_times_wf_b = self.s.derivative_wavefunction(positions)*energy
        energy_times_wf_W = self.s.derivative_wavefunction(positions)*energy

        return energy_times_wf_a, energy_times_wf_b, energy_times_wf_W
