"""System class."""
import math
import numpy as np


class System:
    """Contains parameters of system and wave equation."""

    def __init__(self, num_particles, num_dimensions, a, b, W):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.alpha = a
        self.beta = b
        self.W = W

    def wavefunction(self, positions):
        """Return the NQS wave function ."""

        return 0

    def interaction(self, positions):
        """Calculate correlation factor."""

        return 0

    def derivative_psi_term(self, positions):
        """Calculate derivative of wave function divided by wave function."""

        return 0
