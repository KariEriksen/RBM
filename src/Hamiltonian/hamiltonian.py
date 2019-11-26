"""Hamiltonian class."""


class Hamiltonian:
    """Calculate variables regarding energy of given system."""

    def __init__(self, gamma, omega, wavefunction):
        """Instance of class."""
        self.gamma = gamma
        self.omega = omega
        self.omega2 = omega*omega
        self.w = wavefunction

    def local_energy(self, positions):
        """Return the local energy."""

        Xi = 0.0
        fd, sd = self.w.gradients_wavefunction(positions)
        interaction = self.interaction(positions)
        for i in range(self.s.M):
            Xi += self.positions[i]
        local_energy = 0.5*(-fd*fd + sd + self.omega2*Xi) + interaction

        return local_energy

    def local_energy_gibbs(self, positions):
        """Return the local energy for gibbs sampling."""

        Xi = 0.0
        fd, sd = self.w.quandratic_gradients_wavefunction(positions)
        interaction = self.interaction(positions)
        for i in range(self.s.M):
            Xi += self.positions[i]
        local_energy = 0.5*(-fd*fd + sd + self.omega2*Xi) + interaction

        return local_energy

    def interaction(self, positions):
        """Return the interaction between particles"""

        "add interaction"

        return 0
