"""Hamiltonian class."""
import math
import numpy as np


class Hamiltonian:
    """Calculate variables regarding energy of given system."""

    def __init__(self, gamma, omega, num_d, num_p, wavefunction, interaction):
        """Instance of class."""
        self.gamma = gamma
        self.omega = omega
        self.omega2 = omega*omega
        self.num_d = num_d
        self.num_p = num_p
        self.w = wavefunction
        self.interaction = interaction

    def local_energy(self, positions):
        """Return the local energy."""

        Xi = 0.0
        first_deri, second_deri = self.w.gradients_wavefunction(positions)
        interaction_energy = self.interaction_energy(positions)
        for i in range(self.w.M):
            Xi += positions[i]*positions[i]

        local_energy = 0.5*(-first_deri*first_deri +
                            second_deri + self.omega2*Xi)

        if self.interaction == 'true':
            local_energy += interaction_energy
        else:
            None

        return local_energy

    def local_energy_gibbs(self, positions):
        """Return the local energy for gibbs sampling."""

        Xi = 0.0
        # quandratic_gradients_wavefunction returns the gradients times 0.5
        fd, sd = self.w.quandratic_gradients_wavefunction(positions)
        interaction_energy = self.interaction_energy(positions)
        for i in range(self.w.M):
            Xi += positions[i]*positions[i]
        local_energy = 0.5*(-fd*fd + sd + self.omega2*Xi)

        if self.interaction:
            local_energy += interaction_energy
        else:
            None

        return local_energy

    def local_energy_numerical(self, positions):
        """Return the local energy using numerical"""
        """calculation of derivatives"""

        Xi = 0.0
        laplacian = self.laplacian_numerical(positions)
        interaction_energy = self.interaction_energy(positions)
        for i in range(self.w.M):
            Xi += positions[i]
        local_energy = -0.5*laplacian + self.omega2*Xi*Xi + interaction_energy

        return local_energy

    def gradient_numerical(self, positions):
        """Numerical differentiation for solving gradient."""
        """f'(x) = f(x + h) - f(x)/h"""

        step = 0.001
        position_forward = np.array(positions)
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.w.M):
            psi_current += self.w.wavefunction(positions)
            position_forward[i] = position_forward[i] + step
            wf_p = self.w.wavefunction(position_forward)
            psi_moved += wf_p
            # Resett positions
            position_forward[i] = position_forward[i] - step

        gradient = (psi_moved - psi_current)/(step*step)
        return gradient

    def laplacian_numerical(self, positions):
        """Numerical differentiation for solving laplacian."""
        """f''(x) = f(x + h) - 2f(x) + f(x - h)/h^2"""

        step = 0.001
        position_forward = np.array(positions)
        position_backward = np.array(positions)
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.w.M):
            psi_current += 2*self.w.wavefunction(positions)
            position_forward[i] = position_forward[i] + step
            position_backward[i] = position_backward[i] - step
            wf_p = self.w.wavefunction(position_forward)
            wf_n = self.w.wavefunction(position_backward)
            psi_moved += wf_p + wf_n
            # Resett positions
            position_forward[i] = position_forward[i] - step
            position_backward[i] = position_backward[i] + step

        laplacian = (psi_moved - psi_current)/(step*step)
        return laplacian

    def interaction_energy(self, positions):
        """Return the interaction between particles"""

        n = self.num_d
        interaction = 0.0
        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                r = 0.0
                for k in range(self.num_d):
                    # ri_minus_rj = np.subtract(positions[i], positions[j+1])

                    ri_minus_rj = positions[(i*n)+k] - positions[((j+1)*n)+k]
                    r += ri_minus_rj**2
                interaction += 1.0/math.sqrt(r)

        return interaction
