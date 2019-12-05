"""Sampler class."""
import numpy as np


class Sampler:
    """Calculate variables regarding energy of given wavefunction."""

    def __init__(self, wavefunction, hamiltonian):
        """Instance of class."""
        self.w = wavefunction
        self.h = hamiltonian

        self.local_energy = 0.0
        self.alpha_gradient_wf = 0.0
        self.accumulate_energy = 0.0
        self.accumulate_psi_term = np.zeros(3)
        self.accumulate_both = np.zeros(3)
        self.expec_val_energy = 0.0
        self.expec_val_psi = np.zeros(3)
        self.expec_val_both = np.zeros(3)
        self.derivative_energy = np.zeros(3)

    def sample_values(self, positions):
        """Get the local energy from Hamiltonian class"""
        """Sample important values"""

        self.local_energy = self.h.local_energy(positions)
        self.accumulate_energy += self.h.local_energy(positions)
        alpha_gradient_wf = self.w.alpha_gradient_wavefunction(positions)

        for i in range(3):
            self.accumulate_psi_term[i] += alpha_gradient_wf[i]
            self.accumulate_both[i] += alpha_gradient_wf[i]*self.local_energy

    def average_values(self, monte_carlo_cycles):
        """Calculate average values"""

        mcc = monte_carlo_cycles
        self.expec_val_energy = self.accumulate_energy/mcc

        # for all variational parameters
        # calcualte the expectation values and the derivative of
        # the energy
        for i in range(3):
            self.expec_val_psi[i] = self.accumulate_psi_term[i]/mcc
            self.expec_val_both[i] = self.accumulate_both[i]/mcc
            self.derivative_energy[i] = 2*(self.expec_val_both[i] -
                                           (self.expec_val_psi[i] *
                                            self.expec_val_energy))

    def print_avereges(self):

        print ('deri energy param a = ', self.derivative_energy[0])
        print ('deri energy param b = ', self.derivative_energy[1])
        print ('deri energy param W = ', self.derivative_energy[2])
        print ('total energy =  ', self.local_energy)
        # energy/num_particles
        print ('----------------------------')
