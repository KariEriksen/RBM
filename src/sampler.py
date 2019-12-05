"""Sampler class."""
import numpy as np


class Sampler:
    """Calculate variables regarding energy of given wavefunction."""

    def __init__(self, wavefunction, hamiltonian):
        """Instance of class."""
        self.w = wavefunction
        self.h = hamiltonian

        self.local_energy = 0.0
        self.accumulate_energy = 0.0
        self.accumulate_psi_term_a = np.array()
        self.accumulate_psi_term_b = np.array()
        self.accumulate_psi_term_W = np.array()
        self.accumulate_both_a = np.array()
        self.accumulate_both_b = np.array()
        self.accumulate_both_W = np.array()
        self.expec_val_energy = 0.0
        self.expec_val_psi_a = np.array()
        self.expec_val_psi_b = np.array()
        self.expec_val_psi_W = np.array()
        self.expec_val_both_a = np.array()
        self.expec_val_both_b = np.array()
        self.expec_val_both_W = np.array()
        self.derivative_energy_a = np.array()
        self.derivative_energy_b = np.array()
        self.derivative_energy_W = np.array()

    def sample_values(self, positions):
        """Get the local energy from Hamiltonian class"""
        """Sample important values"""

        self.local_energy = self.h.local_energy(positions)
        self.accumulate_energy += self.h.local_energy(positions)
        gradient_wf_a = self.w.gradient_wavefunction_a(positions)
        gradient_wf_b = self.w.gradient_wavefunction_b(positions)
        gradient_wf_W = self.w.gradient_wavefunction_W(positions)

        self.accumulate_psi_term_a += gradient_wf_a
        self.accumulate_psi_term_b += gradient_wf_b
        self.accumulate_psi_term_W += gradient_wf_W
        self.accumulate_both += gradient_wf_a*self.local_energy

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
            derivative_energy_a = self.derivative_energy[i]
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
