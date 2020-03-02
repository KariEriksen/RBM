"""Sampler class."""
import numpy as np


class Sampler:
    """Calculate variables regarding energy of given wavefunction."""

    def __init__(self, wavefunction, hamiltonian):
        """Instance of class."""
        self.w = wavefunction
        self.h = hamiltonian

    def sample_values(self, positions, gibbs):
        """Get the local energy from Hamiltonian class"""
        """Sample important values"""

        if gibbs:
            self.local_energy = self.h.local_energy_gibbs(positions)
            self.accumulate_energy += self.h.local_energy_gibbs(positions)
            self.accumulate_energy_sq += self.local_energy*self.local_energy
            gradient_wf_a = 0.5*self.w.gradient_wavefunction_a(positions)
            gradient_wf_b = 0.5*self.w.gradient_wavefunction_b(positions)
            gradient_wf_W = 0.5*self.w.gradient_wavefunction_W(positions)
        else:
            self.local_energy = self.h.local_energy(positions)
            self.accumulate_energy += self.h.local_energy(positions)
            self.accumulate_energy_sq += self.local_energy*self.local_energy
            gradient_wf_a = self.w.gradient_wavefunction_a(positions)
            gradient_wf_b = self.w.gradient_wavefunction_b(positions)
            gradient_wf_W = self.w.gradient_wavefunction_W(positions)
        # self.local_energy = self.h.local_energy_numerical(positions)
        # self.accumulate_energy += self.h.local_energy_numerical(positions)
        # gradient_wf_a = np.zeros(self.w.M)
        # gradient_wf_b = np.zeros(self.w.N)
        # gradient_wf_W = np.zeros((self.w.M, self.w.N))

        self.accumulate_psi_term_a += gradient_wf_a
        self.accumulate_psi_term_b += gradient_wf_b
        self.accumulate_psi_term_W += gradient_wf_W
        self.accumulate_both_a += gradient_wf_a*self.local_energy
        self.accumulate_both_b += gradient_wf_b*self.local_energy
        self.accumulate_both_W += gradient_wf_W*self.local_energy

    def average_values(self, monte_carlo_cycles):
        """Calculate average values"""

        mcc = monte_carlo_cycles
        self.expec_val_energy = self.accumulate_energy/mcc

        # for all variational parameters
        # calcualte the expectation values and the derivative of
        # the energy
        self.expec_val_psi_a = self.accumulate_psi_term_a/mcc
        self.expec_val_psi_b = self.accumulate_psi_term_b/mcc
        self.expec_val_psi_W = self.accumulate_psi_term_W/mcc
        self.expec_val_both_a = self.accumulate_both_a/mcc
        self.expec_val_both_b = self.accumulate_both_b/mcc
        self.expec_val_both_W = self.accumulate_both_W/mcc

        self.derivative_energy_a = 2*(self.expec_val_both_a -
                                      (self.expec_val_psi_a *
                                       self.expec_val_energy))
        self.derivative_energy_b = 2*(self.expec_val_both_b -
                                      (self.expec_val_psi_b *
                                       self.expec_val_energy))
        self.derivative_energy_W = 2*(self.expec_val_both_W -
                                      (self.expec_val_psi_W *
                                       self.expec_val_energy))

        expec_energy_sq = self.expec_val_energy*self.expec_val_energy
        energy_sq = self.accumulate_energy_sq/monte_carlo_cycles
        self.variance = energy_sq - expec_energy_sq

    def initialize(self):
        """Set all sampling values to zero"""

        self.local_energy = 0.0
        self.accumulate_energy = 0.0
        self.accumulate_energy_sq = 0.0
        self.variance = 0.0
        self.accumulate_psi_term_a = np.zeros(self.w.M)
        self.accumulate_psi_term_b = np.zeros(self.w.N)
        self.accumulate_psi_term_W = np.zeros((self.w.M, self.w.N))
        self.accumulate_both_a = np.zeros(self.w.M)
        self.accumulate_both_b = np.zeros(self.w.N)
        self.accumulate_both_W = np.zeros((self.w.M, self.w.N))
        self.expec_val_energy = 0.0
        self.expec_val_psi_a = np.zeros(self.w.M)
        self.expec_val_psi_b = np.zeros(self.w.N)
        self.expec_val_psi_W = np.zeros((self.w.M, self.w.N))
        self.expec_val_both_a = np.zeros(self.w.M)
        self.expec_val_both_b = np.zeros(self.w.N)
        self.expec_val_both_W = np.zeros((self.w.M, self.w.N))
        self.derivative_energy_a = np.zeros(self.w.M)
        self.derivative_energy_b = np.zeros(self.w.N)
        self.derivative_energy_W = np.zeros((self.w.M, self.w.N))
