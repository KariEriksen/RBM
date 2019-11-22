"""Wavefunction class."""
import math
import numpy as np


class Wavefunction:
    """Contains parameters of system and wave equation."""

    def __init__(self, num_particles, num_dimensions, N, a, b, W, sigma):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.M = self.num_p*self.num_d
        self.N = N
        self.a = a
        self.b = b
        self.W = W
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.sigma4 = self.sigma2**2
        self.dpsi_da = np.zeros((1, self.M))
        self.dpsi_db = np.zeros((1, self.N))
        self.dpsi_dW = np.zeros((self.M, self.N))

    def wavefunction(self, positions):
        """Return the NQS wave function ."""
        """Partition function will disappear when it divided by itself
        in the probability term or in the energy term where its derivative
        equals zero."""

        Z = 1.0
        rbm_visible = rbm_connected = 0.0
        rbm_hidden = 1.0
        for i in range(self.M):
            rbm_visible += ((positions[i] - self.a[i])**2)/(2*self.sigma2)

        for j in range(self.N):
            for i in range(self.M):
                rbm_connected += positions[i]*self.W[i, j]/self.sigma2
            rbm_hidden *= 1 + math.exp(self.b[j]) + rbm_connected

        wavefunction = (1/Z)*math.exp(rbm_visible)*rbm_hidden
        return wavefunction

    def derivatives_wavefunction(self, positions):
        """Return the first and second derivative of ln of the wave function"""

        first_derivative = 0.0
        second_derivative = 0.0

        for i in range(self.s.M):
            sum2 = 0.0
            sum3 = 0.0
            for j in range(self.s.N):
                sum1 = 0.0
                for k in range(self.s.M):
                    sum1 += self.positions[k]*self.s.W[k, j]/self.s.sigma2

                exponent = math.exp(-self.s.b[j] - sum1)
                sum2 += self.s.W[i, j]/(1 + exponent)
                sum3 += sum2*sum2*exponent

            first_derivative += (-(self.positions[i] - self.a[i])/self.s.sigma2
                                 + (1/self.s.sigma2)*sum2)

            second_derivative += -1/self.s.sigma2 + (1/self.s.sigma4)*sum3

        return first_derivative, second_derivative

    def quandratic_derivatives_wavefunction(self, positions):
        """Return the first and second derivative of ln of the"""
        """quadratic wave function"""
        """Used in Gibbs sampling"""

        first_derivative_gibbs = 0.5*self.derivatives_wavefunction[0]
        second_derivative_gibbs = 0.5*self.derivatives_wavefunction[1]

        return first_derivative_gibbs, second_derivative_gibbs

    def derivative_wavefunction_params(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        for k in range(self.M):
            for n in range(self.N):
                sum = 0.0
        for i in range(self.M):
            sum += positions[i]*self.W[i, n]/self.sigma2

        dpsi_da = (1/self.sigma2)*(positions[k] - self.a[k])
        dpsi_db = 1/(1 + math.exp(-self.b[n] - sum))
        dpsi_dW = (positions[k]/self.sigma2)*dpsi_db

        return dpsi_da, dpsi_db, dpsi_dW

    def probability(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""

        wf_old = self.s.wavefunction(positions)
        wf_new = self.s.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio
