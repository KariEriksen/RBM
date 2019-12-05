"""Wavefunction class."""
import math
import numpy as np


class Wavefunction:
    """Contains parameters of system and wave equation."""

    def __init__(self, M, N, a, b, W, sigma):
        """Instance of class."""
        self.M = M
        self.N = N
        self.a = a
        self.b = b
        self.W = W
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.sigma4 = self.sigma2**2

    def wavefunction(self, positions):
        """Return the NQS wave function ."""
        """Partition function will disappear when it divided by itself
        in the probability term or in the energy term where its derivative
        equals zero."""

        sum1 = sum2 = 0.0
        prod = 1.0
        for i in range(self.M):
            sum1 += ((positions[i] - self.a[i])**2)/(2*self.sigma2)
        for j in range(self.N):
            for i in range(self.M):
                sum2 += positions[i]*self.W[i, j]/self.sigma2
            sum2 = self.b[j] + sum2
            prod *= 1 + math.exp(sum2)

        wavefunction = math.exp(sum1)*prod
        return wavefunction

    def gradients_wavefunction(self, positions):
        """Return the first and second derivative of ln of the wave function"""

        first_derivative = 0.0
        second_derivative = 0.0

        for i in range(self.M):
            sum2 = 0.0
            sum3 = 0.0
            for j in range(self.N):
                sum1 = 0.0
                for k in range(self.M):
                    sum1 += positions[k]*self.W[k, j]/self.sigma2

                exponent = math.exp(-self.b[j] - sum1)
                sum2 += self.W[i, j]/(1 + exponent)
                sum3 += sum2*sum2*exponent

            first_derivative += (-(positions[i] - self.a[i])/self.sigma2
                                 + (1/self.sigma2)*sum2)

            second_derivative += -1/self.sigma2 + (1/self.sigma4)*sum3

        return first_derivative, second_derivative

    def quandratic_gradients_wavefunction(self, positions):
        """Return the first and second derivative of ln of the"""
        """quadratic wave function"""
        """Used in Gibbs sampling"""

        first_derivative_gibbs = 0.5*self.derivatives_wavefunction[0]
        second_derivative_gibbs = 0.5*self.derivatives_wavefunction[1]

        return first_derivative_gibbs, second_derivative_gibbs

    def gradient_wavefunction_a(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        dpsi_da = np.array(self.M)

        for k in range(self.M):
            dpsi_da[k] = (1/self.sigma2)*(positions[k] - self.a[k])

        return dpsi_da

    def gradient_wavefunction_b(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        dpsi_db = np.array(self.N)

        for n in range(self.N):
            sum = 0.0
            for i in range(self.M):
                sum += positions[i]*self.W[i, n]/self.sigma2

            dpsi_db[n] = 1/(1 + math.exp(-self.b[n] - sum))

        return dpsi_db

    def gradient_wavefunction_W(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        dpsi_dW = np.array((self.M, self.N))

        for k in range(self.M):
            for n in range(self.N):
                sum = 0.0
            for i in range(self.M):
                sum += positions[i]*self.W[i, n]/self.sigma2

            term = 1/(1 + math.exp(-self.b[n] - sum))
            dpsi_dW[k, n] = (positions[k]/self.sigma2)*term

        return dpsi_dW

    def wavefunction_ratio(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""

        wf_old = self.wavefunction(positions)
        wf_new = self.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio

    def sigmoid(self, positions):
        """Calculate the sigmoid function given positions"""
        # sigmoid = np.zeros(self.M)
        sigmoid = 1/(1 + np.exp(positions))
        return sigmoid
