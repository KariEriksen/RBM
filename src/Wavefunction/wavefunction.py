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

        sum1 = 0.0
        prod = 1.0
        for i in range(self.M):
            sum1 += ((positions[i] - self.a[i])**2)/(2*self.sigma2)
        for j in range(self.N):
            sum2 = 0.0
            for i in range(self.M):
                sum2 += positions[i]*self.W[i, j]/self.sigma2
            prod *= (1 + math.exp(self.b[j] + sum2))

        wavefunction = math.exp(-sum1)*prod
        return wavefunction

    def gradients_wavefunction(self, positions):
        """Return the first and second derivative of ln of the wave function"""

        first_derivative = 0.0
        second_derivative = 0.0

        # TOMORROW
        """Is it the weights that are initialized to high?"""

        for k in range(self.M):
            sum2 = 0.0
            sum3 = 0.0
            for j in range(self.N):
                sum1 = 0.0
                for i in range(self.M):
                    sum1 += positions[i]*self.W[i, j]/self.sigma2

                exponent = math.exp(-self.b[j] - sum1)
                denominator = 1.0 + exponent

                sigmoid = 1.0/denominator
                sigmoid_deri = exponent/(denominator*denominator)
                sum2 += self.W[k, j]*sigmoid
                sum3 += self.W[k, j]*self.W[k, j]*sigmoid_deri

            first_derivative += (-(positions[k] - self.a[k])/self.sigma2
                                 + sum2/self.sigma2)

            second_derivative += -1.0/self.sigma2 + sum3/self.sigma4
        # print -first_derivative*first_derivative
        # print second_derivative
        # ksk
        return first_derivative, second_derivative

    def quandratic_gradients_wavefunction(self, positions):
        """Return the first and second derivative of ln of the"""
        """quadratic wave function"""
        """Used in Gibbs sampling"""

        first, second = self.gradients_wavefunction(positions)
        first_derivative_gibbs = 0.5*first
        second_derivative_gibbs = 0.5*second

        return first_derivative_gibbs, second_derivative_gibbs

    def gradient_wavefunction_a(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        dpsi_da = np.zeros(self.M)

        for k in range(self.M):
            dpsi_da[k] = (1/(2*self.sigma2))*(positions[k] - self.a[k])

        return dpsi_da

    def gradient_wavefunction_b(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        dpsi_db = np.zeros(self.N)

        for n in range(self.N):
            sum = 0.0
            for i in range(self.M):
                sum += positions[i]*self.W[i, n]/self.sigma2

            dpsi_db[n] = 1/(1 + math.exp(-self.b[n] - sum))
        dpsi_db = 0.5*dpsi_db

        return dpsi_db

    def gradient_wavefunction_W(self, positions):
        """Return the first derivative of ln of the wave function"""
        """with respect to the variational parameter a, b and W"""
        """This is equivalant to the first derivative of the wave
        function divided by the wave equation"""

        dpsi_dW = np.zeros((self.M, self.N))

        for k in range(self.M):
            for n in range(self.N):
                sum = 0.0
                for i in range(self.M):
                    sum += positions[i]*self.W[i, n]/self.sigma2

                term = 1/(1 + math.exp(-self.b[n] - sum))
                dpsi_dW[k, n] = term*positions[k]/(2*self.sigma2)

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

    def quantum_force(self, positions):
        """Return the first derivative of ln of the wave function"""

        first_derivative = np.zeros(self.M)

        for k in range(self.M):
            sum2 = 0.0
            for j in range(self.N):
                sum1 = 0.0
                for i in range(self.M):
                    sum1 += positions[i]*self.W[i, j]/self.sigma2

                exponent = math.exp(-self.b[j] - sum1)
                denominator = 1.0 + exponent

                sigmoid = 1.0/denominator
                sum2 += self.W[k, j]*sigmoid
            pos_neg = positions[k] - self.a[k]
            first_derivative[k] = (-pos_neg/self.sigma2 + sum2/self.sigma2)

        return first_derivative
