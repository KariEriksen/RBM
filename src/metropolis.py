"""Metropolis class."""
import numpy as np
import random
import math
from sampler import Sampler # noqa: 401


class Metropolis:
    """Metropolis methods."""

    # Hamiltonian(omega, step)

    def __init__(self, monte_carlo_steps, delta_R, delta_t, num_particles,
                 num_dimensions, wavefunction, hamiltonian):
        """Instance of class."""
        self.mc_cycles = monte_carlo_steps
        self.delta_R = delta_R
        self.delta_t = delta_t
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.w = wavefunction
        self.h = hamiltonian
        self.c = 0.0

        self.s = Sampler(self.w, self.h)

    def metropolis_step(self, positions):
        """Calculate new metropolis step."""
        """with brute-force sampling of new positions."""

        r = random.random()*random.choice((-1, 1))
        # Pick a random particle
        random_index = random.randrange(self.num_p)
        j = random_index*self.num_d
        new_positions = np.array(positions)
        for i in range(self.num_d):
            # Suggest a new move
            new_positions[j+i] += r*self.delta_R

        acceptance_ratio = self.w.wavefunction_ratio(positions, new_positions)
        epsilon = np.random.sample()

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        return positions

    def importance_sampling_step(self, positions, analytic):
        """Calculate new step with Importance sampling."""
        """With upgrad method for suggetion of new positions."""
        """Given through the Langevin equation.
        D is the diffusion coefficient equal 0.5, xi is a gaussion random
        variable and delta_t is the time step between 0.001 and 0.01"""

        D = 0.5
        greens_function = 0.0

        if analytic:
            F_old = self.w.quantum_force(positions)
        else:
            F_old = self.w.quantum_force_numerical(positions)

        r = random.random()*random.choice((-1, 1))
        # Pick a random particle and calculate new position
        random_index = random.randrange(len(positions))
        new_positions = np.array(positions)

        term1 = D*F_old[random_index, :]*self.delta_t
        term2 = r*np.sqrt(self.delta_t)
        new_random_position = new_positions[random_index, :] + term1 + term2
        new_positions[random_index, :] = new_random_position
        prob_ratio = self.w.wavefunction_ratio(positions, new_positions)

        if analytic:
            F_new = self.w.quantum_force(new_positions)
        else:
            F_new = self.w.quantum_force_numerical(new_positions)

        for i in range(self.num_p):
            for j in range(self.num_d):
                term1 = 0.5*((F_old[i, j] + F_new[i, j]) *
                             (positions[i, j] - new_positions[i, j]))
                term2 = D*self.delta_t*(F_old[i, j] - F_new[i, j])
                greens_function += term1 + term2

        greens_function = np.exp(greens_function)

        epsilon = np.random.sample()
        acceptance_ratio = prob_ratio*greens_function

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        return positions

    def gibbs_step(self, positions):
        """Calculate new Gibbs step."""

        h_j = np.zeros(self.w.N)
        sigma = 1.0
        sigma2 = 1.0

        for j in range(self.w.N):
            sum = 0.0
            for i in range(self.w.M):
                sum += positions[i]*self.w.W[i, j]/sigma2

            b = self.w.b[j] + sum
            exponent = math.exp(-b)
            h_j[j] = 1.0/(1 + exponent)

        for i in range(self.w.M):
            sum = 0.0
            for j in range(self.w.N):
                sum += self.w.W[i, j]*h_j[j]

            mu = self.w.a[i] + sum
            # acceptance with probability of 1
            positions[i] = np.random.normal(mu, sigma)

        return positions

    def run_metropolis(self):
        """Run the naive metropolis algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p*self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.metropolis_step(positions)
            positions = new_positions
            self.s.sample_values(positions, 'false')
        self.s.average_values(self.mc_cycles)

        # the derivative_energy is an array
        d_El_a = self.s.derivative_energy_a
        d_El_b = self.s.derivative_energy_b
        d_El_W = self.s.derivative_energy_W
        self.print_averages()
        return d_El_a, d_El_b, d_El_W

    def run_importance_sampling(self, analytic):
        """Run importance algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p*self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.importance_sampling_step(positions, analytic)
            positions = new_positions
            self.s.sample_values(positions, 'false')
        self.s.average_values(self.mc_cycles)

        # the derivative_energy is an array
        d_El_a = self.s.derivative_energy_a
        d_El_b = self.s.derivative_energy_b
        d_El_W = self.s.derivative_energy_W
        self.print_averages()
        return d_El_a, d_El_b, d_El_W

    def run_gibbs_sampling(self):
        """Run Gibbs sampling."""
        """Fix"""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p*self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.gibbs_step(positions)
            positions = new_positions
            self.s.sample_values(positions, 'true')
        self.s.average_values(self.mc_cycles)

        # the derivative_energy is an array
        d_El_a = self.s.derivative_energy_a
        d_El_b = self.s.derivative_energy_b
        d_El_W = self.s.derivative_energy_W
        self.print_averages()
        return d_El_a, d_El_b, d_El_W

    def print_averages(self):

        print ('acceptance rate = ', self.c/self.mc_cycles)
        print ('a parameter = ', self.w.a)
        print ('b parameter = ', self.w.b)
        print ('W parameter = ', self.w.W)
        print ('deri energy param a = ', self.s.derivative_energy_a)
        print ('deri energy param b = ', self.s.derivative_energy_b)
        print ('deri energy param W = ', self.s.derivative_energy_W)
        print ('\033[1m total energy \033[0m =  ', self.s.local_energy)
        # energy/num_particles
        print ('----------------------------')
