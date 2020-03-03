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
        self.sqrt_delta_t = np.sqrt(self.delta_t)

    def metropolis_step(self, positions):
        """Calculate new metropolis step."""
        """with brute-force sampling of new positions."""

        r = np.zeros(self.num_d)
        for i in range(self.num_d):
            r[i] = np.random.uniform(-1, 1)
        # r = random.random()*random.choice((-1, 1))
        # Pick a random particle
        random_index = random.randrange(self.num_p)
        j = random_index*self.num_d
        new_positions = np.array(positions)
        for i in range(self.num_d):
            # Suggest a new move
            new_positions[j+i] += r[i]*self.delta_R

        acceptance_ratio = self.w.wavefunction_ratio(positions, new_positions)
        epsilon = np.random.sample()

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        return positions

    def importance_sampling_step(self, positions):
        """Calculate new step with Importance sampling."""
        """With upgrad method for suggetion of new positions."""
        """Given through the Langevin equation.
        D is the diffusion coefficient equal 0.5, xi is a gaussion random
        variable and delta_t is the time step between 0.001 and 0.01"""

        D = 0.5
        greens_function = 0.0
        F_old = self.w.quantum_force(positions)

        r = np.zeros(self.num_d)
        for i in range(self.num_d):
            r[i] = random.gauss(0, 1)
        # Pick a random particle
        random_index = random.randrange(self.num_p)
        j = random_index*self.num_d
        new_positions = np.array(positions)
        for i in range(self.num_d):
            # Suggest a new move
            term1 = D*F_old[j+i]*self.delta_t
            term2 = r[i]*self.sqrt_delta_t
            new_positions[j+i] += term1 + term2

        prob_ratio = self.w.wavefunction_ratio(positions, new_positions)
        F_new = self.w.quantum_force(new_positions)

        for i in range(self.num_p*self.num_d):
            term1 = 0.5*((F_old[i] + F_new[i]) *
                         (positions[i] - new_positions[i]))
            term2 = D*self.delta_t*(F_old[i] - F_new[i])
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

            term = -self.w.b[j] - sum
            exponent = math.exp(term)
            P_hj = 1.0/(1 + exponent)
            r = np.random.uniform(0, 1)
            if P_hj > r:
                h_j[j] = 1.0
            else:
                h_j[j] = 0.0

        for i in range(self.w.M):
            sum = 0.0
            for j in range(self.w.N):
                sum += self.w.W[i, j]*h_j[j]

            mu = self.w.a[i] + sum
            # acceptance with probability of 1
            positions[i] = np.random.normal(mu, sigma)

        self.c += 1.0
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
            self.s.sample_values(positions, False)
        self.s.average_values(self.mc_cycles)

        # the derivative_energy is an array
        d_El_a = self.s.derivative_energy_a
        d_El_b = self.s.derivative_energy_b
        d_El_W = self.s.derivative_energy_W
        self.print_averages()
        return d_El_a, d_El_b, d_El_W, self.s.local_energy

    def run_importance_sampling(self):
        """Run importance algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p*self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.importance_sampling_step(positions)
            positions = new_positions
            self.s.sample_values(positions, False)
        self.s.average_values(self.mc_cycles)

        # the derivative_energy is an array
        d_El_a = self.s.derivative_energy_a
        d_El_b = self.s.derivative_energy_b
        d_El_W = self.s.derivative_energy_W
        self.print_averages()
        return d_El_a, d_El_b, d_El_W, self.s.local_energy

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
            self.s.sample_values(positions, True)
        self.s.average_values(self.mc_cycles)

        # the derivative_energy is an array
        d_El_a = self.s.derivative_energy_a
        d_El_b = self.s.derivative_energy_b
        d_El_W = self.s.derivative_energy_W
        self.print_averages()
        return d_El_a, d_El_b, d_El_W, self.s.local_energy

    def run_one_body_sampling(self):
        """Sample position of particles."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p*self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        density_adding = np.zeros(41)

        # Run Metropolis while finding one body density
        for i in range(self.mc_cycles):
            new_positions = self.metropolis_step(positions)
            positions = new_positions
            density = self.one_body_density(positions)
            density_adding += density
            # self.sam.sample_values(positions)

        # self.sam.average_values(self.mc_cycles)
        # self.print_averages()

        return density_adding

    def one_body_density(self, positions):
        """Run one-body density count."""

        num_radii = 41
        density = np.zeros(num_radii)
        r_vec = np.linspace(0, 4, num_radii)
        step = r_vec[1] - r_vec[0]

        # Calculate the distance from origo of each particle
        radii = np.zeros(self.num_p)
        n = self.num_d
        for i in range(self.num_p):
            r = 0
            for j in range(self.num_d):
                r += positions[i*n+j]*positions[i*n+j]
            radii[i] = math.sqrt(r)

        # Check in which segment each particle is in
        for i in range(self.num_p):
            dr = 0.0
            for j in range(num_radii):
                if(dr <= radii[i] < dr+step):
                    density[j] += 1
                    break
                else:
                    dr += step

        return density

    def blocking(self):
        """Sample for blocking"""
        """using importance sampling"""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p*self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()
        # Initialize sampler method for each new Monte Carlo run
        energy = np.zeros(self.mc_cycles)

        for i in range(self.mc_cycles):
            # new_positions = self.importance_sampling_step(positions)
            new_positions = self.metropolis_step(positions)
            positions = new_positions
            self.sam.sample_values(positions)
            energy[i] = self.sam.local_energy

        self.sam.average_values(self.mc_cycles)
        self.print_averages()
        return energy

    def print_averages(self):

        print ('acceptance rate = ', self.c/self.mc_cycles)
        print ('a parameter = ', self.w.a)
        print ('b parameter = ', self.w.b)
        print ('W parameter = ', self.w.W)
        print ('deri energy param a = ', self.s.derivative_energy_a)
        print ('deri energy param b = ', self.s.derivative_energy_b)
        print ('deri energy param W = ', self.s.derivative_energy_W)
        print ('\033[1m total energy \033[0m =  ', self.s.local_energy)
        print ('variance = ', self.s.variance)
        # energy/num_particles
        print ('----------------------------')
