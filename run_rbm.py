"""Restricted Boltzmann Machine with Variational Monte Carlo."""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401
from sampler import Sampler # noqa: 401

"""
Restricted Boltzmann Machine with Variational Monte Carlo.
Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles = 10000
num_particles = 2
num_dimensions = 3
hidden_nodes = 3
visible_nodes = num_particles*num_dimensions
numerical_step_length = 0.001
step_metropolis = 1.0
step_importance = 0.01
omega = 1.0
gamma = 1.0
learning_rate = 0.01
gradient_iterations = 1000
sigma = 1.0

opt = Optimizer(learning_rate)
# Initialize the variational parameters
a = np.zeros((1, visible_nodes))
b = np.zeros((1, hidden_nodes))
W = np.zeros((visible_nodes, hidden_nodes))


def run_vmc(a_i, b_j, W_ij):
    """Run the variational monte carlo."""
    # Set all values to zero for each new Monte Carlo run
    accumulate_energy = 0.0
    accumulate_psi_term = np.array((1, 3))
    accumulate_both = np.array((1, 3))
    new_energy = 0.0

    # Initialize the posistions for each new Monte Carlo run
    positions = np.random.rand(num_particles, num_dimensions)

    # Call system class in order to set new parameters
    sys = Wavefunction(num_particles, num_dimensions, hidden_nodes,
                       a_i, b_j, W_ij, sigma)
    sam = Sampler(gamma, omega, numerical_step_length, sys)
    met = Metropolis(step_metropolis, step_importance, num_particles,
                     num_dimensions, sam, 0.0)
    for i in range(monte_carlo_cycles):

        new_energy, new_positions, count = met.metropolis(positions)
        # new_energy, new_positions, count = met.importance_sampling(positions)
        positions = new_positions
        accumulate_energy += sam.local_energy(positions)
        for i in range(3):
            accumulate_psi_term[i] += sys.derivative_wavefunction(positions)[i]
            accumulate_both[i] += sam.local_energy_times_wf(positions)[i]

    expec_val_energy = accumulate_energy/(monte_carlo_cycles)
    expec_val_psi_a = accumulate_psi_term[0]/(monte_carlo_cycles)
    expec_val_psi_b = accumulate_psi_term[1]/(monte_carlo_cycles)
    expec_val_psi_W = accumulate_psi_term[2]/(monte_carlo_cycles)
    expec_val_both_a = accumulate_both[0]/(monte_carlo_cycles)
    expec_val_both_b = accumulate_both[1]/(monte_carlo_cycles)
    expec_val_both_W = accumulate_both[2]/(monte_carlo_cycles)

    derivative_energy_a = 2*(expec_val_both_a -
                             expec_val_psi_a*expec_val_energy)
    derivative_energy_b = 2*(expec_val_both_b -
                             expec_val_psi_b*expec_val_energy)
    derivative_energy_W = 2*(expec_val_both_W -
                             expec_val_psi_W*expec_val_energy)

    print 'deri energy param a = ', derivative_energy_a
    print 'deri energy param b = ', derivative_energy_b
    print 'deri energy param W = ', derivative_energy_W
    print 'counter (accepted moves in metropolis) = ', count
    return derivative_energy_a, derivative_energy_b, derivative_energy_W,
    expec_val_energy


for i in range(gradient_iterations):

    d_El_a, d_El_b, d_El_W, energy = run_vmc(a, b, W)
    new_a, new_b, new_W = opt.gradient_descent(a, b, W, d_El_a, d_El_b, d_El_W)
    a = new_a
    b = new_b
    W = new_W
    e = 0.5*num_dimensions*num_particles
    # prints total energy of the system, NOT divided by N
    print 'new a =  ', new_a
    print 'new b =  ', new_b
    print 'new W =  ', new_W
    print '----------------------------'
    print 'total energy =  ', energy, 'correct energy = ', e
