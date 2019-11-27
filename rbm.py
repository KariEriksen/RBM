"""Restricted Boltzmann Machine."""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.hamiltonian import Hamiltonian # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401
from sampler import Sampler # noqa: 401

"""
Restricted Boltzmann Machine with Gibbs sampling.
Alternative, Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

step_metropolis = 1.0
step_importance = 0.01
learning_rate = 0.01
gradient_iterations = 1000

opt = Optimizer(learning_rate)
# Initialize the variational parameters


def non_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                         hidden_nodes):
    """Run Restricted Boltzmann Machine."""

    visible_nodes = num_particles*num_dimensions
    a_i = np.zeros((1, visible_nodes))
    b_j = np.zeros((1, hidden_nodes))
    W_ij = np.zeros((visible_nodes, hidden_nodes))
    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    param_a = a_i
    param_b = b_j
    param_W = W_ij
    for i in range(gradient_iterations):

        # Call system class in order to set new parameters
        wave = Wavefunction(num_particles, num_dimensions, hidden_nodes,
                            param_a, param_b, param_W, sigma)
        hamilton = Hamiltonian(gamma, omega, wave)
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        d_El = met.run_metropolis()
        # Run with analytical expression for quantum force = true
        # d_El = met.run_importance_sampling('true')
        d_El_a = d_El[0]
        d_El_b = d_El[1]
        d_El_W = d_El[2]
        new_a, new_b, new_W = opt.gradient_descent(param_a, param_b, param_W,
                                                   d_El_a, d_El_b, d_El_W)

        print 'new alpha = ', new_a
        print 'number of gradien descent runs = ', i
        param_a = new_a
        param_b = new_b
        param_W = new_W


def weak_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                          hidden_nodes):
    """Run Restricted Boltzmann Machine."""

    visible_nodes = num_particles*num_dimensions
    a_i = np.zeros((1, visible_nodes))
    b_j = np.zeros((1, hidden_nodes))
    W_ij = np.zeros((visible_nodes, hidden_nodes))
    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    param_a = a_i
    param_b = b_j
    param_W = W_ij
    for i in range(gradient_iterations):

        # Call system class in order to set new parameters
        wave = Wavefunction(num_particles, num_dimensions, hidden_nodes,
                            param_a, param_b, param_W, sigma)
        hamilton = Hamiltonian(gamma, omega, wave)
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        d_El = met.run_metropolis()
        # Run with analytical expression for quantum force = true
        # d_El = met.run_importance_sampling('true')
        d_El_a = d_El[0]
        d_El_b = d_El[1]
        d_El_W = d_El[2]
        new_a, new_b, new_W = opt.gradient_descent(param_a, param_b, param_W,
                                                   d_El_a, d_El_b, d_El_W)

        print 'new alpha = ', new_a
        print 'number of gradien descent runs = ', i
        param_a = new_a
        param_b = new_b
        param_W = new_W


"""case(monte_carlo_cycles, number of particles,
        number of dimensions, number of hidden nodes)"""

non_interaction_case(100000, 2, 3, 3)
# weak_interaction_case(100000, 2, 3, 3)
