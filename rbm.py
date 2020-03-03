"""Restricted Boltzmann Machine."""

import numpy as np
import sys
import os
import csv
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.hamiltonian import Hamiltonian # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401
from sampler import Sampler # noqa: 401

"""
Restricted Boltzmann Machine with Gibbs sampling.
Alternative, Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

step_metropolis = 1.0
step_importance = 0.01
learning_rate = 0.1
gradient_iterations = 100

opt = Optimizer(learning_rate)
# Initialize the variational parameters


def non_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                         hidden_nodes):
    """Run Restricted Boltzmann Machine."""

    # Initialize weights and biases
    visible_nodes = num_particles*num_dimensions
    a_i = np.random.normal(0, 1, visible_nodes)
    b_j = np.random.normal(0, 1, hidden_nodes)
    W_ij = np.random.normal(0, 1, (visible_nodes, hidden_nodes))
    # a_i = np.random.rand(visible_nodes)
    # b_j = np.random.rand(hidden_nodes)
    # W_ij = np.random.rand(visible_nodes, hidden_nodes)
    # a_i = np.zeros(visible_nodes)
    # b_j = np.zeros(hidden_nodes)
    # W_ij = np.zeros((visible_nodes, hidden_nodes))

    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    param_a = a_i
    param_b = b_j
    param_W = W_ij

    d_El_array = np.zeros(gradient_iterations)
    energy_array = np.zeros(gradient_iterations)
    parameter_array = np.zeros(gradient_iterations)
    var_array = np.zeros(gradient_iterations)
    for i in range(gradient_iterations):

        # Call system class in order to set new parameters
        wave = Wavefunction(visible_nodes, hidden_nodes,
                            param_a, param_b, param_W, sigma)
        # Hamiltonian(..., weak_interaction, strong_interaction)
        hamilton = Hamiltonian(gamma, omega, num_dimensions, num_particles,
                               wave, False, False)
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        d_El = met.run_metropolis()
        # d_El = met.run_importance_sampling()
        # d_El = met.run_gibbs_sampling()
        d_El_a = d_El[0]
        d_El_b = d_El[1]
        d_El_W = d_El[2]

        new_a, new_b, new_W = opt.gradient_descent(param_a, param_b, param_W,
                                                   d_El_a, d_El_b, d_El_W)

        print ('number of gradien descent runs = ', i)
        param_a = new_a
        param_b = new_b
        param_W = new_W

        # d_El_array[i] = d_El
        # var_array[i] = var
        energy_array[i] = d_El[3]

    plt.plot(energy_array)
    plt.show()


def weak_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                          hidden_nodes):
    """Run Restricted Boltzmann Machine."""

    # Initialize weights and biases
    visible_nodes = num_particles*num_dimensions
    a_i = np.random.rand(visible_nodes)
    b_j = np.random.rand(hidden_nodes)
    W_ij = np.random.rand(visible_nodes, hidden_nodes)
    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    param_a = a_i
    param_b = b_j
    param_W = W_ij
    for i in range(gradient_iterations):

        # Call system class in order to set new parameters
        wave = Wavefunction(visible_nodes, hidden_nodes,
                            param_a, param_b, param_W, sigma)
        # Hamiltonian(..., weak_interaction, strong_interaction)
        hamilton = Hamiltonian(gamma, omega, num_dimensions, num_particles,
                               wave, True, False)
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

        print ('number of gradien descent runs = ', i)
        param_a = new_a
        param_b = new_b
        param_W = new_W


def strong_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                            hidden_nodes):
    """Run Restricted Boltzmann Machine."""

    # Initialize weights and biases
    visible_nodes = num_particles*num_dimensions
    a_i = np.random.rand(visible_nodes)
    b_j = np.random.rand(hidden_nodes)
    W_ij = np.random.rand(visible_nodes, hidden_nodes)
    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    param_a = a_i
    param_b = b_j
    param_W = W_ij
    for i in range(gradient_iterations):

        # Call system class in order to set new parameters
        wave = Wavefunction(visible_nodes, hidden_nodes,
                            param_a, param_b, param_W, sigma)
        # Hamiltonian(..., weak_interaction, strong_interaction)
        hamilton = Hamiltonian(gamma, omega, num_dimensions, num_particles,
                               wave, False, True)
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

        print ('number of gradien descent runs = ', i)
        param_a = new_a
        param_b = new_b
        param_W = new_W


def one_body_density(monte_carlo_cycles, num_particles, num_dimensions,
                     hidden_nodes):
    """Run the variational monte carlo"""
    """using brute force"""

    # Set optimal values for weights and biases
    visible_nodes = num_particles*num_dimensions
    a_i = np.random.normal(0, 1, visible_nodes)
    b_j = np.random.normal(0, 1, hidden_nodes)
    W_ij = np.random.normal(0, 1, (visible_nodes, hidden_nodes))

    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    # Call system class in order to set new parameters
    wave = Wavefunction(visible_nodes, hidden_nodes,
                        a_i, b_j, W_ij, sigma)
    # Hamiltonian(..., weak_interaction, strong_interaction)
    hamilton = Hamiltonian(gamma, omega, num_dimensions, num_particles,
                           wave, False, False)
    met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                     num_particles, num_dimensions, wave, hamilton)

    r_vec = np.linspace(0, 4, 41)
    p_r = met.run_one_body_sampling()
    with open('/home/kari/VMC/data/obd_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["r", "density"])
        for i in range(len(r_vec)):
            writer.writerow([r_vec[i], p_r[i]/monte_carlo_cycles])


def run_blocking(monte_carlo_cycles, num_particles, num_dimensions,
                 hidden_nodes):
    """Run the sampling in metropolis to be used for blocking."""

    # Set optimal values for weights and biases
    visible_nodes = num_particles*num_dimensions
    a_i = np.random.normal(0, 1, visible_nodes)
    b_j = np.random.normal(0, 1, hidden_nodes)
    W_ij = np.random.normal(0, 1, (visible_nodes, hidden_nodes))

    sigma = 1.0
    omega = 1.0
    gamma = 1.0

    # Call system class in order to set new parameters
    wave = Wavefunction(visible_nodes, hidden_nodes,
                        a_i, b_j, W_ij, sigma)
    # Hamiltonian(..., weak_interaction, strong_interaction)
    hamilton = Hamiltonian(gamma, omega, num_dimensions, num_particles,
                           wave, False, False)
    met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                     num_particles, num_dimensions, wave, hamilton)

    # d_El, energy = met.run_metropolis()
    # Run with analytical expression for quantum force = true
    energy = met.blocking()

    with open('/home/kari/VMC/data/blocking.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["local_energy"])
        for i in range(len(energy)):
            writer.writerow([energy[i]])
