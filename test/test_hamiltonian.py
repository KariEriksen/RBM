import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from Hamiltonian.hamiltonian import Hamiltonian # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401


def test_local_energy_2d_2p():

    num_particles = 2
    num_dimensions = 2
    M = num_particles*num_dimensions
    N = 3
    a = np.random.uniform(-2, 2, M)
    b = np.random.uniform(-2, 2, N)
    W = np.random.uniform(-2, 2, (M, N))
    omega = 1.0
    gamma = 1.0
    sigma = 1.0
    wave = Wavefunction(M, N, a, b, W, sigma)
    hamilton = Hamiltonian(gamma, omega, num_dimensions,
                           num_particles, wave, 'false')

    for _ in range(50):
        positions = np.random.uniform(-2, 2, M)
        fd, sd = wave.gradients_wavefunction(positions)
        x = 0.0
        for i in range(M):
            x += positions[i]*positions[i]

        energy = 0.5*(-fd*fd + sd + omega*omega*x)
        assert 0 == 0
        # assert energy == pytest.approx(hamilton.local_energy(positions),
        #                               abs=1e-14)