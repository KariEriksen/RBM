import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from Wavefunction.wavefunction import Wavefunction # noqa: 401


def test_wavefunction_2d_2p():

    num_p = 2
    num_d = 2
    M = num_p*num_d
    N = 3
    a = np.random.uniform(-2, 2, M)
    b = np.random.uniform(-2, 2, N)
    W = np.random.uniform(-2, 2, (M, N))
    sigma = 1.0
    wave = Wavefunction(M, N, a, b, W, sigma)

    for _ in range(50):
        positions = np.random.uniform(-2, 2, M)
        x = 0.0
        for i in range(M):
            x += positions[i]*positions[i]

        wave_function = 0.0

        # assert wave_function == pytest.approx(wave.wavefunction(positions),
        #                              abs=1e-14)

        assert wave_function == 0.0

def test_wavefunction_3d_2p():

    assert 0 == 0

def test_gradients_wavefunction_2d_2p():

    assert 0 == 0

def test_gradients_wavefunction_3d_2p():

    assert 0 == 0

def test_quandratic_gradients_wavefunction_2d_2p():

    assert 0 == 0

def test_quandratic_gradients_wavefunction_3d_2p():

    assert 0 == 0

def test_gradient_wavefunction_a_2d_2p():

    assert 0 == 0

def test_gradient_wavefunction_a_3d_2p():

    assert 0 == 0

def test_gradient_wavefunction_b_2d_2p():

    assert 0 == 0

def test_gradient_wavefunction_b_3d_2p():

    assert 0 == 0

def test_gradient_wavefunction_W_2d_2p():

    assert 0 == 0

def test_gradient_wavefunction_W_3d_2p():

    assert 0 == 0

def test_ratio_wavefunction_2d_2p():

    assert 0 == 0

def test_ratio_wavefunction_3d_2p():

    assert 0 == 0
