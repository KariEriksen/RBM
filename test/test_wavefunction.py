import pytest
import numpy as np
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from Wavefunction.wavefunction import Wavefunction # noqa: 401


def test_wavefunction_2d_2p():

    num_p = 2
    num_d = 2
    M = num_p*num_d
    N = 3
    sigma = 1.0

    for _ in range(50):
        a = np.random.uniform(-2, 2, M)
        b = np.random.uniform(-2, 2, N)
        W = np.random.uniform(-2, 2, (M, N))
        wave = Wavefunction(M, N, a, b, W, sigma)
        positions = np.random.uniform(-1, 1, M)
        sum1 = np.subtract(positions, a)
        sum1 = np.sum(sum1*sum1)/(2*sigma*sigma)
        term1 = math.exp(-sum1)
        sum2 = 0.0
        prod = 1.0
        for j in range(N):
            sum2 += np.dot(positions, W[:, j])/(sigma*sigma)
            prod *= (1 + math.exp(b[j] + sum2))

        wave_function = term1*prod

        assert wave_function == pytest.approx(wave.wavefunction(positions),
                                              abs=1e-10)


def test_wavefunction_3d_2p():

    assert 0 == 0


def test_gradients_wavefunction_2d_2p():

    num_p = 2
    num_d = 2
    M = num_p*num_d
    N = 3
    sigma = 1.0

    for _ in range(50):
        a = np.random.uniform(-2, 2, M)
        b = np.random.uniform(-2, 2, N)
        W = np.random.uniform(-2, 2, (M, N))
        wave = Wavefunction(M, N, a, b, W, sigma)
        positions = np.random.uniform(-1, 1, M)
        sum1 = np.subtract(positions, a)
        sum1 = -np.sum(sum1*sum1)/(2*sigma*sigma)
        sum2 = 0.0
        sum3 = 0.0
        for j in range(N):
            sum2 += np.dot(positions, W[:, j])/(sigma*sigma)
            sum3 += (1 + math.exp(b[j] + sum2))
            for i in range(M):
                sum3 += W[i, j]

        wave_function = term1*prod

        # assert wave_function == pytest.approx(wave.wavefunction(positions),
        #                                       abs=1e-10)
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
