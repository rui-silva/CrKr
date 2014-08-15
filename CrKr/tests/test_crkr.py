from nose.tools import assert_true, assert_equal, assert_raises
from unittest import TestCase

import numpy as np
import numpy.linalg as npla
import numbers

from CrKr.crkr import CrKr

class TestCrKr(TestCase):
    def test_init_default(self):
        """Tests if default variables exists, and if are consistent.
        """
        crkr = Crkr()
        S = crkr.S
        C = crkr.C
        D = crkr.D
        LAMBDA = crkr.LAMBDA
        SIGMA = crkr.SIGMA
        A = crkr.A

        # Check if the types are correct
        assert_true(isinstance(S, np.ndarray))
        assert_true(isinstance(C, np.ndarray))
        assert_true(isinstance(D, np.ndarray))
        assert_true(isinstance(LAMBDA, numbers.Number))
        assert_true(isinstance(SIGMA, numbers.Number))
        assert_true(isinstance(A, numbers.Number))

        # Check if the matrix dimensions are consistent
        assert_equal(S.shape[0], D.shape[0])
        assert_equal(S.shape[0], C.shape[0])
        assert_equal(C.shape[0], C.shape[1]) # if square matrix

    def test_init_custom(self):
        """Tests if custom variables are assigned.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.0], [0.0, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12]])
        LAMBDA = 0.5
        SIGMA = 0.3
        A = 2.1234

        crkr = CrKr(S, C, D, LAMBDA, SIGMA, A)

        assert_true(np.array_equal(S, crkr.S))
        assert_true(np.array_equal(C, crkr.C))
        assert_true(np.array_equal(D, crkr.D))
        assert_equal(LAMBDA, crkr.LAMBDA)
        assert_equal(SIGMA, crkr.SIGMA)
        assert_equal(A, crkr.A)

    def test_init_inconsistent_shape_c_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped C parameter.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.0], [0.3, 0.4], [0.5, 0.6]])
        D = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
        LAMBDA = 0.5
        SIGMA = 0.2
        A = 2.1234

        assert_raises(ValueError, CrKr(S, C, D, LAMBDA, SIGMA, A))

    def test_init_non_square_c_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped C parameter.
        """
        S = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        C = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        D = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
        LAMBDA = 0.5
        SIGMA = 0.3
        A = 2.1234

        assert_raises(ValueError, CrKr(S, C, D, LAMBDA, SIGMA, A))

    def test_init_inconsistent_shape_d_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped D parameter.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.0], [0.0, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12], [14, 15, 16]])
        LAMBDA = 0.5
        SIGMA = 0.3
        A = 2.1234
        
        assert_raises(ValueError, CrKr(S, C, D, LAMBDA, SIGMA, A))

    def test_gaussian_kernel(self):
        """Tests if the gaussian kernel is correctly computed.
        """
        SIGMA = 0.1
        A = 0.5
        crkr = CrKr(SIGMA=SIGMA, A=A)
        
        s1 = np.array([[1, 2, 3]])
        s2 = np.array([[4, 5, 6]])

        expected_gk = A * npla.norm(s1 - s2) / (2 * (SIGMA ** 2))

        assert_equal(expected_gk, crkr.gaussian_kernel(s1, s2))

    def test_compute_k(self):
        """Tests if k is correctly computed.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.0], [0.0, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12]])
        LAMBDA = 0.5
        SIGMA = 0.4
        A = 2.1234

        new_s = np.array([[0, 1, 2]])
        expected_k = A * npla.norm(new_s - S, axis=1) / (2 * (SIGMA ** 2))

        crkr = CrKr(S, C, D, LAMBDA, SIGMA, A)

        assert_true(np.allclose(expected_k, crkr._compute_k(new_s)))
        
    def test_compute_K(self):
        """Tests if K is correctly computed.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.0], [0.0, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12]])
        LAMBDA = 0.5
        SIGMA = 0.4
        A = 2.1234

        expected_K = np.zeros((S.shape[0], S.shape[0]))
        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[0]):
                s1 = np.array([S[i, :]])
                s2 = np.array([S[j, :]])
                expected_K[i, j] = A * npla.norm(s1 - s2) / (2 * (SIGMA ** 2))

        crkr = CrKr(S, C, D, LAMBDA, SIGMA, A)

        assert_true(np.allclose(expected_K, crkr._compute_K()))

    def test_delta_mean(self):
        """Tests if delta mean is correctly computed.
        """
        S = np.array([[1, 1], [2, 2]])
        C = np.array([[1, 0], [0, 2]])
        D = np.array([[3, 3], [3, 3]])
        LAMBDA = 0.5
        SIGMA = 1
        A = 1

        crkr = CrKr(S, C, D, LAMBDA, SIGMA, A)
        
        new_s = np.array([[0, 0]])
        k = crkr.compute_k(new_s)
        K = crkr.compute_K()
        expected_dm = np.dot(k.T, np.dot(np.linalg.inv(K + LAMBDA * C), D))
        
        assert_true(np.allclose(expected_dm, crkr._delta_mean(k, K)))

    def test_delta_variance(self):
        """Tests if delta variance is correctly computed.
        """
        S = np.array([[1, 1], [2, 2]])
        C = np.array([[1, 0], [0, 2]])
        D = np.array([[3, 3], [3, 3]])
        LAMBDA = 0.5
        SIGMA = 1
        A = 1

        crkr = CrKr(S, C, D, LAMBDA, SIGMA, A)

        new_s = np.array([[1, 1]])
        k = crkr.compute_k(new_s)
        K = crkr.compute_K()

        expected_dv = (A + 
                       LAMBDA - 
                       np.dot(k.T, np.dot(npla.inv(K + LAMBDA * C), k)))

        assert_true(np.allclose(expected_dv, crkr._delta_variance(k, K)))

    def test_delta_estimate_diff_shape_state(self):
        """Tests if an exception is raised, for wrongly-shaped state
        """
        S = np.array([[1, 1], [2, 2]])
        C = np.array([[1, 0], [0, 2]])
        D = np.array([[3, 3], [3, 3]])
        LAMBDA = 0.5
        SIGMA = 1
        A = 1

        crkr = CrKr(S, C, D, LAMBDA, SIGMA, A)

        new_s = np.array([[1, 2, 3, 4]])

        assert_raises(ValueError, crkr.delta_estimate(new_s))

    def test_update_matrices(self):
        """Tests if the matrices are correctly updated.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.0], [0.0, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12]])

        crkr = CrKr(S, C, D)

        new_s = np.array([[7, 8, 9]])
        new_reward = 20
        new_d = np.array([[13, 14, 15]])

        expected_S = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_C = np.array([[0.1, 0, 0], [0, 0.4, 0], [0, 0, 0.05]])
        expected_D = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

        crkr.update_matrices(new_s, new_reward, new_d)

        assert_true(np.allclose(expected_S, crkr.S))
        assert_true(np.allclose(expected_C, crkr.C))
        assert_true(np.allclose(expected_D, crkr.D))
