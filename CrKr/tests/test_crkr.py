from nose.tools import assert_true, assert_equal, assert_raises
from unittest import TestCase

import numpy as np
import numpy.linalg as npla
import numbers

from CrKr.crkr import CrKr

class TestCrKr(TestCase):
    def setUp(self):
        self.S_2x3 = np.array([[1, 2, 3], [4, 5, 6]])
        self.C_2x2 = np.array([[1, 0], [0, 5]])
        self.D_2x3 = np.array([[1, 2, 3], [4, 5, 6]])
        self.ridge_factor_05 = 0.5
        self.sigma_05 = 0.5
        self.a_1 = 1

        self.C_2x3 = np.array([[0.1, 0.0], [0.3, 0.4], [0.5, 0.6]])
        self.C_3x3 = np.array([[0.1, 0.0, 0.0], 
                               [0.0, 0.4, 0.0], 
                               [0.0, 0.0, 0.6]])

        self.D_3x3 = np.array([[7, 8, 9], [10, 11, 12], [14, 15, 16]])

    def test_init_default(self):
        """Tests if default variables exists and their data types.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)
        ridge_factor = crkr.ridge_factor
        sigma = crkr.sigma
        a = crkr.a

        assert_true(isinstance(ridge_factor, numbers.Number))
        assert_true(isinstance(sigma, numbers.Number))
        assert_true(isinstance(a, numbers.Number))

    def test_init_custom(self):
        """Tests if custom variables are assigned.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3, 
                    self.ridge_factor_05, self.sigma_05, self.a_1)

        assert_true(np.array_equal(self.S_2x3, crkr.S))
        assert_true(np.array_equal(self.C_2x2, crkr.C))
        assert_true(np.array_equal(self.D_2x3, crkr.D))
        assert_equal(self.ridge_factor_05, crkr.ridge_factor)
        assert_equal(self.sigma_05, crkr.sigma)
        assert_equal(self.a_1, crkr.a)

    def test_init_inconsistent_shape_c_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped C parameter.
        """
        assert_raises(ValueError, CrKr, self.S_2x3, self.C_3x3, self.D_2x3, 
                      self.ridge_factor_05, self.sigma_05, self.a_1)

    def test_init_non_square_c_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped C parameter.
        """
        assert_raises(ValueError, CrKr, self.S_2x3, self.C_2x3, self.D_2x3, 
                      self.ridge_factor_05, self.sigma_05, self.a_1)

    def test_init_inconsistent_shape_d_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped D parameter.
        """
        assert_raises(ValueError, CrKr, self.S_2x3, self.C_2x2, self.D_3x3, 
                      self.ridge_factor_05, self.sigma_05, self.a_1)

    def test_gaussian_kernel(self):
        """Tests if the gaussian kernel is correctly computed.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3, 
                    self.ridge_factor_05, self.sigma_05, self.a_1)
        
        s1 = np.array([[1, 2, 3]])
        s2 = np.array([[4, 5, 6]])

        expected_gk = np.exp(-(self.a_1 * np.power(npla.norm(s1 - s2), 2) / 
                               (2 * (self.sigma_05 ** 2))))

        assert_equal(expected_gk, crkr._gaussian_kernel(s1, s2))

    def test_gaussian_kernel_same_state(self):
        """Tests if the gaussian kernel is 1 for two equal states.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)
        s = np.array([[1, 2, 3]])

        assert_equal(1, crkr._gaussian_kernel(s, s))

    def test_compute_k(self):
        """Tests if k is correctly computed.
        """
        S = self.S_2x3
        new_s = np.array([[0, 1, 2]])
        exponent = (-self.a_1 * np.power(npla.norm(new_s - S, axis=1), 2) /
                    (2 * (self.sigma_05 ** 2)))
        expected_k = np.exp(exponent)
        expected_k = np.array([expected_k]).T

        crkr = CrKr(S, self.C_2x2, self.D_2x3, 
                    self.ridge_factor_05, self.sigma_05, self.a_1)
        result_k = crkr._compute_k(new_s)
        
        assert_equal(expected_k.shape, result_k.shape)
        assert_true(np.allclose(expected_k, result_k))

    def test_compute_K(self):
        """Tests if K is correctly computed.
        """
        S = self.S_2x3

        expected_K = np.zeros((S.shape[0], S.shape[0]))
        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[0]):
                s1 = np.array([S[i, :]])
                s2 = np.array([S[j, :]])
                exponent = (-self.a_1 * np.power(npla.norm(s1 - s2), 2) / 
                            (2 * (self.sigma_05 ** 2)))
                expected_K[i, j] = np.exp(exponent)

        crkr = CrKr(S, self.C_2x2, self.D_2x3, 
                    self.ridge_factor_05, self.sigma_05, self.a_1)

        assert_true(np.allclose(expected_K, crkr._compute_K()))

    def test_delta_mean(self):
        """Tests if delta mean is correctly computed.
        """
        S = self.S_2x3
        C = self.C_2x2
        D = self.D_2x3
        ridge_factor = self.ridge_factor_05
        sigma = self.sigma_05
        a = self.a_1

        crkr = CrKr(S, C, D, ridge_factor, sigma, a)
        
        new_s = np.array([[0, 0, 0]])
        k = crkr._compute_k(new_s)
        K = crkr._compute_K()
        expected_dm = np.dot(k.T, 
                             np.dot(np.linalg.inv(K + ridge_factor * C), D))
        
        assert_true(np.allclose(expected_dm, crkr._delta_mean(k, K)))

    def test_delta_variance(self):
        """Tests if delta variance is correctly computed.
        """
        S = self.S_2x3
        C = self.C_2x2
        D = self.D_2x3
        ridge_factor = self.ridge_factor_05
        sigma = self.sigma_05
        a = self.a_1

        crkr = CrKr(S, C, D, ridge_factor, sigma, a)

        new_s = np.array([[1, 1, 1]])
        k = crkr._compute_k(new_s)
        K = crkr._compute_K()

        expected_dv = (a + 
                       ridge_factor - 
                       np.dot(k.T, np.dot(npla.inv(K + ridge_factor * C), k)))

        assert_true(np.allclose(expected_dv, crkr._delta_variance(k, K)))

    def test_delta_estimate(self):
        """Test if a 2-dimensional numpy array is returned.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)

        new_s = np.array([[1, 2, 5]])

        delta_estimate = crkr.delta_estimate(new_s)

        assert_equal(delta_estimate.shape, (1, self.S_2x3.shape[1]))

    def test_delta_estimate_diff_shape_state(self):
        """Tests if an exception is raised, for wrongly-shaped state
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)

        new_s = np.array([[1, 2, 3, 4]])

        assert_raises(ValueError, crkr.delta_estimate, new_s)

    def test_update_matrices(self):
        """Tests if the matrices are correctly updated.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)

        new_s = np.array([[7, 8, 9]])
        new_reward = 20
        new_d = np.array([[13, 14, 15]])

        expected_S = np.vstack((self.S_2x3, new_s))
        expected_C_diag = np.append(np.diagonal(self.C_2x2), 1.0 / new_reward)
        expected_C = np.diag(expected_C_diag)
        expected_D = np.vstack((self.D_2x3, new_d))

        crkr.update_matrices(new_s, new_reward, new_d)

        assert_true(np.allclose(expected_S, crkr.S))
        assert_true(np.allclose(expected_C, crkr.C))
        assert_true(np.allclose(expected_D, crkr.D))

    def test_update_matrices_diff_shape_state(self):
        """Tests if an exception is raises, for wrongly-shaped state.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)

        new_s = np.array([[7, 8, 9, 10]])
        new_reward = 20
        new_d = np.array([[13, 14, 15]])

        assert_raises(ValueError, 
                      crkr.update_matrices, 
                      new_s, 
                      new_reward, 
                      new_d)

    def test_update_matrices_zero_reward(self):
        """Tests if an exception is raises, for reward of 0.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)

        new_s = np.array([[7, 8, 9]])
        new_reward = 0
        new_d = np.array([[13, 14, 15]])

        assert_raises(ValueError, 
                      crkr.update_matrices, 
                      new_s, 
                      new_reward, 
                      new_d)

    def test_update_matrices_diff_shape_delta(self):
        """Tests if an exception is raises, for reward of 0.
        """
        crkr = CrKr(self.S_2x3, self.C_2x2, self.D_2x3)

        new_s = np.array([[7, 8, 9]])
        new_reward = 0.2
        new_d = np.array([[13, 14, 15, 16]])

        assert_raises(ValueError, 
                      crkr.update_matrices, 
                      new_s, 
                      new_reward, 
                      new_d)
