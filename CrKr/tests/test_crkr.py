from nose.tools import assert_true, assert_equal, assert_raises
from unittest import TestCase

import numpy as np
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
        U = crkr.U
        A = crkr.A
        LAMBDA = crkr.LAMBDA

        # Check if the types are correct
        assert_true(isinstance(S, np.ndarray))
        assert_true(isinstance(C, np.ndarray))
        assert_true(isinstance(D, np.ndarray))
        assert_true(isinstance(U, np.ndarray))
        assert_true(isinstance(A, numbers.Number))
        assert_true(isinstance(LAMBDA, numbers.Number))

        # Check if the matrix dimensions are consistent
        assert_equal(S.shape[0], D.shape[0])
        assert_equal(S.shape[0], C.shape[0])
        assert_equal(C.shape[0], C.shape[1]) # if square matrix
        assert_equal(S.shape[1], U.shape[0])
        assert_equal(U.shape[0], U.shape[1]) # if square matrix

    def test_init_custom(self):
        """Tests if custom variables are assigned.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.2], [0.3, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12]])
        U = np.array([[100, 101, 102], [103, 104, 105], [106, 107, 108]])
        A = 2.1234
        
        crkr = CrKr(S, C, D, U, A)

        assert_true(np.array_equal(S, crkr.S))
        assert_true(np.array_equal(C, crkr.C))
        assert_true(np.array_equal(D, crkr.D))
        assert_true(np.array_equal(U, crkr.U))
        assert_equal(A, crkr.A)

    def test_init_inconsistent_shape_c_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped C parameter.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        D = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
        U = np.array([[100, 101, 102], [103, 104, 105], [106, 107, 108]])
        A = 2.1234

        assert_raises(ValueError, CrKr(S, C, D, U, A))

    def test_init_non_square_c_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped C parameter.
        """
        S = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        C = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        D = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
        U = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [1, 2, 3, 4]])
        A = 2.1234

        assert_raises(ValueError, CrKr(S, C, D, U, A))

    def test_init_inconsistent_shape_d_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped D parameter.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.2], [0.3, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12], [14, 15, 16]])
        U = np.array([[100, 101, 102], [103, 104, 105], [106, 107, 108]])
        A = 2.1234
        
        assert_raises(ValueError, CrKr(S, C, D, U, A))

    def test_init_inconsistent_shape_u_matrix(self):
        """Tests if an exception is raised, for wrongly-shaped U parameter.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.2], [0.3, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12]])
        U = np.array([[100, 101], [103, 104]])
        A = 2.1234
        
        assert_raises(ValueError, CrKr(S, C, D, U, A))

    def test_init_non_square_u_matrix(self):
        """Tests if an exception is raised, for a non-square U matrix parameter.
        """
        S = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[0.1, 0.2], [0.3, 0.4]])
        D = np.array([[7, 8, 9], [10, 11, 12], [14, 15, 16]])
        U = np.array([[100, 101], [103, 104], [106, 107]])
        A = 2.1234
        
        assert_raises(ValueError, CrKr(S, C, D, U, A))

    def test_compute_k(self):
        """Tests if k is correctly computed.
        """
        assert_true(False)
        
    def test_compute_K(self):
        """Tests if K is correctly computed.
        """
        assert_true(False)

    def test_delta_mean(self):
        """Tests if delta mean is correctly computed.
        """
        assert_true(False)

    def test_delta_variance(self):
        """Tests if delta variance is correctly computed.
        """
        assert_true(False)

    def test_delta_estimate(self):
        assert_true(False)

    def test_delta_estimate_diff_shape_state(self):
        """Tests if an exception is raised, for wrongly-shaped state
        """
        assert_true(False)

    def test_update_matrices(self):
        """Tests if the matrices are correctly updated.
        """
        assert_true(False)
