import numpy as np
import numpy.linalg as npla
import numbers

class CrKr:
    """Implementation of the CrKr algorithm.

    Implementation of the Cost Regularized Kernel Regression algorithm,
    proposed in 'Reinforcement Learning to Adjust Robot Movements 
    to New Situations', by Kober.

    :author: Rui Silva

    Example of usage:
    -----------------
    >>> S = np.array([[1, 2], [3, 4]])
    >>> C = np.array([[0.2, 0], [0, 0.5]])
    >>> D = np.array([[10, 11], [9, 8]])
    >>> crkr = CrKr(S, C, D)
    >>> new_state = np.array([[1, 1.5]])
    >>> delta_estimate = crkr.delta_estimate(new_state)
    >>> reward = ...
    >>> crkr.update_matrices(new_state, reward, delta_estimate)
    """

    def __init__(self, S, C, D, ridge_factor=0.5, sigma=1, a=1):
        """Returns an instance of CrKr.

        Parameters:
        ----------
        S : N x m np.array
          Matrix with the states of the training examples.
        C : N x N np.array
          Diagonal matrix with the costs (reward^-1) of each training example.
        D : N x n np.array
          Matrix with the deltas of the training examples.
        ridge_factor : float
          Ridge factor of the kernel regression.
        sigma : float
          Width of the gaussian kernel.
        a : float
          Normalization constant of the gaussian kernel.
        
        Returns:
        --------
        Instance of CrKr
        """
        self._check_S_matrix(S)
        self._check_C_matrix(C, S)
        self._check_D_matrix(D, S)
        self._check_ridge_factor(ridge_factor)
        self._check_sigma(sigma)
        self._check_a(a)
        
        self.S = S
        self.C = C
        self.D = D
        self.ridge_factor = ridge_factor
        self.sigma = sigma
        self.a = a
        self.K = self._compute_K()

    def delta_estimate(self, s):
        """Returns the delta estimate, for a given state.

        Parameters:
        ----------
        s : 1 x m np.array
          Array with the new observed state.
        
        Returns:
        --------
        delta_estimate : 1 x n np.array
          Delta estimate for the given state.

        Raises:
        -------
        ValueError : Exception
          When the given state is not a 2-dimensional np.array, or
          when the number of features (columns) is different than the
          S matrix.
        """
        self._check_state(s, self.S)

        k = self._compute_k(s)
        delta_mean = self._delta_mean(k, self.K)
        delta_variance = self._delta_variance(k, self.K)
        delta_variance_matrix = delta_variance * np.eye(self.D.shape[1])

        dm = np.random.multivariate_normal(delta_mean[0, :], 
                                           delta_variance_matrix)
        
        return np.array([dm])
        
    def update_matrices(self, s, reward, d):
        """Updates the kernel matrices.

        Parameters:
        ----------
        s : 1 x m np.array
          Array with the new observed state.
        reward : float
          Reward given for the execution of the delta estimate.
        d : 1 x n np.array
          Array with the estimated delta.

        Raises:
        -------
        ValueError : Exception
          - When the given state is not a 2-dimensional np.array, or
            when the number of features (columns) is different than the
            S matrix.
          - When the given delta is not a 2-dimenbsional np.array, or
            when the number of features (columns) is different than the
            D matrix.
          - When the reward is 0.
        """
        self._check_state(s, self.S)
        self._check_reward(reward)
        self._check_delta(d, self.D)
        
        self.S = np.vstack((self.S, s))
        self.D = np.vstack((self.D, d))

        C_diagonal = np.hstack((np.diagonal(self.C), 1.0 / reward))
        self.C = np.diag(C_diagonal)

        self.K = self._compute_K()

    def _gaussian_kernel(self, s1, s2):
        """Returns the gaussian kernel between two given arrays.

        Parameters:
        ----------
        s1 : a x b np.array
          The first array.
        s2 : c x b np.array
          The second array.

        Note that it supports the gaussian kernel between 1 state,
        and multiple other states at the same time.
        For that, one of the arrays should have 1 row.

        Returns:
        -------
        gaussian_kernel : 1 x min(a, c) np.array
          The gaussian kernel between the two arrays.
        """
        gk = np.exp(-self.a * np.power(npla.norm(s1 - s2, axis=1), 2) / 
                    (2 * (self.sigma ** 2)))
        return gk

    def _compute_k(self, s):
        """Returns k, the gaussian kernel between s and all training examples.

        Parameters:
        ----------
        s : 1 x m np.array
          Array with the new observed state.

        Returns:
        -------
        k : 1 x n np.array
          Array with the gaussian kernel between s and all training examples.
          Measures the distance between the observed state, and the training
          examples.
        """
        k = self._gaussian_kernel(s, self.S)
        return np.array([k]).T

    def _compute_K(self):
        """Returns K, array that contains the pairwise distance between all 
           training points

        Returns:
        -------
        K : N x N np.array
          Array with the pairwise distance between all training points.
        """
        num_trainings = self.S.shape[0]
        
        K = np.zeros((num_trainings, num_trainings))
        for i in range(0, num_trainings):
            for j in range(i, num_trainings):
                s1 = np.array([self.S[i, :]])
                s2 = np.array([self.S[j, :]])
                K[i, j] = K[j, i] = self._gaussian_kernel(s1, s2)
                
        return K

    def _delta_mean(self, k, K):
        """Returns the delta mean.

        Parameters:
        ----------
        k : 1 x n np.array
          Array with the gaussian kernel between s and all training examples.
          Measures the distance between the observed state, and the training
          examples.
        K : N x N np.array
          Array with the pairwise distance between all training points.

        Returns:
        --------
        delta_mean : 1 x n np.array
          The estimated delta mean.
        """
        dm = np.dot(k.T, 
                    np.dot(npla.inv(K + self.ridge_factor * self.C), self.D))
        return dm

    def _delta_variance(self, k, K):
        """Returns the delta variance.

        Parameters:
        ----------
        k : 1 x n np.array
          Array with the gaussian kernel between s and all training examples.
          Measures the distance between the observed state, and the training
          examples.
        K : N x N np.array
          Array with the pairwise distance between all training points.

        Returns:
        -------
        delta_variance : 1 x 1 np.array
          The estimate delta_variance
        
        """
        dv = (self.a + 
              self.ridge_factor - 
              np.dot(k.T, np.dot(npla.inv(K + self.ridge_factor * self.C), k)))
        return dv

    def _check_S_matrix(self, S):
        """Verifies if the S matrix respects the necessary constraints.

        Parameters:
        ----------
        S : N x m np.array
          S matrix to test.

        Raises:
        ------
        ValueError : Exception
          When the S matrix is not a 2-dimensional np.array.
        """
        if S.ndim != 2:
            raise ValueError('S must be a 2-dimensional matrix')

    def _check_C_matrix(self, C, S):
        """Verifies if the C matrix respects the necessary constraints.

        Parameters:
        ----------
        C : N x N np.array
          C matrix to test.

        Raises:
        -------
        ValueError : Exception
          When the C matrix:
            - Is not a 2-dimensional np.array
            - Does not have the same number of lines as S.
            - Is not a square matrix.
        """
        if C.ndim != 2:
            raise ValueError('C must be a 2-dimensional matrix.')
        if C.shape[0] != S.shape[0]:
            raise ValueError('C must have the same number of lines as S.')
        if C.shape[0] != C.shape[1]:
            raise ValueError('C must be a square matrix.')

    def _check_D_matrix(self, D, S):
        """Verifies if the D matrix respects the necessary constraints.

        Parameters:
        ----------
        D : N x m np.array
          D matrix to test.

        Raises:
        ValueError : Exception
          When the D matrix:
            - Is not a 2-dimensional np.array.
            - Does not have the same number of lines as S.
        """
        if S.ndim != 2:
            raise ValueError('D must be a 2-dimensional matrix.')
        if D.shape[0] != S.shape[0]:
            raise ValueError('D must have the same number of lines as S.')

    def _check_ridge_factor(self, ridge_factor):
        """Verifies if the ridge factor respects the necessary constraints.

        Parameters:
        ----------
        ridge_factor : number
          The ridge factor to test.

        Raises:
        ------
        ValueError : Exception
          When the ridge factor is not a number.
        """
        if not isinstance(ridge_factor, numbers.Number):
            raise ValueError('Ridge Factor must be a numberic value.')
        
    def _check_sigma(self, sigma):
        """Verifies if sigma respects the necessary constraints.

        Parameters:
        ----------
        sigma : number
          The sigma to test.

        Raises:
        ------
        ValueError : Exception
          When sigma is not a number.
        """
        if not isinstance(sigma, numbers.Number):
            raise ValueError('Sigma must be a numberic value.')

    def _check_a(self, a):
        """Verifies if a respects the necessary constraints.

        Parameters:
        ----------
        a : number
          a parameter to test.

        Raises:
        ------
        ValueError : Exception
          When a is not a number.
        """
        if not isinstance(a, numbers.Number):
            raise ValueError('a must be a numberic value.')

    def _check_state(self, s, S):
        """Verifies is the new observed state respects necessary constraints.

        Parameters:
        ----------
        s : np.array
          State to test

        Raises:
        -------
        ValueError : Exception
          When the state:
            - Is not a 2-dimensional np.array
            - Does not have the same number of features (columns) as S
        """
        if s.ndim != 2 or s.shape[1] != S.shape[1]:
            exception_msg = 'State should be a 2-dimensional numpy array ' + \
                            'with the same number of columns as S: ' + \
                            str(S.shape[1])
            raise ValueError(exception_msg)

    def _check_reward(self, reward):
        """Verifies if the observed reward respects the necessary constraints.

        Parameters:
        ----------
        r : float
          Reward to test.

        Raises:
        ValueError : Exception
          When the reward is equal to 0.
        """
        if reward == 0.0:
            raise ValueError('Reward must be different than 0.')

    def _check_delta(self, d, D):
        """Verifies is the delta respects necessary constraints.

        Parameters:
        ----------
        d : np.array
          Delta to test

        Raises:
        -------
        ValueError : Exception
          When the delta:
            - Is not a 2-dimensional np.array
            - Does not have the same number of features (columns) as D
        """
        if d.ndim != 2 or d.shape[1] != D.shape[1]:
            exception_msg = 'Delta should be a 2-dimensional numpy array with ' + \
                            'the same number of columns as D: ' + \
                            str(D.shape[1])
            raise ValueError(exception_msg)
