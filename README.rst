****
CrKr
****

Implementation in Python of the Cost Regularized Kernel Regression (CrKr) algorithm.


Dependencies
############

* Numpy 1.8.2

Example of usage
################

::
   S = np.array([[1, 2], [3, 4]])
   C = np.array([[0.2, 0], [0, 0.5]])
   D = np.array([[10, 11], [9, 8]])
   crkr = CrKr(S, C, D)
   new_state = np.array([[1, 1.5]])
   delta_estimate = crkr.delta_estimate(new_state)
   reward = your_method_to_compute_reward()
   crkr.update_matrices(new_state, reward, delta_estimate)
