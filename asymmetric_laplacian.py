import scipy.integrate as integrate
import numpy as np
from math import inf

'''
Given privacy parameters epsilon, global_sensitivity, and scale parameter k, return the function for the probability density function 
for asymmetric laplace distribution.
'''
def asymmetric_laplace(epsilon, global_sensitivity, k):
    lam = epsilon / global_sensitivity
    def f(x):
        if x < 0:
            return lam / (k + 1 / k) * np.exp(lam * x / k)
        else :
            return lam / (k + 1 / k) * np.exp(-lam * x * k)
    return f


'''
Given epsilon, global_sensitivity, and k, return the L1 cost for the asymmetric laplacian mechanism.
'''
def asymmetric_laplace_L1_eval(epsilon, global_sensitivity, k):
    return integrate.quad(lambda x: abs(x) * asymmetric_laplace(epsilon, global_sensitivity, k)(x), -inf, inf)[0]


'''
Given epsilon, global_sensitivity, and k, return the L2 cost for the asymmetric laplacian mechanism.
'''
def asymmetric_laplace_L2_eval(epsilon, global_sensitivity, k):
    return integrate.quad(lambda x: x**2 * asymmetric_laplace(epsilon, global_sensitivity,k)(x), -inf, inf)[0]


# Example for getting resutls for asymmetric laplacian
def asymmetric_laplacian_example():
	epsilons = list(np.arange(1e-4, 1e-3, 1e-4)) # change here for different epsilon values
	global_sensitivity = 1 # changee here for different global sensitivity values
	ks = list(np.arange(2, 3, 1)) # change here for different scale parameter values
	print(['epsilon', 'global sensitivity', 'scale parameter(k)', 'L_1 cost', 'L_2 cost'])
	for epsilon in epsilons:
		for k in ks: 	
			print([epsilon, global_sensitivity, k, asymmetric_laplace_L1_eval(epsilon, global_sensitivity, k), asymmetric_laplace_L2_eval(epsilon, global_sensitivity, k)])

