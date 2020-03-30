import scipy.integrate as integrate
import numpy as np

'''
Given privacy parameters epsilon, delta, and scale parameter k, return the function for the probability density function 
for asymmetric laplace distribution.
'''
def asymmetric_laplace(epsilon, delta, k):
    lam = epsilon / delta
    def f(x):
        if x < 0:
            return lam / (k + 1 / k) * np.exp(lam * x / k)
        else :
            return lam / (k + 1 / k) * np.exp(-lam * x * k)
    return f


'''
Given epsilon, delta, and k, return the L1 cost for the asymmetric laplacian mechanism.
'''
def truncated_laplace_L1_eval(epsilon, delta, k):
    return integrate.quad(lambda x: abs(x) * asymmetric_laplace(epsilon, delta, k)(x))[0]


'''
Given epsilon, delta, and k, return the L2 cost for the asymmetric laplacian mechanism.
'''
def truncated_laplace_L2_eval(epsilon, delta, k):
    return integrate.quad(lambda x: x**2 * asymmetric_laplace(epsilon, delta,k)(x))[0]
