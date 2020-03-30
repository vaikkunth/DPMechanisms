import scipy.integrate as integrate
import numpy as np

'''
This is a helper function for generating the probability density function for truncated laplacee distribution.
This function processes A and B to be the desired value. If the input A, B value is not appropriate for the desired 
epsilon, we default to the values for symmetric A, B values. 
'''
def process(epsilon, delta, A, B):
    lam = 1 / epsilon
    A_ = lam * np.log(2 + (1 - delta) / delta * np.exp(-B / lam) - 1 / delta * np.exp(-(B - 1) / lam))
    if abs(B) < abs(A_):
        return A_, B
    B_ = -lam * np.log(2 + (1 - delta)/delta * np.exp(A/lam) - 1/delta * np.exp((B+1)/lam))
    if abs(A) < abs(B_):
        return A, B_
    A_ = delta / epsilon * np.log(1 + (np.exp(epsilon) - 1)/(2 * delta))
    B_ = - A_
    return A_, B_

'''
Given privacy parameters epsilon, delta, and scale parameters A, and B, return a function that is the probability 
density function for the truncated laplace distribution.
'''
def truncated_laplace(epsilon, delta, A, B):
    lam = 1 / epsilon
    a, b = process(epsilon, delta, A, B)
    M = 1 / (lam * (2 - np.exp(a / lam) - np.exp(-b / lam)))
    return lambda x: M * np.exp(-abs(x) / lam)

'''
Given epsilon, delta, A, B, return the L1 cost for the truncated laplacian mechanism.
'''
def truncated_laplace_L1_eval(epsilon, delta, A, B):
    return integrate.quad(lambda x: abs(x) * truncated_laplace(epsilon, delta, A, B)(x), A, B)[0]

'''
Given epsilon, delta, A, B, return the L2 cost for the truncated laplacian mechanism.
'''
def truncated_laplace_L2_eval(epsilon, delta, A, B):
    return integrate.quad(lambda x: x**2 * truncated_laplace(epsilon, delta, A, B)(x), A, B)[0]

