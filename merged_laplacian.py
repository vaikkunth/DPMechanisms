import scipy.integrate as integrate
import numpy as np
from math import inf

'''
Given a list of break points c, and a list of privacy parameters epsilon and global_sensitivity, return the probability density 
function that merges the two distribution. Errors are raised when the lengths of the three lists do not match 
accordingly. Currently the function only handles merging less than three laplace distribution. 
'''
def merged_laplace(epsilon, global_sensitivity, c):
    # Check for list size
    if len(c) > len(epsilon) - 1:
        raise RuntimeError('Too many break points.')
    if len(c) < len(epsilon) - 1:
        raise RuntimeError('Not enough break points.')

    lam = [(global_sensitivity / epsilon[i]) for i in range(len(epsilon))]
    c.sort()

    # pdf for merging different number of laplace distributioin
    def laplace(x):
        lambda1 = lam[0]
        return 1/(2 * lambda1) * np.exp(-abs(x) / lambda1)

    def merge_two(x):
        lambda1, lambda2 = lam
        c1 = c[0]
        a1 = 1 / (2 * (lambda1 + (lambda2 * np.exp(-c1 / lambda1)) - (lambda1 * np.exp(-c1 / lambda1))))
        a2 = np.exp((c1 / lambda2) - c1 / lambda1) / (
                2 * (lambda1 + (lambda2 * np.exp(-c1 / lambda1)) - (lambda1 * np.exp(-c1 / lambda1))))
        if x < -c1:
            return a2 * np.exp(x / lambda2)

        elif x < 0:
            return a1 * np.exp(x / lambda1)

        elif x < c1:
            return a1 * np.exp(-x / lambda1)

        else:
            return a2 * np.exp(-x / lambda2)

    def merge_three(x):
        lambda1, lambda2, lambda3 = lam
        c1, c2 = c
        b1 = lambda1 * (1 - np.exp(-c1 / lambda1))
        b2 = lambda2 * (np.exp(-c1 / lambda1) - np.exp(c1 / lambda2 - c2 / lambda2 - c1 / lambda1))
        b3 = lambda3 * (np.exp(-c2 / lambda2 + c1 / lambda2 - c1 / lambda1))
        a1 = 1 / (2 * (b1 + b2 + b3))
        a2 = (np.exp(c1 / lambda2 - c1 / lambda1)) / (2 * (b1 + b2 + b3))
        a3 = (np.exp(-c2 / lambda3 - c2 / lambda2 + c1 / lambda2 - c1 / lambda1)) / (2 * (b1 + b2 + b3))
        if x < -c2:
            return a3 * np.exp(x / lambda3)
        elif x < -c1:
            return a2 * np.exp(x / lambda2)
        elif x < 0:
            return a1 * np.exp(x / lambda1)
        elif x < c1:
            return a1 * np.exp(-x / lambda1)
        elif x < c2:
            return a2 * np.exp(-x / lambda2)
        else:
            return a3 * np.exp(-x / lambda3)

    if len(epsilon) == 1:
        return laplace
    elif len(epsilon) == 2:
        return merge_two
    elif len(epsilon) == 3:
        return merge_three
    else:
        raise RuntimeError("Too many laplace distributions to merge.")


'''
Given a list of break points c, and a list of privacy parameters epsilon and global_sensitivity, return the L1 cost for the 
merged laplacian mechanism.
'''
def merged_laplace_L1_eval(epsilon, global_sensitivity, c):
    return integrate.quad(lambda x: abs(x) * merged_laplace(epsilon, global_sensitivity, c)(x), -inf, inf)[0]

'''
Given a list of break points c, and a list of privacy parameters epsilon and global_sensitivity, return the L2 cost for the 
merged laplacian mechanism.
'''
def merged_laplace_L2_eval(epsilon, global_sensitivity, c):
    return integrate.quad(lambda x: x**2 * merged_laplace(epsilon, global_sensitivity, c)(x), -inf, inf)[0]

# Example for getting resutls for merged laplacian
def merged_two_laplacian_example():
    # for regular laplacian 
    epsilons = [1e-4] # change here for different epsilon values
    global_sensitivity = 1 # changee here for different global sensitivity values
    break_points = [] # change here for different break points
    L1_cost_laplace = merged_laplace_L1_eval(epsilons, global_sensitivity, break_points)
    L2_cost_laplace = merged_laplace_L2_eval(epsilons, global_sensitivity, break_points)

    # for merging two laplacian 
    epsilons = [1e-4, 1e-5] # change here for different epsilon values
    global_sensitivity = 1 # changee here for different global sensitivity values
    break_points = [3] # change here for different break points
    L1_cost_two_laplace = merged_laplace_L1_eval(epsilons, global_sensitivity, break_points)
    L2_cost_two_laplace = merged_laplace_L2_eval(epsilons, global_sensitivity, break_points)

    # for merging three laplacian 
    epsilons = [1e-4, 1e-5, 1e-6] # change here for different epsilon values
    global_sensitivity = 1 # changee here for different global sensitivity values
    break_points = [3, 5] # change here for different break points
    L1_cost_three_laplace = merged_laplace_L1_eval(epsilons, global_sensitivity, break_points)
    L2_cost_three_laplace = merged_laplace_L2_eval(epsilons, global_sensitivity, break_points)