""" This script implement quadratic programming with constrains
"""
print __doc__

import numpy as np
from scipy.optimize import fmin_slsqp

# Define objective function that need to be minimized
def func(x):
    tmp_x = x.reshape(len(x),1)
    tmp_mtr = np.array([[1,0],[0,2]])
    tmp_r = x.dot(tmp_mtr).dot(tmp_x)
    return tmp_r[0]

# Define the derivative of the objective function
def func_deriv(x):
    tmp_mtr = np.array([[1,0],[0,2]])
    tmp_x = x.reshape(len(x),1)
    tmp_r = tmp_mtr.dot(tmp_x)
    return tmp_r[:,0]

# write constrains
def cons_eq(x):
    tmp = x.dot(np.ones((len(x),1)))-1
    return np.array([tmp])

def cons_eq_deriv(x):
    return np.ones((1,len(x)))

res = fmin_slsqp(func,[0.5,0.5],f_eqcons=cons_eq,bounds=[(0,1),(0,1)],\
        fprime=func_deriv,fprime_eqcons=cons_eq_deriv,iter=100,acc=1e-06,\
        iprint=1,disp=None)

