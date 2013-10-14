import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def rosen(x):
    """ The Rosenbrock function"""
    return sum(100.*(x[1:]-x[:-1]**2.0)**2.0+(1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
#res = minimize(rosen, x0, method='BFGS',jac=rosen_der,options={'xtol':1e-8,'disp':True})

x,f,d = fmin_l_bfgs_b(rosen,x0,fprime=rosen_der)


#print res.x
