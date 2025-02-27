import numpy as np 
from math import factorial
import casadi as ca


####### ELLIPTIC INTEGRALS SERIES REPRESENTATION #################



def K(m,n_max): #series representation of complete elliptic Integral of the first kind
    kk = 0
    for n in range(0,n_max):
        kk += np.pi/2 * (factorial(2*n)/(2**(2*n)*factorial(n)**2))**2 *m**(n)
    return(kk)


def E(m,n_max): #series representation of complete elliptic Integral of the second kind
    ee = 0
    for n in range(0,n_max):
        ee += np.pi/2 * (factorial(2*n)/(2**(2*n)*factorial(n)**2))**2 *m**(n)/(1-2*n)
    return(ee)


def AGM(m, n_max):

    x0 = ca.sqrt(1 - m**2)
    y0 = 1

    xn = x0
    yn = y0
    sum_c = 0

    for n in range(n_max):

        # AGM recursion
        x_next = 0.5 * (xn + yn)
        y_next = ca.sqrt(xn*yn)
        
        # sum for elliptic integral of the second kind
        sum_c += 2**(n-2)*(x_next - y_next)**2

        # re-initialize
        xn = x_next
        yn = y_next

    # elliptic integral
    K = np.pi / (xn + yn)
    C = ((x0 + y0)**2/4 - sum_c)
    E = K * C

    return (K, E), C

####### INTEGRANDS IN TERMS OF ELLIPTIC INTEGRALS #################



def elliptic_integrand_series_axial(Rj,Rf,h,n_max, method = 'AGM'):
    ''''
    Rj : float
        radial distance evaluation point to filament
    Rf : float
        radius of the vortex filaments
    h: float
        axial distance evaluation point to filament
    n_max: int
        maximum value until the eliptic integrals are evaluated
    '''
    
    # define the argument of the elliptic integrals
    m = (4 * Rj * Rf) / (h**2 + (Rj + Rf)**2)

    # define the elliptic integrals
    if method == 'AGM':
        (K0, E0), C0 = AGM(m, n_max)
        numerator = 2*(-((h**2-Rf**2+Rj**2)/(h**2 + (Rf - Rj)**2)*C0 + 1) * K0)
        denominator = np.sqrt(h**2 + (Rj + Rf)**2)
 
    elif method == 'power':
        elliptic_k = K(m,n_max) #complete elliptic Integral of the first kind 
        elliptic_e = E(m,n_max) #complete elliptic integral of the second kind
        numerator = 2*(-((h**2-Rf**2+Rj**2)*elliptic_e)+(h**2+(Rf-Rj)**2)*elliptic_k)
        denominator = ((h**2 + (Rf - Rj)**2) * np.sqrt(h**2 + (Rj + Rf)**2))

    return(numerator / denominator)

def elliptic_integrand_series_radial(Rj,Rf,h,n_max):
    ''''
    Rj : float
        radial distance evaluation point to filament
    Rf : float
        radius of the vortex filaments
    h: float
        axial distance evaluation point to filament
    n_max: int
        maximum value until the eliptic integrals are evaluated
    '''

    # define the argument of the elliptic integrals
    m = (4 * Rj * Rf) / (h**2 + (Rj + Rf)**2)

    # define the elliptic integrals
    elliptic_k = K(m,n_max) #complete elliptic Integral of the first kind 
    elliptic_e = E(m,n_max) #complete elliptic integral of the second kind

    numerator = 2*h*((h**2+Rf**2+Rj**2)*elliptic_e-(h**2+(Rf-Rj)**2 * elliptic_k))
    denominator = ((h**2 + (Rf-Rj)**2)*Rj*np.sqrt(h**2+(Rf+Rj)**2))

    # compute the final expression
    return(numerator / denominator)