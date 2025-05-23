import sympy as sp
import numpy as np
from scipy.special import hankel1
from numpy.linalg import norm
from numpy.polynomial.legendre import leggauss


def assem_duque(coords, elems, k):
    """Assembly matrices for the BEM problem

    Parameters
    ----------
    coords : ndarray, float
        Coordinates for the nodes.
    elems : ndarray, int
        Connectivity for the elements.

    Returns
    -------
    Gmat : ndarray, float
        Influence matrix for the flow.
    Hmat : ndarray, float
        Influence matrix for primary variable.
    """
    nelems = elems.shape[0]

    Gmat = np.zeros((nelems, nelems))
    Hmat = np.zeros((nelems, nelems))

    for ev_cont, elem1 in enumerate(elems):
    # j-element varia dentro de este loop

        for col_cont, elem2 in enumerate(elems):
        # p_i varia dentro de este loop

            pt_col = np.mean(coords[elem2], axis=0)

            if ev_cont == col_cont: # i == j

               pass

            else: # i != j

                Gij = G_ij_nonsingular(elem1, coords, pt_col, k)
                Hij = H_ij_nonsingular(elem1, coords, pt_col, k)
            
                Gmat[ev_cont, col_cont] = Gij
                Hmat[ev_cont, col_cont] = Hij

    return Gmat, Hmat

def G_ij_nonsingular(elem_j, coords, p_i, k, n_gauss = 8):
    
    ## Parameterization
    xi = sp.symbols('xi')

    EP_j = coords[elem_j[0]] # End Point j
    EP_j_1 = coords[elem_j[1]] # End Point j+1
    L_j = norm(EP_j_1 - EP_j) # Length of the j-th element
    
    x_xi = (EP_j[0] * (1 - xi) + EP_j_1[0] * (1 + xi)) / 2
    y_xi = (EP_j[1] * (1 - xi) + EP_j_1[1] * (1 + xi)) / 2

    X_i = p_i[0] #X_i
    Y_i = p_i[1] #Y_i

    r_x_xi = x_xi - X_i
    r_y_xi = y_xi - Y_i
    r_magnitude_symbolic = sp.sqrt(r_x_xi**2 + r_y_xi**2)

    r_magnitude_callable = sp.lambdify(xi, r_magnitude_symbolic, modules='numpy')

    ## Gauss Integration
    xi_vals, w_vals = leggauss(n_gauss)
    r_magnitudes = r_magnitude_callable(xi_vals)

    integrand = hankel1(0, k * r_magnitudes)
    integral = np.dot(w_vals, integrand)
    result = ( (1j * L_j) / (8) ) * integral

    return result

def H_ij_nonsingular(elem_j, coords, p_i, k, n_gauss = 8):
    
    ## Parameterization
    xi = sp.symbols('xi')

    EP_j = coords[elem_j[0]] # End Point j
    EP_j_1 = coords[elem_j[1]] # End Point j+1
    L_j = norm(EP_j_1 - EP_j) # Length of the j-th element.
    
    x_xi = (EP_j[0] * (1 - xi) + EP_j_1[0] * (1 + xi)) / 2
    y_xi = (EP_j[1] * (1 - xi) + EP_j_1[1] * (1 + xi)) / 2

    X_i = p_i[0] #X_i
    Y_i = p_i[1] #Y_i

    r_x_xi = x_xi - X_i
    r_y_xi = y_xi - Y_i
    r_magnitude_symbolic = sp.sqrt(r_x_xi**2 + r_y_xi**2)

    E_j_vect = EP_j_1 - EP_j # The j-th element as a vector.
    E_j_vect_unitary = E_j_vect / norm(E_j_vect) # Unit vector of the j-th element.
    normal_unitary = np.array([-E_j_vect_unitary[1], E_j_vect_unitary[0]]) # Normal vector of the j-th element. 

    dot_product = r_x_xi * normal_unitary [0] + r_y_xi * normal_unitary[1]
    cos_phi_symbolic = dot_product / r_magnitude_symbolic
    integrand_symbolic = hankel1(1, k * r_magnitude_symbolic) * cos_phi_symbolic
    integrand_callable = sp.lambdify(xi, integrand_symbolic, modules='numpy')

    ## Gauss Integration
    xi_vals, w_vals = leggauss(n_gauss)
    integrand_vals = integrand_callable(xi_vals)
    integral = np.dot(w_vals, integrand_vals)
    result = -( (1j * k * L_j) / (8) ) * integral
    
    return result




###################################################################################