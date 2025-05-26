import sympy as sp
import numpy as np
from scipy.special import hankel1
from numpy.linalg import norm
from numpy.polynomial.legendre import leggauss


def eval_sol_duque(ev_coords, coords, elems, u_boundary, q_boundary, k):
    """Evaluate the solution in a set of points

    Parameters
    ----------
    ev_coords : ndarray, float
        Coordinates of the evaluation points.
    coords : ndarray, float
        Coordinates for the nodes.
    elems : ndarray, int
        Connectivity for the elements.
    u_boundary : ndarray, float
        Primary variable in the nodes.
    q_boundary : [type]
        Flows in the nodes.
    k: float
        Wavenumber.

    Returns
    -------
    solution : ndarray, float
        Solution evaluated in the given points.
    """
    npts = ev_coords.shape[0]
    solution = np.zeros(npts)
    for pt in range(npts):
        for ev_cont, elem in enumerate(elems):        
            pt_col = ev_coords[pt]

            G = G_ij_nonsingular(elem, coords, pt_col, k)
            H = H_ij_nonsingular(elem, coords, pt_col, k)           
    
            # if problem is interior leave the -1 that multiplies, else remove it
            solution[pt] +=  -1 * (u_boundary[ev_cont]*H - q_boundary[ev_cont]*G)
    return solution
    

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

    Gmat = np.zeros((nelems, nelems),dtype=complex)
    Hmat = np.zeros((nelems, nelems),dtype=complex)

    for ev_cont, elem1 in enumerate(elems):
    # j-element varia dentro de este loop

        for col_cont, elem2 in enumerate(elems):
        # p_i varia dentro de este loop

            pt_col = np.mean(coords[elem2], axis=0)

            if ev_cont == col_cont: # i == j

               Gij = G_ij_singular(elem1, coords, pt_col, k,15)
               # if problem is interior it is +0.5, else it is -0.5
               Hij = 0.5

            else: # i != j

                Gij = G_ij_nonsingular(elem1, coords, pt_col, k,15)
                Hij = H_ij_nonsingular(elem1, coords, pt_col, k,15)
            
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

'''
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
'''

def H_ij_nonsingular(elem_j, coords, p_i, k, n_gauss = 8):
    
    ## Parameterization
    xi = sp.symbols('xi')

    EP_j = coords[elem_j[0]]    # End Point j
    EP_j_1 = coords[elem_j[1]]  # End Point j+1
    L_j = norm(EP_j_1 - EP_j)   # Length of the j-th element

    # Parametrize q(xi) on the element
    x_xi = (EP_j[0] * (1 - xi) + EP_j_1[0] * (1 + xi)) / 2
    y_xi = (EP_j[1] * (1 - xi) + EP_j_1[1] * (1 + xi)) / 2

    # Collocation point
    X_i = p_i[0]
    Y_i = p_i[1]

    # Vector from x_i to q(xi)
    r_x_xi = x_xi - X_i
    r_y_xi = y_xi - Y_i
    r_magnitude_symbolic = sp.sqrt(r_x_xi**2 + r_y_xi**2)

    # Normal vector to the element (unit)
    E_j_vect = EP_j_1 - EP_j
    E_j_vect_unitary = E_j_vect / norm(E_j_vect)
    normal_unitary = np.array([-E_j_vect_unitary[1], E_j_vect_unitary[0]])

    # Cos(phi) = dot(r, n̂) / ||r||
    dot_product = r_x_xi * normal_unitary[0] + r_y_xi * normal_unitary[1]
    cos_phi_symbolic = dot_product / r_magnitude_symbolic

    # Lambdify components separately
    r_mag_func = sp.lambdify(xi, r_magnitude_symbolic, modules='numpy')
    cos_phi_func = sp.lambdify(xi, cos_phi_symbolic, modules='numpy')

    # Define numerical integrand
    def integrand_callable(xi_vals):
        r_vals = r_mag_func(xi_vals)
        cos_vals = cos_phi_func(xi_vals)
        return hankel1(1, k * r_vals) * cos_vals

    ## Gauss Integration
    xi_vals, w_vals = leggauss(n_gauss)
    integrand_vals = integrand_callable(xi_vals)
    integral = np.dot(w_vals, integrand_vals)

    result = - (1j * k * L_j / 8) * integral
    return result



def G_ij_singular(elem_j, coords, p_i, k, n_gauss = 8):
    """
    This function was generated with ChatGPT based on the nonsingular case function.

    Regularized computation of the singular G_ii value using singularity subtraction.

    Parameters
    ----------
    elem_j : list[int]
        Indices of the two endpoints of the element (i = j).
    coords : np.ndarray
        Array of node coordinates.
    p_i : np.ndarray
        Collocation point (x_i, y_i) lying on the element.
    k : float
        Helmholtz wavenumber.
    n_gauss : int
        Number of Gauss points.

    Returns
    -------
    result : complex
        Regularized G_ii value.
    """
    ## Parameterization
    xi = sp.symbols('xi')

    EP_j = coords[elem_j[0]]     # End Point j
    EP_j_1 = coords[elem_j[1]]   # End Point j+1
    L_j = norm(EP_j_1 - EP_j)    # Element length

    x_xi = (EP_j[0] * (1 - xi) + EP_j_1[0] * (1 + xi)) / 2
    y_xi = (EP_j[1] * (1 - xi) + EP_j_1[1] * (1 + xi)) / 2

    X_i = p_i[0]  # Collocation point x
    Y_i = p_i[1]  # Collocation point y

    r_x_xi = x_xi - X_i
    r_y_xi = y_xi - Y_i
    r_magnitude_symbolic = sp.sqrt(r_x_xi**2 + r_y_xi**2)
    log_r_symbolic = sp.log(r_magnitude_symbolic)

    # Lambdify for numerical evaluation
    r_magnitude_callable = sp.lambdify(xi, r_magnitude_symbolic, modules='numpy')
    log_r_callable = sp.lambdify(xi, log_r_symbolic, modules='numpy')

    ## Gauss Integration
    xi_vals, w_vals = leggauss(n_gauss)
    r_vals = r_magnitude_callable(xi_vals)
    log_vals = log_r_callable(xi_vals)

    # Regularized integrand
    reg_integrand = hankel1(0, k * r_vals) - (2j / np.pi) * log_vals
    reg_integral = np.dot(w_vals, reg_integrand)

    # Analytic integral of log term over [-1, 1]
    analytic_integral = 2 * np.log(L_j / 2)

    # Combine both parts
    result = (1j * L_j / 8) * (reg_integral + (2j / np.pi) * analytic_integral)
    return result

###################################################################################