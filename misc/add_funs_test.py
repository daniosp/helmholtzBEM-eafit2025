"""
Boundary element method
=======================

Routines for boundary element analysis for the Helmholtz Equation

        nabla^2 phi + kappa^2 phi = 0

with mixed (Dirichlet, Neumann) boundary conditions.

Elements approximate the potential and flow as constants.

"""
import numpy as np
from numpy import log, sin, cos, arctan2, pi, mean, dot
from numpy.linalg import norm, solve
from scipy.special import roots_legendre, hankel1
from scipy.integrate import quad
import matplotlib.pyplot as plt
import meshio
import sys


#%% Pre-process
def read_geo_gmsh(fname, dir_groups, neu_groups):
    """Read the geometry from a Gmsh file with physical groups

    Parameters
    ----------
    fname : str
        Path to the mesh file.
    dir_groups : list
        List with the number of the physical groups associated
        with Dirichlet boundary conditions.
    neu_groups : list
        List with the number of the physical groups associated
        with Dirichlet boundary conditions.

    Returns
    -------
    mesh : meshio Mesh object
        Mesh object.
    coords : ndarray, float
        Coordinates for the endpoints of the elements in the
        boundary.
    elems : ndarray, int
        Connectivity for the elements.
    x_m : ndarray, float
        Horizontal component of the midpoint of the elements.
    y_m : ndarray, float
        Vertical component of the midpoint of the elements.
    id_dir : list
        Identifiers for elements with Dirichlet boundary conditions.
    id_neu : list
        Identifiers for elements with Neumann boundary conditions.
    """
    mesh = meshio.read(fname)
    elems_dir = np.vstack([mesh.cells[k].data for k in dir_groups])
    if neu_groups is None:
        elems_neu = np.array([])
        elems = elems_dir.copy()
    else:
        elems_neu = np.vstack([mesh.cells[k].data for k in neu_groups])
        elems = np.vstack((elems_dir, elems_neu))
    bound_nodes = list(set(elems.flatten()))
    coords = mesh.points[bound_nodes, :2]
    x_m, y_m = 0.5*(coords[elems[:, 0]] + coords[elems[:, 1]]).T
    id_dir = range(elems_dir.shape[0])
    id_neu = range(elems_dir.shape[0],
                   elems_dir.shape[0] + elems_neu.shape[0])
    return mesh, coords, elems, x_m, y_m, id_dir, id_neu


#%% Process
# ESTE DEBE REVISARSE PARA HACERLO COMPLIANT CON HEMLHOLTZ
def influence_coeff(elem, coords, pt_col):
    """Compute influence coefficients

    Parameters
    ----------
    elems : ndarray, int
        Connectivity for the elements. [1x2]
    coords : ndarray, float
        Coordinates for the nodes. [n_nodes x 2]
    pt_col : ndarray
        Coordinates of the colocation point. [1x2]

    Returns
    -------
    G_coeff : float
        Influence coefficient for flows.
    H_coeff : float
        Influence coefficient for primary variable.
    """
    dcos = coords[elem[1]] - coords[elem[0]]
    dcos = dcos / norm(dcos)
    rotmat = np.array([[dcos[1], -dcos[0]],
                       [dcos[0], dcos[1]]])
    r_A = rotmat.dot(coords[elem[0]] - pt_col)
    r_B = rotmat.dot(coords[elem[1]] - pt_col)
    theta_A = arctan2(r_A[1], r_A[0])
    theta_B = arctan2(r_B[1], r_B[0])
    if norm(r_A) <= 1e-6:
        G_coeff = r_B[1]*(log(norm(r_B)) - 1) + theta_B*r_B[0]
    elif norm(r_B) <= 1e-6:
        G_coeff = -(r_A[1]*(log(norm(r_A)) - 1) + theta_A*r_A[0])
    else:
        G_coeff = r_B[1]*(log(norm(r_B)) - 1) + theta_B*r_B[0] -\
                  (r_A[1]*(log(norm(r_A)) - 1) + theta_A*r_A[0])
    H_coeff = theta_B - theta_A
    return -G_coeff/(2*pi), H_coeff/(2*pi)


def Gij_coefficient_function(chi, coords_i, coords_j, k, complex_part="Re"):

    # i element is in which the constant point is considered
    # j element is the one over which the intrgration is carried out

    # Function to integrate (Helmholtz+Coord change)
    dist_vec = coords_j[1]-coords_j[0]
    ele_len = norm(dist_vec) # Element length

    x_m, y_m = [mean([coords_i[0][0],coords_i[1][0]]), mean([coords_i[0][1],coords_i[1][1]])]

    x_chi = (coords_j[1][0]+coords_j[0][0])/2 + (coords_j[1][0]-coords_j[0][0])/2*chi
    y_chi = (coords_j[1][1]+coords_j[0][1])/2 + (coords_j[1][1]-coords_j[0][1])/2*chi
    
    r_chi = ((x_chi-x_m)**2+(y_chi-y_m)**2)**0.5

    if complex_part == "Re" or complex_part =="re":
        return (1j/4 * hankel1(0,k*r_chi) * ele_len/2).real
    elif complex_part == "Im" or complex_part == "im":
        return (1j/4 * hankel1(0,k*r_chi) * ele_len/2).imag
    else:
        raise Exception("Wrong input specification.")

    

    
def Hij_coefficient_function(chi,  coords_i, coords_j, k, complex_part = "Re"):

    # Function to integrate (Helmholtz+Coord change)
    dist_vec = coords_j[1]-coords_j[0]
    ele_len = norm(dist_vec) # Element length

    x_m, y_m = [mean([coords_i[0][0],coords_i[1][0]]), mean([coords_i[0][1],coords_i[1][1]])]

    x_chi = (coords_j[1][0]+coords_j[0][0])/2 + (coords_j[1][0]-coords_j[0][0])/2*chi
    y_chi = (coords_j[1][1]+coords_j[0][1])/2 + (coords_j[1][1]-coords_j[0][1])/2*chi
    

    r_chi = ((x_chi-x_m)**2+(y_chi-y_m)**2)**0.5
    r_chi_vec = [(x_chi-x_m)/r_chi ,  (y_chi-y_m)/r_chi]

    dcos = dist_vec / norm(dist_vec)
    rotmat = np.array([[dcos[1], -dcos[0]],
                       [dcos[0], dcos[1]]])
    normal = rotmat @ dcos

    if complex_part == "Re" or complex_part =="re":
        return (-1j/4 * hankel1(1,k*r_chi) * dot(r_chi_vec, normal)* ele_len/2).real
    elif complex_part == "Im" or complex_part == "im":
        return (-1j/4 * hankel1(1,k*r_chi) * dot(r_chi_vec, normal)* ele_len/2).imag
    else:
        raise Exception("Wrong input specification.")



def assem(coords,elems,k,domain_type):
    """Assembly matrices for the BEM Helmholtz problem

    Parameters
    ----------
    coords : ndarray, float
        Coordinates for the nodes.
    elems : ndarray, int
        Connectivity for the elements.
    k : float
        Wavenumber
    domain_type: string, "internal" or "external"
        Sets the type of domain problem for Helmholtz (Changes the sign of Hmat when i == j)

    Returns
    -------
    Gmat : ndarray, float
        Influence matrix for the flow.
    Hmat : ndarray, float
        Influence matrix for primary variable.
    """

    if domain_type != "external" and domain_type != "internal":
        sys.exit("Invalid domain_type, please enter a valid type and re-run the code.")

    nelems = elems.shape[0]
    Gmat = np.zeros((nelems, nelems),dtype=complex)
    Hmat = np.zeros((nelems, nelems),dtype=complex)
    for ev_cont, elem1 in enumerate(elems):
        for col_cont, elem2 in enumerate(elems):
            if ev_cont == col_cont:

                wrapped_GijRe = lambda chi: Gij_coefficient_function(chi,coords[elem1],coords[elem2],k,"re")
                wrapped_GijIm = lambda chi: Gij_coefficient_function(chi,coords[elem1],coords[elem2],k,"im")
                gquadRe,_ = quad(wrapped_GijRe,0,1)
                gquadIm,_ = quad(wrapped_GijIm,0,1)
                Gmat[ev_cont, ev_cont] = 2 * (gquadRe+1j*gquadIm)
                
                if domain_type == "external":
                    Hmat[ev_cont, ev_cont] = -0.5
                elif domain_type == "internal":
                    Hmat[ev_cont, ev_cont] = 0.5
                else:
                    sys.exit("Invalid domain_type, please enter a valid type and re-run the code.")
            else:

                wrapped_GijRe = lambda chi: Gij_coefficient_function(chi,coords[elem1],coords[elem2],k,"re")
                wrapped_GijIm = lambda chi: Gij_coefficient_function(chi,coords[elem1],coords[elem2],k,"im")
                gquadRe,_ = quad(wrapped_GijRe,-1,1)
                gquadIm,_ = quad(wrapped_GijIm,-1,1)
                Gmat[ev_cont, ev_cont] = gquadRe+1j*gquadIm

                wrapped_HijRe = lambda chi: Hij_coefficient_function(chi,coords[elem1],coords[elem2],k,"re")
                wrapped_HijIm = lambda chi: Hij_coefficient_function(chi,coords[elem1],coords[elem2],k,"im")
                hquadRe,_ = quad(wrapped_HijRe,-1,1)
                hquadIm,_ = quad(wrapped_HijIm,-1,1)
                Hmat[ev_cont, ev_cont] = hquadRe+1j*hquadIm

    return Gmat, Hmat
###################################################################################
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

            pt_col = mean(coords[elem2], axis=0)

            if ev_cont == col_cont: # i == j

                L = norm(coords[elem1[1]] - coords[elem1[0]])
                Gmat[ev_cont, ev_cont] = - L/(2*pi)*(log(L/2) - 1)
                Hmat[ev_cont, ev_cont] = - 0.5

            else: # i != j

                Gij = G_ij_nonsingular(elem1, coords, pt_col, k)
                Hij = H_ij_nonsingular(elem1, coords, pt_col, k)
            
                Gmat[ev_cont, col_cont] = Gij
                Hmat[ev_cont, col_cont] = Hij

    return Gmat, Hmat

import sympy as sp
import numpy as np
from scipy.special import hankel1
from numpy.linalg import norm
from numpy.polynomial.legendre import leggauss

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
def rearrange_mats(Hmat, Gmat, id_dir, id_neu):
    """Rearrange BEM matrices to account for boundary conditions

    Parameters
    ----------
    Hmat : ndarray, float
        Influence coefficient matrix accompanying potential.
    Gmat : ndarray, float
        Influence coefficient matrix accompanying flow.
    id_dir : list
        Identifiers for elements with Dirichlet boundary conditions.
    id_neu : list
        Identifiers for elements with Neumann boundary conditions.

    Returns
    -------
    A : ndarray, float
        Matrix accompanying unknown values (left-hand side).
    B : ndarray, float
        Matrix accompanying known values (right-hand side).
    """
    A = np.zeros_like(Hmat)
    B = np.zeros_like(Hmat)
    A[np.ix_(id_dir, id_dir)] = Gmat[np.ix_(id_dir, id_dir)]
    A[np.ix_(id_dir, id_neu)] = -Hmat[np.ix_(id_dir, id_neu)]
    A[np.ix_(id_neu, id_dir)] = Gmat[np.ix_(id_neu, id_dir)]
    A[np.ix_(id_neu, id_neu)] = -Hmat[np.ix_(id_neu, id_neu)]
    B[np.ix_(id_dir, id_dir)] = Hmat[np.ix_(id_dir, id_dir)]
    B[np.ix_(id_dir, id_neu)] = -Gmat[np.ix_(id_dir, id_neu)]
    B[np.ix_(id_neu, id_dir)] = Hmat[np.ix_(id_neu, id_dir)]
    B[np.ix_(id_neu, id_neu)] = -Gmat[np.ix_(id_neu, id_neu)]
    return A, B


def create_rhs(x_m, y_m, u_bc, q_bc, id_dir, id_neu):
    """Create vector with known values for potential and flow

    Parameters
    ----------
    x_m : ndarray, float
        Horizontal component of the midpoint of the elements.
    y_m : ndarray, float
        Vertical component of the midpoint of the elements.
    u_bc : callable or float
        Value for prescribed Dirichlet boundary conditions. If it
        is a callable it evaluates it on `(x_m[id_dir], y_m[id_dir])`.
        If it is a float it assigns a constant value.
    q_bc : callable or float
        Value for prescribed Neumann boundary conditions. If it
        is a callable it evaluates it on `(x_m[id_dir], y_m[id_dir])`.
        If it is a float it assigns a constant value.
    id_dir : list
        Identifiers for elements with Dirichlet boundary conditions.
    id_neu : list
        Identifiers for elements with Neumann boundary conditions.

    Returns
    -------
    rhs : ndarray, float
        Vector with known values for potential and flow
    """
    rhs = np.zeros(x_m.shape[0])
    if callable(u_bc):
        rhs[id_dir] = u_bc(x_m[id_dir], y_m[id_dir])
    else:
        rhs[id_dir] = u_bc
    if callable(q_bc):
        rhs[id_neu] = q_bc(x_m[id_neu], y_m[id_neu])
    else:
        rhs[id_neu] = q_bc
    return rhs


#%% Post-process
def eval_sol(ev_coords, coords, elems, u_boundary, q_boundary):
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
    q_boundary : ndarray, float
        Flows in the nodes.

    Returns
    -------
    solution : ndarray, float
        Solution evaluated in the given points.
    """
    npts = ev_coords.shape[0]
    solution = np.zeros(npts)
    for k in range(npts):
        for ev_cont, elem in enumerate(elems):        
            pt_col = ev_coords[k]
            G, H = influence_coeff(elem, coords, pt_col)
            solution[k] += u_boundary[ev_cont]*H - q_boundary[ev_cont]*G
    return solution


def rearrange_sol(sol, rhs, id_dir, id_neu):
    """[summary]

    Parameters
    ----------
    sol : [type]
        [description]
    rhs : [type]
        [description]
    id_dir : [type]
        [description]
    id_neu : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    u_bound = np.zeros_like(sol)
    q_bound = np.zeros_like(sol)
    u_bound[id_dir] = rhs[id_dir]
    u_bound[id_neu] = sol[id_neu]
    q_bound[id_dir] = sol[id_dir]
    q_bound[id_neu] = rhs[id_neu]
    return u_bound, q_bound
