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


def Gij_coefficient_function(chi, elem_j, coords, pt_i, k, complex_part="Re"):

    # i element is in which the constant point is considered
    # j element is the one over which the intrgration is carried out

    EP_j = coords[elem_j[0]] # End Point j
    EP_j_1 = coords[elem_j[1]] # End Point j+1
    L_j = norm(EP_j_1 - EP_j) # Length of the j-th element
    
    x_i, y_i = pt_i

    x_chi = (EP_j[0] * (1 - chi) + EP_j_1[0] * (1 + chi)) / 2
    y_chi = (EP_j[1] * (1 - chi) + EP_j_1[1] * (1 + chi)) / 2
    
    r_chi = ((x_chi-x_i)**2+(y_chi-y_i)**2)**0.5

    if complex_part == "Re" or complex_part =="re":
        return (1j/8 * hankel1(0,k*r_chi) * L_j).real
    elif complex_part == "Im" or complex_part == "im":
        return (1j/8 * hankel1(0,k*r_chi) * L_j).imag
    else:
        raise Exception("Wrong input specification.")

    
def Hij_coefficient_function(chi, elem_j, coords, pt_i, k, complex_part="Re"):

    EP_j = coords[elem_j[0]] # End Point j
    EP_j_1 = coords[elem_j[1]] # End Point j+1
    
    j_dist = EP_j_1 - EP_j
    L_j = norm(EP_j_1 - EP_j) # Length of the j-th element
    j_dir = j_dist/L_j

    x_i, y_i = pt_i

    x_chi = (EP_j[0] * (1 - chi) + EP_j_1[0] * (1 + chi)) / 2
    y_chi = (EP_j[1] * (1 - chi) + EP_j_1[1] * (1 + chi)) / 2
    
    r_chi = ((x_chi-x_i)**2+(y_chi-y_i)**2)**0.5
    r_chi_vec = [(x_chi-x_i)/r_chi ,  (y_chi-y_i)/r_chi]

    # Find the normal dir vector
    rotmat = np.array([[j_dir[1], -j_dir[0]],
                       [j_dir[0], j_dir[1]]])
    normal = rotmat @ j_dir
    
    normal = normal/norm(normal)

    if complex_part == "Re" or complex_part =="re":
        return (-1j/8 * k * L_j * hankel1(1,k*r_chi) * dot(r_chi_vec, normal) ).real
    elif complex_part == "Im" or complex_part == "im":
        return (-1j/8 * k * L_j * hankel1(1,k*r_chi) * dot(r_chi_vec, normal) ).imag
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

            pt_col = np.mean(coords[elem2], axis=0)
            
            if ev_cont == col_cont:

                wrapped_GijRe = lambda chi: Gij_coefficient_function(chi, elem1, coords, pt_col, k,"re")
                wrapped_GijIm = lambda chi: Gij_coefficient_function(chi, elem1, coords, pt_col, k,"im")
                gquadRe,_ = quad(wrapped_GijRe,0,1)
                gquadIm,_ = quad(wrapped_GijIm,0,1)
                Gmat[ev_cont, ev_cont] = 2 * (gquadRe+1j*gquadIm)
                
                if domain_type == "external":
                    Hmat[ev_cont, col_cont] = -0.5
                elif domain_type == "internal":
                    Hmat[ev_cont, col_cont] = 0.5
                else:
                    sys.exit("Invalid domain_type, please enter a valid type and re-run the code.")
            else:

                wrapped_GijRe = lambda chi: Gij_coefficient_function(chi, elem1, coords, pt_col, k,"re")
                wrapped_GijIm = lambda chi: Gij_coefficient_function(chi, elem1, coords, pt_col, k,"im")
                gquadRe,_ = quad(wrapped_GijRe,-1,1)
                gquadIm,_ = quad(wrapped_GijIm,-1,1)
                Gmat[ev_cont, col_cont] = gquadRe+1j*gquadIm

                wrapped_HijRe = lambda chi: Hij_coefficient_function(chi, elem1, coords, pt_col, k,"re")
                wrapped_HijIm = lambda chi: Hij_coefficient_function(chi, elem1, coords, pt_col, k,"im")
                hquadRe,_ = quad(wrapped_HijRe,-1,1)
                hquadIm,_ = quad(wrapped_HijIm,-1,1)
                Hmat[ev_cont, col_cont] = hquadRe+1j*hquadIm

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
def eval_sol(ev_coords, coords, elems, u_boundary, q_boundary, k, domain_type):
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

    if domain_type != "external" and domain_type != "internal":
        sys.exit("Invalid domain_type, please enter a valid type and re-run the code.")
        
    npts = ev_coords.shape[0]
    solution = np.zeros((npts),dtype=complex)
    for pt in range(npts):
        for ev_cont, elem in enumerate(elems):        
            pt_col = ev_coords[pt]

            wrapped_GijRe = lambda chi: Gij_coefficient_function(chi, elem, coords, pt_col, k,"re")
            wrapped_GijIm = lambda chi: Gij_coefficient_function(chi, elem, coords, pt_col, k,"im")
            gquadRe,_ = quad(wrapped_GijRe,-1,1)
            gquadIm,_ = quad(wrapped_GijIm,-1,1)
            G = gquadRe+1j*gquadIm

            wrapped_HijRe = lambda chi: Hij_coefficient_function(chi, elem, coords, pt_col, k,"re")
            wrapped_HijIm = lambda chi: Hij_coefficient_function(chi, elem, coords, pt_col, k,"im")
            hquadRe,_ = quad(wrapped_HijRe,-1,1)
            hquadIm,_ = quad(wrapped_HijIm,-1,1)
            H = hquadRe+1j*hquadIm      
    
            if domain_type == "external":
                solution[pt] +=  -1 * (u_boundary[ev_cont]*H - q_boundary[ev_cont]*G)
            elif domain_type == "internal":
                solution[pt] +=  u_boundary[ev_cont]*H - q_boundary[ev_cont]*G
            else:
                raise Exception("Wrong input specification.")
     
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



#########################################################################################################

# Generate square .geo file

def create_square_geo(lower_grid_size, upper_grid_size, sizeInner, ngrid_pts):
    geom_file = open("simple_square_bound.geo", "w", encoding="utf-8")
    
    geom_file.write(" /* \n " +
                    ".geo file for simple square boundary, \n"+
                    "Introduccion al Metodo de Frontera Universidad EAFIT 2025-1 \n"+
                    "by: Daniel Ospina Pajoy, Sebastián Duque Lotero & Mateo Tabares. \n */ "+
                    "\n \n \n"+
                    "// Inner Scatterer Element Size"+
                    f"\n sizeRoI = {sizeInner}; \n \n")
    
    geom_file.write("// Points \n") 
    geom_file.write(f"Point(1) = {{ {lower_grid_size}, {lower_grid_size}, 0.0, {sizeInner} }}; \n")
    geom_file.write(f"Point(2) = {{ {upper_grid_size}, {lower_grid_size}, 0.0, {sizeInner} }}; \n")
    geom_file.write(f"Point(3) = {{ {upper_grid_size}, {upper_grid_size}, 0.0, {sizeInner} }}; \n")
    geom_file.write(f"Point(4) = {{ {lower_grid_size}, {upper_grid_size}, 0.0, {sizeInner} }}; \n")
    geom_file.write("\n \n") 
    
    geom_file.write("// Lines \n") 
    geom_file.write("Line(1) = { 1, 2 } ; \n")
    geom_file.write("Line(2) = { 2, 3 } ; \n")
    geom_file.write("Line(3) = { 3, 4 } ; \n")
    geom_file.write("Line(4) = { 4, 1 } ; \n")
    geom_file.write("\n \n") 
    
    geom_file.write("// Surfaces \n"+
                    "Curve Loop(1) = { 1: 4 }; \n"+
                   "Plane Surface(1) = {1}; \n") 
    geom_file.write("\n \n") 
    
    geom_file.write("// Physical groups \n"+
                    "Physical Curve(1) = { 1,2,3,4 }; \n"+
                   "Physical Surface(2) = {1}; \n") 
    geom_file.write("\n \n") 
    
    
    ndiv = ngrid_pts
    geom_file.write("// Mesh parameters \n"+
                   f"ndiv = {ndiv}; \n"+
                   "Transfinite Curve { 1,2,3, 4 } = ndiv Using Progression 1; \n"+
                   "Transfinite Surface {1}; \n") 
        
    geom_file.close()


def create_circle_geo(radius, ngrid_pts):

    geom_file = open("simple_circle_bound.geo", "w", encoding="utf-8")
    
    geom_file.write(" /* \n " +
                    ".geo file for simple circle boundary, \n"+
                    "Introduccion al Metodo de Frontera Universidad EAFIT 2025-1 \n"+
                    "by: Daniel Ospina Pajoy, Sebastián Duque Lotero & Mateo Tabares. \n */ "+
                    "\n \n \n")
    
    geom_file.write("// Points \n") 
    geom_file.write(f"Point(1) = {{ 0.0 , 0.0 , 0.0, 0.5 }}; \n")
    geom_file.write(f"Point(2) = {{ {-radius}*Cos(Pi/4), {-radius}*Cos(Pi/4), 0.0, 0.5 }}; \n")
    geom_file.write(f"Point(3) = {{ {radius}*Cos(Pi/4), {-radius}*Cos(Pi/4), 0.0, 0.5 }}; \n")
    geom_file.write(f"Point(4) = {{ {radius}*Cos(Pi/4), {radius}*Cos(Pi/4), 0.0, 0.5 }}; \n")
    geom_file.write(f"Point(5) = {{ {-radius}*Cos(Pi/4), {radius}*Cos(Pi/4), 0.0, 0.5 }}; \n")
    geom_file.write("\n \n") 
    
    geom_file.write("// Lines \n") 
    geom_file.write("Circle(1) = { 2, 1, 3 } ; \n")
    geom_file.write("Circle(2) = { 3, 1, 4 } ; \n")
    geom_file.write("Circle(3) = { 4, 1, 5 } ; \n")
    geom_file.write("Circle(4) = { 5, 1, 2 } ; \n")
    geom_file.write("\n \n") 
    
    geom_file.write("// Surfaces \n"+
                    "Curve Loop(1) = { 1,2,3,4 }; \n"+
                   "Plane Surface(1) = {1}; \n") 
    geom_file.write("\n \n") 
    
    geom_file.write("// Physical groups \n"+
                    "Physical Curve(1) = { 1,2,3,4 }; \n"+
                   "Physical Surface(2) = {1}; \n") 
    geom_file.write("\n \n") 
    
    
    ndiv = ngrid_pts
    geom_file.write("// Mesh parameters \n"+
                   f"ndiv = {ndiv}; \n"+
                   "Transfinite Curve { 1,2,3, 4 } = ndiv Using Progression 1; \n"+
                   "Transfinite Surface {1}; \n") 
        
    geom_file.close()

