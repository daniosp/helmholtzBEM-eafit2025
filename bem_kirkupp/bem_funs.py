# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
import numpy as np
from BoundaryData import *
from Geometry import *

class HelmholtzSolver(object):
    def __init__(self, aVertex = None, aElement = None, c = 344.0, density = 1.205):
        assert not (aVertex is None), "Cannot construct InteriorHelmholtzProblem2D without valid vertex array."
        self.aVertex = aVertex
        assert not (aElement is None), "Cannot construct InteriorHelmholtzProblem2D without valid element array."
        self.aElement = aElement
        self.c       = c
        self.density = density

        '''
        # compute the centers of the discrete elements (depending on kind, i.e line segment or triangle).
        # The solver computes the velocity potential at at these center points. 
        if (self.aElement.shape[1] ==  2):
            self.aCenters = 0.5 * (self.aVertex[self.aElement[:, 0]] + self.aVertex[aElement[:, 1]])
        elif (self.aElement.shape[1] == 3):
            self.aCenters = 1.0/3.0 * (self.aVertex[self.aElement[:, 0]] +\
                                       self.aVertex[self.aElement[:, 1]] +\
                                       self.aVertex[self.aElement[:, 2]])
        '''

    def __repr__(self):
        result = "HelmholtzSolover("
        result += "  aVertex = " + repr(self.aVertex) + ", "
        result += "  aElement = " + repr(self.aElement) + ", "
        result += "  c = " + repr(self.c) + ", "
        result += "  rho = " + repr(self.rho) + ")"
        return result

    def solveExteriorBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        mu = mu or (1j / (k + 1))
        assert boundaryCondition.f.size == self.aElement.shape[0]
        A, B = self.computeBoundaryMatricesExterior(k, mu)
        c = np.empty(self.aElement.shape[0], dtype=complex)
        for i in range(self.aElement.shape[0]):
            # Note, the only difference between the interior solver and this
            # one is the sign of the assignment below.
            c[i] = -(boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i])

        phi, v = self.SolveLinearEquation(B, A, c,
                                          boundaryCondition.alpha,
                                          boundaryCondition.beta,
                                          boundaryCondition.f)
        return BoundarySolution(self, k, phi, v)

    def solveInteriorBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        mu = mu or (1j / (k + 1))
        assert boundaryCondition.f.size == self.aElement.shape[0]
        A, B = self.computeBoundaryMatricesInterior(k, mu)
        c = np.empty(self.aElement.shape[0], dtype=complex)
        for i in range(self.aElement.shape[0]):
            # Note, the only difference between the interior solver and this
            # one is the sign of the assignment below.
            c[i] = boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i]

        phi, v = self.SolveLinearEquation(B, A, c,
                                          boundaryCondition.alpha,
                                          boundaryCondition.beta,
                                          boundaryCondition.f)
        return BoundarySolution(self, k, phi, v)

    
    @classmethod
    def SolveLinearEquation(cls, Ai, Bi, ci, alpha, beta, f):
        A = np.copy(Ai)
        B = np.copy(Bi)
        c = np.copy(ci)

        x = np.empty(c.size, dtype=np.complex)
        y = np.empty(c.size, dtype=np.complex)

        gamma = np.linalg.norm(B, np.inf) / np.linalg.norm(A, np.inf)
        swapXY = np.empty(c.size, dtype=bool)
        for i in range(c.size):
            if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
                swapXY[i] = False
            else:
                swapXY[i] = True

        for i in range(c.size):
            if swapXY[i]:
                for j in range(alpha.size):
                    c[j] += f[i] * B[j,i] / beta[i]
                    B[j, i] = -alpha[i] * B[j, i] / beta[i]
            else:
                for j in range(alpha.size):
                    c[j] -= f[i] * A[j, i] / alpha[i]
                    A[j, i] = -beta[i] * A[j, i] / alpha[i]

        A -= B
        y = np.linalg.solve(A, c)

        for i in range(c.size):
            if swapXY[i]:
                x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
            else:
                x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

        for i in range(c.size):
            if swapXY[i]:
                temp = x[i]
                x[i] = y[i]
                y[i] = temp

        return x, y


def printInteriorSolution(solution, aPhiInterior):
    print "\nSound pressure at the sample points\n"
    print "index          Potential                    Pressure               Magnitude         Phase\n"
    for i in range(aPhiInterior.size):
        pressure = soundPressure(solution.k, aPhiInterior[i], c=solution.parent.c, density=solution.parent.density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase)

def printInteriorSolution(solution, aPhiInterior):
    print "\nSound pressure at the sample points\n"
    print "index          Potential                    Pressure               Magnitude         Phase\n"
    for i in range(aPhiInterior.size):
        pressure = soundPressure(solution.k, aPhiInterior[i], c=solution.parent.c, density=solution.parent.density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase)


# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
import numpy as np
from AcousticProperties import *

class BoundaryCondition(object):
    def __init__(self, size):
        self.alpha = np.empty(size, dtype = np.complex64)
        self.beta  = np.empty(size, dtype = np.complex64)
        self.f     = np.empty(size, dtype = np.complex64)


# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
from HelmholtzSolver import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals2D_C import *
else:
    from HelmholtzIntegrals2D import *        
        
from Geometry import *
from AcousticProperties import *
import numpy as np


class HelmholtzSolver2D(HelmholtzSolver):
    def __init__(self, *args, **kwargs):
        super(HelmholtzSolver2D, self).__init__(*args, **kwargs)
        self.aCenters = 0.5 * (self.aVertex[self.aElement[:, 0]] + self.aVertex[self.aElement[:, 1]])
        # lenght of the boundary elements (for the 3d shapes this is replaced by aArea
        self.aLength = np.linalg.norm(self.aVertex[self.aElement[:, 0]] - self.aVertex[self.aElement[:, 1]])

    def computeBoundaryMatrices(self, k, mu, orientation):
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.aElement.shape[0]):
            pa = self.aVertex[self.aElement[i, 0]]
            pb = self.aVertex[self.aElement[i, 1]]
            pab = pb - pa
            center = 0.5 * (pa + pb)
            centerNormal = Normal2D(pa, pb)
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = ComputeL(k, center, qa, qb, i==j)
                elementM  = ComputeM(k, center, qa, qb, i==j)
                elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
                elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
                
                A[i, j] = elementL + mu * elementMt
                B[i, j] = elementM + mu * elementN

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i,i] -= 0.5 * mu
                B[i,i] += 0.5
            elif orientation == 'exterior':
                A[i,i] += 0.5 * mu
                B[i,i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)
                
        return A, B

    def computeBoundaryMatricesInterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'interior')
    
    def computeBoundaryMatricesExterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'exterior')

    
    def solveSamples(self, solution, aIncidentPhi, aSamples, orientation):
        assert aIncidentPhi.shape == aSamples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aIncidentPhi.size):
            p  = aSamples[i]
            sum = aIncidentPhi[i]
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = ComputeL(solution.k, p, qa, qb, False)
                elementM  = ComputeM(solution.k, p, qa, qb, False)
                if orientation == 'interior':
                    sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
                elif orientation == 'exterior':
                    sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            aResult[i] = sum
        return aResult

    def solveInterior(self, solution, aIncidentInteriorPhi, aInteriorPoints):
        return self.solveSamples(solution, aIncidentInteriorPhi, aInteriorPoints, 'interior')
    
    def solveExterior(self, solution, aIncidentExteriorPhi, aExteriorPoints):
        return self.solveSamples(solution, aIncidentExteriorPhi, aExteriorPoints, 'exterior')


# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1
from Geometry import Normal2D

def ComplexQuad(func, start, end):                                                 
    samples = np.array([[0.980144928249, 5.061426814519E-02],                           
                        [0.898333238707, 0.111190517227],                               
                        [0.762766204958, 0.156853322939],                               
                        [0.591717321248, 0.181341891689],                               
                        [0.408282678752, 0.181341891689],                               
                        [0.237233795042, 0.156853322939],                               
                        [0.101666761293, 0.111190517227],                               
                        [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)    
    
    vec = end - start                                                                                                  
    sum = 0.0                                                                                                          
    for n in range(samples.shape[0]):                                                                                  
        x = start + samples[n, 0] * vec                                                                                
        sum += samples[n, 1] * func(x)                                                                                 
    return sum * norm(vec)                                                                                         
    
def ComputeL(k, p, qa, qb, pOnElement):                                                                       
    qab = qb - qa                                                                                                  
    if pOnElement:                                                                                                 
        if k == 0.0:                                                                                               
            ra = norm(p - qa)                                                                                      
            rb = norm(p - qb)                                                                                      
            re = norm(qab)                                                                                         
            return 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))                                        
        else:                                                                                                      
            def func(x):                                                                                           
                R = norm(p - x)                                                                                    
                return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)                                         
            return ComplexQuad(func, qa, p) + ComplexQuad(func, p, qa) \
                 + ComputeL(0.0, p, qa, qb, True)                                                              
    else:                                                                                                          
        if k == 0.0:                                                                                               
            return -0.5 / np.pi * ComplexQuad(lambda q: np.log(norm(p - q)), qa, qb)                           
        else:                                                                                                      
            return 0.25j * ComplexQuad(lambda q: hankel1(0, k * norm(p - q)), qa, qb)                          
    return 0.0                                                                                                     
                                                                                                                   
def ComputeM(k, p, qa, qb, pOnElement):                                                                       
    qab = qb - qa                                                                                                  
    vecq = Normal2D(qa, qb)                                                                                    
    if pOnElement:                                                                                                 
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecq) / np.dot(r, r)                                                              
            return -0.5 / np.pi * ComplexQuad(func, qa, qb)                                                    
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecq) / R                                                     
            return 0.25j * k * ComplexQuad(func, qa, qb)                                                       
    return 0.0                                                                                                 
                                                                                                                   
def ComputeMt(k, p, vecp, qa, qb, pOnElement):                                                                
    qab = qb - qa                                                                                                  
    if pOnElement:                                                                                                 
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecp) / np.dot(r, r)                                                              
            return -0.5 / np.pi * ComplexQuad(func, qa, qb)                                                    
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecp) / R                                                     
            return -0.25j * k * ComplexQuad(func, qa, qb)                                                      
                                                                                                                   
def ComputeN(k, p, vecp, qa, qb, pOnElement):                                                                 
    qab = qb- qa                                                                                                   
    if pOnElement:                                                                                                 
        ra = norm(p - qa)                                                                                          
        rb = norm(p - qb)                                                                                          
        re = norm(qab)                                                                                             
        if k == 0.0:                                                                                               
            return -(1.0 / ra + 1.0 / rb) / (re * 2.0 * np.pi) * re                                                
        else:                                                                                                      
            vecq = Normal2D(qa, qb)                                                                            
            k2 = k * k                                                                                             
            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                R = np.sqrt(R2)                                                                                    
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                dpnu = np.dot(vecp, vecq)                                                                          
                c1 =  0.25j * k / R * hankel1(1, k * R)                                  - 0.5 / (np.pi * R2)      
                c2 =  0.50j * k / R * hankel1(1, k * R) - 0.25j * k2 * hankel1(0, k * R) - 1.0 / (np.pi * R2)      
                c3 = -0.25  * k2 * np.log(R) / np.pi                                                               
                return c1 * dpnu + c2 * drdudrdn + c3                                                              
            return ComputeN(0.0, p, vecp, qa, qb, True) - 0.5 * k2 * ComputeL(0.0, p, qa, qb, True) \
                 + ComplexQuad(func, qa, p) + ComplexQuad(func, p, qb)                                     
    else:                                                                                                          
        sum = 0.0j                                                                                                 
        vecq = Normal2D(qa, qb)                                                                                
        un = np.dot(vecp, vecq)                                                                                    
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                return (un + 2.0 * drdudrdn) / R2                                                                  
            return 0.5 / np.pi * ComplexQuad(func, qa, qb)                                                     
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / np.dot(r, r)                                       
                R = norm(r)                                                                                        
                return hankel1(1, k * R) / R * (un + 2.0 * drdudrdn) - k * hankel1(0, k * R) * drdudrdn            
            return 0.25j * k * ComplexQuad(func, qa, qb)          