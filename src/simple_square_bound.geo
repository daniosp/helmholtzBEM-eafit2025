 /* 
 .geo file for simple square boundary, 
Introduccion al Metodo de Frontera Universidad EAFIT 2025-1 
by: Daniel Ospina Pajoy, Sebastián Duque Lotero & Mateo Tabares. 
 */ 
 
 
// Inner Scatterer Element Size
 sizeRoI = 0.1; 
 
// Points 
Point(1) = { 0, 0, 0.0, 0.1 }; 
Point(2) = { 5, 0, 0.0, 0.1 }; 
Point(3) = { 5, 5, 0.0, 0.1 }; 
Point(4) = { 0, 5, 0.0, 0.1 }; 

 
// Lines 
Line(1) = { 1, 2 } ; 
Line(2) = { 2, 3 } ; 
Line(3) = { 3, 4 } ; 
Line(4) = { 4, 1 } ; 

 
// Surfaces 
Curve Loop(1) = { 1: 4 }; 
Plane Surface(1) = {1}; 

 
// Physical groups 
Physical Curve(1) = { 1,2,3,4 }; 
Physical Surface(2) = {1}; 

 
// Mesh parameters 
ndiv = 70; 
Transfinite Curve { 1,2,3, 4 } = ndiv Using Progression 1; 
Transfinite Surface {1}; 
