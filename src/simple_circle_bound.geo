 /* 
 .geo file for simple circle boundary, 
Introduccion al Metodo de Frontera Universidad EAFIT 2025-1 
by: Daniel Ospina Pajoy, Sebastián Duque Lotero & Mateo Tabares. 
 */ 
 
 
// Points 
Point(1) = { 0.0 , 0.0 , 0.0, 0.5 }; 
Point(2) = { -7*Cos(Pi/4), -7*Cos(Pi/4), 0.0, 0.5 }; 
Point(3) = { 7*Cos(Pi/4), -7*Cos(Pi/4), 0.0, 0.5 }; 
Point(4) = { 7*Cos(Pi/4), 7*Cos(Pi/4), 0.0, 0.5 }; 
Point(5) = { -7*Cos(Pi/4), 7*Cos(Pi/4), 0.0, 0.5 }; 

 
// Lines 
Circle(1) = { 2, 1, 3 } ; 
Circle(2) = { 3, 1, 4 } ; 
Circle(3) = { 4, 1, 5 } ; 
Circle(4) = { 5, 1, 2 } ; 

 
// Surfaces 
Curve Loop(1) = { 1,2,3,4 }; 
Plane Surface(1) = {1}; 

 
// Physical groups 
Physical Curve(1) = { 1,2,3,4 }; 
Physical Surface(2) = {1}; 

 
// Mesh parameters 
ndiv = 20; 
Transfinite Curve { 1,2,3, 4 } = ndiv Using Progression 1; 
Transfinite Surface {1}; 
