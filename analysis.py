from sympy import *
import numpy as np

f0, f1, f2, f3, f4, f5, f6, f7, f8 = symbols("f0 f1 f2 f3 f4 f5 f6 f7 f8")
c = Matrix(9, 2, [0,  0, -1,  1, -1, 0, -1, -1, 0, -1, 1, -1, 1,  0, 1,  1, 0,  1])
weights = Matrix(9, 1, [Rational(4,9), Rational(1,36), Rational(1,9), Rational(1,36), Rational(1,9), Rational(1,36), Rational(1,9), Rational(1,36), Rational(1,9)])

pops = Matrix(9, 1, [f0, f1, f2, f3, f4, f5, f6, f7, f8])
rho = 1 + sum(pops)
ux = pops.dot(c[:,0])/rho
uy = pops.dot(c[:,1])/rho
u = Matrix(2, 1, [ux, uy])
invCs2 = 3
c_u = c*u

equilibrium_term1 = ones(9,1)
equilibrium_term2 = invCs2*c_u
equilibrium_term3 = invCs2**2 * Rational(1,2) * hadamard_product(c_u, c_u)
equilibrium_term4 = ones(9,1) * invCs2 * Rational(-1,2) * u.dot(u)

equilibrium = hadamard_product(rho*weights, equilibrium_term1 + equilibrium_term2 + equilibrium_term3 + equilibrium_term4)

desiredRho = 1.0
desiredU = np.matrix([[0.1*invCs2, 0.]]).T
desiredC_U = np.matrix(c).astype(np.float64)*desiredU
desiredPops = np.multiply(desiredRho*np.array(weights).astype(np.float64),1.0 + invCs2*desiredC_U + invCs2**2 * 0.5 * np.multiply(desiredC_U, desiredC_U) - invCs2*0.5*desiredU.T*desiredU)

equilibriumNumerical = equilibrium.evalf(subs={f0: 0.1, f1: 0.11, f2: 0.12, f3: 0.13, f4: 0.14, f5: 0.15, f6: 0.16, f7: 0.17, f8: 0.18})

Jij = np.zeros((9,9))

for i in range(0,8):
    Jij[i,0] = series(equilibrium[i], f0, x0=Rational(desiredPops[0,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,1] = series(equilibrium[i], f1, x0=Rational(desiredPops[1,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,2] = series(equilibrium[i], f2, x0=Rational(desiredPops[2,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,3] = series(equilibrium[i], f3, x0=Rational(desiredPops[3,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,4] = series(equilibrium[i], f4, x0=Rational(desiredPops[4,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,5] = series(equilibrium[i], f5, x0=Rational(desiredPops[5,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,6] = series(equilibrium[i], f6, x0=Rational(desiredPops[6,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,7] = series(equilibrium[i], f7, x0=Rational(desiredPops[7,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    Jij[i,8] = series(equilibrium[i], f8, x0=Rational(desiredPops[8,0].item()), n=2).evalf(subs={f0: Rational(desiredPops[0,0].item()), 
                                                                                                f1: Rational(desiredPops[1,0].item()),
                                                                                                f2: Rational(desiredPops[2,0].item()),
                                                                                                f3: Rational(desiredPops[3,0].item()),
                                                                                                f4: Rational(desiredPops[4,0].item()),
                                                                                                f5: Rational(desiredPops[5,0].item()),
                                                                                                f6: Rational(desiredPops[6,0].item()),
                                                                                                f7: Rational(desiredPops[7,0].item()),
                                                                                                f8: Rational(desiredPops[8,0].item())})
    
# print(equilibrium[4])
# Jij_0_5 = series(equilibrium[4], f5, x0=Rational(desiredPops[5,0].item()), n=2)
# print(Jij_0_5)
# Jij_0_5_real = Jij_0_5.evalf(subs={f0: Rational(desiredPops[0,0].item()), 
#                                    f1: Rational(desiredPops[1,0].item()),
#                                    f2: Rational(desiredPops[2,0].item()),
#                                    f3: Rational(desiredPops[3,0].item()),
#                                    f4: Rational(desiredPops[4,0].item()),
#                                    f5: Rational(desiredPops[5,0].item()),
#                                    f6: Rational(desiredPops[6,0].item()),
#                                    f7: Rational(desiredPops[7,0].item()),
#                                    f8: Rational(desiredPops[8,0].item())})

print(Jij)