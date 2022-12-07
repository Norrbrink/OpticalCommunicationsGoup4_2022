# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:47:45 2022

@author: sophi
"""

#importing packages
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

#%% Initial Parameters
n1 = 1.48
n2 = 1.47
a = 3.5e-6
wavelength = 630e-9
k0 = 2*np.pi/wavelength
V = a* k0 * np.sqrt(n1**2 - n2**2)
mu0 = 4e-7*np.pi
e0 = 8.85e-12

'why is omega 2 different'
omega = 3e8*2*np.pi/wavelength
omega2 = np.sqrt((k0**2/e0*mu0))

'Chosen mode HE, m=1, j=1'
beta = 14701123.9
p = np.sqrt((n1*k0)**2-beta**2) 
# q = np.sqrt(beta**2 - (n2*k0)**2)
# q = np.emath.sqrt((n2*k0)**2 - beta**2)
pa = p*a
qa = (V**2-pa**2)**0.5
q = qa/a
'keeping q to be real here'


neff = beta/k0

r_core = np.linspace(0, a, 1000) # Array of radii for plotting values r<a
r_clad = np.linspace(a, 3*a, 1000) # Array of radii for plotting values r>a
phi = np.linspace(0, 2*np.pi, 1000)  # Array of angles between 0 and 2pi

R_CORE, PHI = np.meshgrid(r_core, phi) #Meshgrid for 3D plotting
R_CLAD, PHI1 = np.meshgrid(r_clad, phi) #Meshgrid for 3D plotting

m = 1 #mode
j = 2 

#defining functions to reduce space

JM = scipy.special.jv(m, p*a)
KM = scipy.special.kv(m, q*a)
#function jvp used
JMprime = scipy.special.jvp(m, p*a)
KMprime = scipy.special.kvp(m, q*a)

#matrix to be solved to determine the A, B, C, D vector
'based on page 6 equations'
mat = [[JM, 0, -KM, 0], [0, JM, 0,  -KM], [1j*m*beta/(a*p**2)*JM, -mu0*omega/p*JMprime, 1j*m*k0/(a*q**2)*KM, -mu0*omega/q*KMprime], [e0*n1**2*omega/p*JMprime, 1j*m*beta/(a*p**2)*JM, e0*n2**2*omega/q*KMprime, -1j*m*beta/(a*q**2)*KM]]

'page 7/8 equation 6 '
A = 1 #Setting A = 1, to determine B, C and D
C = A*(JM/KM)
twoDmat = [[-mu0*omega/p*JMprime, -mu0*omega/q*KMprime], [1j*m*beta/(a*p**2)*JM, -1j*m*beta/(a*q**2)*KM]]
B, D = np.linalg.solve(twoDmat, [A*1j*m*beta/(a*p**2)*JM + C*1j*m*k0/(a*q**2)*KM, A*e0*n1**2*omega/p*JMprime + C*e0*n2**2*omega/q*KMprime])
vector = [A, B, C, D]

#Defining Electric Field Strength of the projections
'matrix on page 6'
E = [[A*scipy.special.jv(m, p*R_CORE)*np.exp(1j*m*PHI), C*scipy.special.kv(m, p*R_CLAD)*np.exp(1j*m*PHI1)], \
     [(-1j/p**2)*np.exp(1j*m*PHI)*(beta*p*A*scipy.special.jv(m+1, p*R_CORE) + 1j*mu0*omega/R_CORE*m*B*scipy.special.jv(m, p*R_CORE)), \
      (-1j/q**2)*np.exp(1j*m*PHI)*(beta*q*C*scipy.special.kv(m+1, q*R_CLAD) + 1j*mu0*omega/R_CLAD*m*D*scipy.special.kv(m, q*R_CLAD))], \
     [(-1j/p**2)*np.exp(1j*m*PHI)*(1j*beta*m*A/R_CORE*scipy.special.jv(m, p*R_CORE) + mu0*omega*p*B*scipy.special.jv(m+1, p*R_CORE)), \
      (-1j/q**2)*np.exp(1j*m*PHI)*(1j*beta*m/R_CORE*C*scipy.special.kv(m, q*R_CLAD) + mu0*omega*q*D*scipy.special.kv(m+1, q*R_CLAD))]]
    
    
titles = ['Electric Field Projection in z', 'Electric Field Projection in r', 'Electric Field Projection in Phi']



def plot_3D(E, title): #plotting function
    ER_CORE, E_CLAD = E
 
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.contour3D(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), ER_CORE, cmap='binary')
    ax.contour3D(R_CLAD*np.cos(PHI), R_CLAD*np.sin(PHI), E_CLAD, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'Electric Field Strength $V m^{-1}$')
    ax.view_init(30, 30)
    plt.show()

for i in range(3):
    plot_3D(E[i], titles[i])
    plt.show()
    
    
#%%
'not convinced about this' 
 
def plot_2D(E, title): #plotting function in 2D 
    ER_CORE, E_CLAD = E
    ax = plt.axes()
    ax.set_title(title)
    ax.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), ER_CORE)
    ax.contourf(R_CLAD*np.cos(PHI), R_CLAD*np.sin(PHI), E_CLAD)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

for i in range(4):
    plot_2D(E[i], titles[i])
    plt.show()

#%% TASK 6
'Plot the spatial distriution'

#sum of the square modulus of the field in the {radial, tangential}

# inten_core = np.sqrt(R_CORE*np.cos(PHI)**2 + R_CORE*np.sin(PHI)**2)
# inten_cladding = np.sqrt(R_CLAD*np.cos(PHI)**2 + R_CLAD*np.sin(PHI)**2)

# inten_core = np.sqrt(r_core*np.cos(phi)**2 + r_core*np.sin(phi)**2)
# inten_cladding = np.sqrt(r_core*np.cos(phi)**2 + r_clad*np.sin(phi)**2)



def plot_intensity(E, title): #plotting function in 2D 
    E_z = E[0]
    E_r = E[1]
    E_phi = E[2]
    
    inten_core = np.sqrt((E_r[0])**2 + E_phi[0]**2)
    inten_cladding = np.sqrt((E_r[1])**2 + E_phi[1]**2)
    
    ax = plt.axes()
    ax.set_title(title)
    ax.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), inten_core)
    ax.contourf(R_CLAD*np.cos(PHI), R_CLAD*np.sin(PHI), inten_cladding)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


plot_2D(E, 'Total Intensity')
plt.show()
    


"""
NOTES
 tough...
A
Out[87]: 1

B
Out[88]: 4.3338602569520893e-08j

C
Out[89]: -16.967021530029275

D
Out[90]: (-0-7.35327002878443e-07j)

#setting A to unity
A = 1 
  
Jm = scipy.special.jv(m, pa)
Km = scipy.special.kv(m, qa)

#derivative of Jm and Km
#mpm
Jm_D = scipy.special.jvp(m, pa)
Km_D = scipy.special.kvp(m, qa)

C = Jm/Km

def eqn6(A,m):
    X = (1j*beta*m/omega*u0)
    Y = (1/(pa)**2)+(1/(qa)**2)
    Z = (Jm_D/pa*Jm)+(Km_D/qa*Km)
    B = (A*X*Y)/Z
    return B

B = eqn6(1,1)
D = B * (Jm/Km)

print('Sophies', A,B,C,D)

"""



