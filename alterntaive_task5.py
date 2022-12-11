# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:38:07 2022

@author: sophi
"""

# -*- coding: utf-8 -*-
"""
Main Code File that produces all the results of our Project
"""
#%% Importing packages
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from scipy.misc import derivative
import scipy.integrate

#%% 
# Initial Parameters
# 
n1 = 1.48
n2 = 1.47
a = 3.5e-6
wavelength = 630e-9
k0 = 2*np.pi/wavelength
mu0 = 4e-7*np.pi
e0 = 8.85e-12
omega = np.sqrt((k0**2/e0*mu0))
c=1/np.sqrt(mu0*e0) 

#Based on Equation 1, Calculating V parameter 
V = a*k0*np.sqrt(n1**2 - n2**2)

#An array of Beta values so that for plotting graphs to find modes
beta = 14701123.9
p = np.sqrt((n1*k0)**2-beta**2) 
pa = p*a
qa = np.sqrt(V**2-pa**2)
q = qa/a
neff = beta/k0

'difference with this is it is zero at zero lol'
q2 = np.sqrt(beta**2 - (n2*k0)**2)
qa2 = q*a
#%% 
m=1

JM = scipy.special.jv(m, p*a)
KM = scipy.special.kv(m, q*a)
JMprime = scipy.special.jvp(m, p*a)
KMprime = scipy.special.kvp(m, q*a)

'ABCD according to top of page 8 of the notes'
A = 1 #Setting A = 1, to determine B, C and D
C = A*(JM/KM)

#FROM notes equation 6 page 8 
Z = JMprime/(pa*JM) + KMprime/(qa*KMprime)
Y = (1/(pa)**2 + 1/(qa)**2)
X = (1j*beta*m)/omega*mu0
B = Z/(X*Y)

D = B*(JM/KM)

#%% 

r_core = np.linspace(0, a, 1000) # Array of radii for plotting values r<a
r_clad = np.linspace(a, 4*a, 1000) # Array of radii for plotting values r>a
phi = np.linspace(0, 2*np.pi, 1000)  # Array of angles between 0 and 2pi

R_CORE, PHI = np.meshgrid(r_core, phi) #Meshgrid for 3D plotting
R_CLAD, PHI_CLAD = np.meshgrid(r_clad, phi) #Meshgrid for 3D plotting


'ELECTRIC FIELD ACCORDING TO hopefully PAGE 7 OF THE NOTES'
E = [[A*scipy.special.jv(m, p*R_CORE), C*scipy.special.kv(m, p*R_CLAD)],  #field in Z 
     
     [(-1j*beta/p**2)*((p*A*scipy.special.jvp(m, p*R_CORE) + (1j*mu0*omega*m/(R_CORE*beta))*B*scipy.special.jv(m, p*R_CORE))),   #field in R core
      
      (1j*beta/q**2)*((q*C*scipy.special.kvp(m, q*R_CLAD) + (1j*mu0*omega*m/(R_CLAD*beta))*D*scipy.special.kv(m, q*R_CLAD)))],  #field in R cladding
         
     [(-1j*beta/p**2)*((1j*m/R_CORE)*A*scipy.special.jv(m, p*R_CORE) - omega*(mu0/beta)*p*B*scipy.special.jvp(m, p*R_CORE)), #field in Phi core
      
      (1j*beta/q**2)*((1j*m/R_CLAD)*C*scipy.special.kv(m, q*R_CLAD) - ((mu0*omega*q*D)/beta)*scipy.special.kvp(m, q*R_CLAD))]] #field in Phi cladding



#%% 

titles = ['Electric Field Projection in z', 'Electric Field Projection in r', 'Electric Field Projection in Phi']

'Task 5: Plot Electric Field Projections'
def plot_2D(E, title): #plotting function in 2D 
    ER_CORE, E_CLAD = E
    # plt.fig = (1)
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    ax.set_title(title)

    
    plot_task5 = plt.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), ER_CORE , cmap= 'bwr', levels=50)
    plt.contourf(R_CLAD*np.cos(PHI_CLAD), R_CLAD*np.sin(PHI_CLAD), E_CLAD, cmap= 'bwr', levels=50)
    
 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(plot_task5)
    plt.show()

for i in range(len(E)):
    plot_2D(E[i], titles[i])
    plt.show()




#%% 
titles2 = ['Imaginary Electric Field Projection in z', 'Imaginary Electric Field Projection in r', 'Imaginary Electric Field Projection in Phi']



'Task 5: Plot Electric Field Projections but this time its imaginary'
def plot_2D_im(E, title): #plotting function in 2D 
    ER_CORE, E_CLAD = E
    # plt.fig = (1)
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    ax.set_title(title)
    

    plot_task5 = plt.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI, np.imag(ER_CORE), cmap= 'bwr'))
    plt.contourf((R_CLAD*np.cos(PHI_CLAD)), (R_CLAD*np.sin(PHI_CLAD)), np.imag(E_CLAD), cmap= 'bwr')
    
    print('whole thing', E_CLAD)
        
    print('trial_IMAGINARYYY', title)
    # print(np.imag(ER_CLAD))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(plot_task5)
    plt.show()

for i in range(len(E)):
    plot_2D(E[i], titles2[i])
    plt.show()
    

#%% 
def plot_intensity(E, title): #plotting function in 2D 
    E_r = E[1]
    E_phi = E[2]
       
    inten_core = abs(E_r[0])**2 + abs(E_phi[0])**2
    inten_cladding = abs(E_r[1])**2 + abs(E_phi[1])**2
    
    
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    ax.set_title(title)
    plot_task5 = plt.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), inten_core)
    plt.contourf(R_CLAD*np.cos(PHI), R_CLAD*np.sin(PHI), inten_cladding)
    
    ax.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), inten_core)
    ax.contourf(R_CLAD*np.cos(PHI), R_CLAD*np.sin(PHI), inten_cladding)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.colorbar(plot_task5)
    plt.show()


plot_intensity(E, 'Total Intensity')
plt.show()

'NEW BETA METHOD STARTS HERE'


'see equation 3 , page 7 of lecture three notes'

m = 1

def roots_checkdef_new(pa):
    qa = (V**2-pa**2)**0.5
    JM = scipy.special.jv(m, pa)
    KM = scipy.special.kv(m, qa)
    JM_PRIME =  scipy.special.jvp(m, pa)
    KM_PRIME = scipy.special.kvp(m, qa)
    
    p_precise = pa/a 
    
    beta_old = np.sqrt((n1**2)*(k0**2) - p_precise**2)
    
    
    #USING EQUATION TOP OF PAGE 7
    Ja = JM_PRIME/(pa*JM)
    Ka = KM_PRIME/(qa*KM)
    N1_p = ((n1**2+n2**2)/(2*n1**2))
    
    N1_m = ((n1**2-n2**2)/(2*n1**2))
    
    SNOW = (1/(pa)**2 + 1/(qa)**2)
    
    pre = (beta_old**2)*(m**2)/((n1**2)*(k0**2))
    
    
    C1 = ((N1_m)**2)*(Ka)**2 + pre*SNOW**2
    

    zeros = -Ja - N1_p*Ka - C1
    return zeros 

 

output_roots = optimize.newton(roots_checkdef_new, x0 = 4 , fprime=None, 
                           tol=1.38e-6, maxiter=1500, 
                          fprime2=None, x1=None, rtol=0.0, full_output=False,
                 
                            disp=True)


beta_b = np.sqrt((n1**2)*(k0**2) - 4.446909818634911**2)
print(beta_b)
   

