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
beta = np.linspace(n2*k0, n1*k0, 1000)

#Defining p (radial component in core) and q (raidal component in cladding)
p = np.sqrt((n1*k0)**2-beta**2) 
pa = p*a
q = np.sqrt(beta**2 - (n2*k0)**2)
qa= q*a

#%% Procuring inital guesses using a graphical method
def plotmode(m, pmp=True, TE=True):
    """
    We used this function to find our intial guessses for solving equation XXX
    m is the order of the mode
    pmp incicates if it is a plus minus plus (pmp) or minus plus minus (mpm) solution
    TE describes the type of mode
    """
    Jm = scipy.special.jv(m, pa) 
    Km = scipy.special.kv(m, qa)
    if pmp:
        Jtop =  scipy.special.jv(m+1, pa) #Derivative of Jm for Plus minus Plus solution
        Ktop = scipy.special.kv(m+1, qa) #Derivative of Km for Plus minus Plus solution
        RHS = -Ktop/(qa*Km)
        plt.title('{}, pmp'.format(m))
    else:
        Jtop =  scipy.special.jvp(m-1, pa) #Derivative of Jm for minus plus minus solution
        Ktop = scipy.special.kvp(m-1, qa) #Derivative of Km for minus plus minus solution
        RHS = Ktop/(qa*Km)
        plt.title('{}, mpm'.format(m))
    if TE:
        plt.plot(pa, RHS, c='red')
    else:
        plt.plot(pa, (n2**2/n1**2)*RHS, c='red')
    plt.plot(pa, Jtop/(pa*Jm), c='black') #Plotting the LHS
    plt.vlines(V, -3.5, 3.5, linestyles='dashed', color='black') # Vertical Dashed line at V
    plt.xlabel('pa')
    plt.xlim(0, V+0.5)
    plt.ylim(-2.5, 2.5)
    plt.show()

for i in range(0, 6):
    plotmode(i, True) #TE, EH pmp route
    plotmode(i, False) #TE, HE mpm route

for i in range(0, 1):
    plotmode(i, True, False) #TM pmp
    plotmode(i, False, False) #TM mpm
    
#%%
# Determination of precis locations based on the positions found from the graphical method 
# It takes data from new_table_task3 as an input, which was made based on our graphs
# It ouputs a Final table new with all the beta values and progation constant for Task 3
# Note TE True models for when TE dominates so TE and HE
# Note TE False models for when TM dominates so TM and EH

table_task3 = pd.read_csv('new_table_task3.csv')
modes_type = table_task3.loc[:,"Mode"]

pa_results = []


for i in range(len(modes_type)):
    """
    This computes the roots using Newton Raphson methods for pmp, TE, TM, EH and HE
    Based on inital guesses from graphical solutions
    """
    initial_guess =  table_task3.loc[:, "Initial guess"][i]
    
    if initial_guess != 0:
         m = table_task3.loc[:, "m"][i]
         sign = table_task3.loc[:, "sign"][i]
         mode_type  = table_task3.loc[:, "Mode"][i]
         
         def roots_checkdef(pa):
            qa = (V**2-pa**2)**0.5
            Jm = scipy.special.jv(m, pa)
            Km = scipy.special.kv(m, qa)
            Jtop =  scipy.special.jvp(m, pa)
            Ktop = scipy.special.kvp(m, qa)
            
            if sign == 'pmp': #plus minus plus
                Jtop =  scipy.special.jv(m+1, pa)
                Ktop = scipy.special.kv(m+1, qa)
            # Jtop =  scipy.special.jvp(m, pa)
            # Ktop = scipy.special.kvp(m, qa)
                LHS = Jtop/(pa*Jm)
                RHS = -Ktop/(qa*Km)
                
            else: #minus plus minus
                Jtop =  scipy.special.jv(m-1, pa)
                Ktop = scipy.special.kv(m-1, qa)
                # Jtop =  scipy.special.jvp(m, pa)
                # Ktop = scipy.special.kvp(m, qa)
                LHS = Jtop/(pa*Jm)
                RHS = Ktop/(qa*Km)
            
            if mode_type == 'TM' or 'EH':
                LHS = (n2**2/n1**2)*LHS #TO ACCOUNT FOR TM modes
                return LHS - RHS
         
         output_roots = optimize.newton(roots_checkdef, x0 = initial_guess , fprime=None, 
                           tol=1.38e-6, maxiter=1500, 
                          fprime2=None, x1=None, rtol=0.0, full_output=False,
                         disp=True)
         
         pa_results.append(output_roots)
    else:
        pa_results.append('No_solution')
    

#inserting results for 'pa' into a table
table_task3.insert(5, 'Precise roots', pa_results)


beta_final = []
n_eff = []

#This loop calculates beta and propagation constant for found root
#it also checks that it is not a no solution case, in which case it appends n/a
for i in range(len(pa_results)): 
    if isinstance(pa_results[i], np.float64) == True:
        p_value = (pa_results[i])/a
        beta_b = np.sqrt((n1**2)*(k0**2) - p_value**2)
        beta_final.append(beta_b) 
        n_eff.append(beta_b/k0)
    
    else:
        beta_final.append('N/a') 
        n_eff.append('N/a')
    

table_task3.insert(6, 'Beta', beta_final)
table_task3.insert(7, 'Neff', n_eff)
 
#This outputs a table that summaries the results of TASKS 1-4 
table_task3.to_csv('Final_table_Task3_new.csv')

def check_results():
   for i in range(len(beta_final)):
        if beta_final[i] < n2*k0:
            print('Beta values incorrect')
        if beta_final > n1*k0:
            print('Beta values incorrect')

#%% TASKS 5-6
'Chosen mode HE, m=1, j=2. Beta value taken from Final_Table'
beta = 14701123.9
neff = beta/k0

r_core = np.linspace(0, a, 1000) # Array of radii for plotting values r<a
r_clad = np.linspace(a, 3*a, 1000) # Array of radii for plotting values r>a
phi = np.linspace(0, 2*np.pi, 1000)  # Array of angles between 0 and 2pi

R_CORE, PHI = np.meshgrid(r_core, phi) #Meshgrid for 3D plotting
R_CLAD, PHI_CLAD = np.meshgrid(r_clad, phi) #Meshgrid for 3D plotting

m = 1 #mode
j = 2 

#defining functions to reduce space
JM = scipy.special.jv(m, p*a)
KM = scipy.special.kv(m, q*a)
JMprime = scipy.special.jvp(m, p*a)
KMprime = scipy.special.kvp(m, q*a)

#matrix to be solved to determine the A, B, C, D vector
'based on page 6 equations'
mat = [[JM, 0, -KM, 0], [0, JM, 0,  -KM], [1j*m*beta/(a*p**2)*JM, -mu0*omega/p*JMprime, 1j*m*k0/(a*q**2)*KM, -mu0*omega/q*KMprime], [e0*n1**2*omega/p*JMprime, 1j*m*beta/(a*p**2)*JM, e0*n2**2*omega/q*KMprime, -1j*m*beta/(a*q**2)*KM]]

A = 1 #Setting A = 1, to determine B, C and D
C = A*(JM/KM)
twoDmat = [[-mu0*omega/p*JMprime, -mu0*omega/q*KMprime], [1j*m*beta/(a*p**2)*JM, -1j*m*beta/(a*q**2)*KM]]
B, D = np.linalg.solve(twoDmat, [A*1j*m*beta/(a*p**2)*JM + C*1j*m*k0/(a*q**2)*KM, A*e0*n1**2*omega/p*JMprime + C*e0*n2**2*omega/q*KMprime])
vector = [A, B, C, D]
#%%
#Defining Electric Field Strength of the projections
'E = Field in z, field in r core, field in r cladding, field in phi core, field in phi cladding'
E = [[A*scipy.special.jv(m, p*R_CORE)*np.exp(1j*m*PHI), C*scipy.special.kv(m, p*R_CLAD)*np.exp(1j*m*PHI_CLAD)],  #field in Z 
     
     [(-1j/p**2)*np.exp(1j*m*PHI)*(beta*p*A*scipy.special.jvp(m, p*R_CORE) + 1j*mu0*omega/R_CORE*m*B*scipy.special.jv(m, p*R_CORE)),  #field in R core
      
      (-1j/q**2)*np.exp(1j*m*PHI_CLAD)*(beta*q*C*scipy.special.kvp(m, q*R_CLAD) + 1j*mu0*omega/R_CLAD*m*D*scipy.special.kv(m, q*R_CLAD))],  #field in R cladding
         
     [(-1j/p**2)*np.exp(1j*m*PHI)*(1j*beta*m*A/R_CORE*scipy.special.jv(m, p*R_CORE) + mu0*omega*p*B*scipy.special.jvp(m, p*R_CORE)), #field in Phi core
      
      (-1j/q**2)*np.exp(1j*m*PHI_CLAD)*(1j*beta*m/R_CLAD*C*scipy.special.kv(m, q*R_CLAD) + mu0*omega*q*D*scipy.special.kvp(m, q*R_CLAD))]] #field in Phi cladding
    
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

#for i in range(3):
    #plot_3D(E[i], titles[i])
    #plt.show()
 
#%%
'Task 5: Plot Electric Field Projections'
def plot_2D(E, title): #plotting function in 2D 
    ER_CORE, E_CLAD = E
    # plt.fig = (1)
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    ax.set_title(title)
    plot_task5 = plt.contourf(R_CORE*np.cos(PHI), R_CORE*np.sin(PHI), ER_CORE)
    plt.contourf(R_CLAD*np.cos(PHI), R_CLAD*np.sin(PHI), E_CLAD)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(plot_task5)
    plt.show()

for i in range(len(E)):
    plot_2D(E[i], titles[i])
    plt.show()

#%%
'Plot the spatial distribution'

def plot_intensity(E, title): #plotting function in 2D 
    E_r = E[1]
    E_phi = E[2]
    inten_core = (E_r[0])**2 + (E_phi[0])**2
    inten_cladding = (E_r[1])**2 + (E_phi[1])**2
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

#%% TASK 7
wavelength_array=np.array((620e-9, 625e-9, 630e-9, 635e-9, 640e-9))
def n_eff(x):
    xn=np.array(x)
    beta=14701123.9
    return (beta*xn)/(2*np.pi)
def D_w(l, two_dev):
    #wavelength in nm
    d=(-l/c)*two_dev
    return d

d = derivative(n_eff, wavelength_array, dx=1e-3, n=2)
#print(d)
Waveguide_dispersion = D_w(wavelength_array, d)*(10**(21))
print('Waveguide Dispersion:', Waveguide_dispersion, 'ps/(nm km)')
#Note waveguide dispersion is in ps/(nm km)

#%% TASK 8

def I_inf_a(r): #Er and Ephi for r<a
    Er = (-1j*beta/p**2) * ( (1j*m*p*A*scipy.special.jvp(m,p*r,1)) + (1j*omega*mu0*B*scipy.special.jv(m,p*r))/(beta*r) )
    Ephi = (-1j*beta/p**2) * ( (1j*m*A*scipy.special.jv(m,p*r)/r) - (omega*mu0*p*B*scipy.special.jvp(m,p*r,1)/beta) )
    I = abs(Er)**2 + abs(Ephi)**2
    return I

def I_sup_a(r): #Er and Ephi for r>a
    Er = (1j*beta/q**2) * ( C*q*scipy.special.kvp(m,q*r,1) + (1j*omega*mu0*m*D*scipy.special.kv(m,q*r))/(beta*r) )
    Ephi = (1j*beta/q**2) * ( 1j*m*C*scipy.special.kv(m,q*r)/r - omega*mu0*q*D*scipy.special.kvp(m,q*r,1)/beta )
    I = abs(Er)**2 + abs(Ephi)**2
    return I


#Plotting the sum of the square modulus
r1 = np.linspace(0,a,1000)
r2 = np.linspace(a,10*a,1000)
plt.plot(r1, I_inf_a(r1), color='r')
plt.plot(r2, I_sup_a(r2), color='b')
plt.show()


def inte_inf(r): #Integrande for the core
    return r*I_inf_a(r)
def inte_sup(r): #Integrande for the cladding
    return r*I_sup_a(r)

#Integration
intensitytot_core = scipy.integrate.quad(inte_inf,0,a)
intensitytot_cladding= scipy.integrate.quad(inte_sup,a,a*10)
frac_core = intensitytot_core[0]/(intensitytot_core[0] + intensitytot_cladding[0])
frac_cladding = intensitytot_cladding[0]/(intensitytot_core[0] + intensitytot_cladding[0]) 

'''
Output of scipy.integrate.quad is (result,error)
In an ideal world, frac_core[0]+frac_cladding[0] = total (which should be the integral of what you calculated task 6)
and then Gamma_core = frac_core/total and Gamma_cladding = 1 - Gamma_core

frac_core = 0.22
frac_cladding = 078

'''
