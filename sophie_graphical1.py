# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:47:45 2022

@author: sophi
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

# Initial Parameters
n1 = 1.48
n2 = 1.47
a = 3.5e-6
wavelength = 630e-9
k0 = 2*np.pi/wavelength
V = a* k0 * np.sqrt(n1**2 - n2**2)

#Effective index is between 1.47 and 1.48 for all modes

#%%
def plotmode(m, pmp=True, TE=True):
    
    Jm = scipy.special.jv(m, pa)
    Km = scipy.special.kv(m, qa)
    if pmp:
        Jtop =  scipy.special.jv(m+1, pa)
        Ktop = scipy.special.kv(m+1, qa)
        LHS = -Ktop/(qa*Km)
        plt.title('{}, pmp'.format(m))
    else:
        Jtop =  scipy.special.jv(m-1, pa)
        Ktop = scipy.special.kv(m-1, qa)
        LHS = Ktop/(qa*Km)
        plt.title('{}, mpm'.format(m))
    if TE:
        plt.plot(pa, LHS, c='red')
    else:
        plt.plot(pa, (n2**2/n1**2)*LHS, c='red')
    plt.plot(pa, Jtop/(pa*Jm), c='black')
    plt.vlines(V, -3.5, 3.5, linestyles='dashed', color='black')
    plt.xlabel('pa')
    plt.xlim(0, V+0.5)
    plt.ylim(-2.5, 2.5)
    plt.show()

'I think these need to be inside the function, so they arent defined outside the function??'
beta = np.linspace(n2*k0, n1*k0, 1000)
p = np.sqrt((n1*k0)**2-beta**2) 
q = np.emath.sqrt((n2*k0)**2 - beta**2)
pa = p*a
qa = (V**2-pa**2)**0.5

#%%
#This is old, delete?
for i in range(0, 6):
    plotmode(i, True)
    plotmode(i, False)
#%%
#This is for ALL modes
for i in range(0, 6):
    plotmode(i, True, False) #TE,EH,HE
    plotmode(i, False, False) #TM
    
#%%
'This function is so that we can find the PRECISE roots' 
'NOTE pa must not be defined so that pa can be optimised?'
'Note TE True models for when TE dominates so TE and HE'
'Note TE False models for when TM dominates so TM and EH'

table_task3 = pd.read_csv('table_task3.csv')
modes_type = table_task3.loc[:,"Mode"]

pa_results = []


for i in range(len(modes_type)):
    """
    This computes the roots using newton optimize for pmp, TE, TM, EH and HE
    Based on inital guesses from graphical solutions
    NOTE: WE NEED TO REMOVE DEGENERATE SOLUTIONS
    
    """
    initial_guess =  table_task3.loc[:, "Initial guess"][i]
    
    if initial_guess != 0:
         m = table_task3.loc[:, "m"][i]
         sign = table_task3.loc[:, "sign"][i]
         mode_type  = table_task3.loc[:, "Mode"][i]
         
         def roots_checkdef(pa):
             #THIS DOESNT REALLY MAKE SENSE SINCE WE ARE OPTIMISING pa, flawed that qa relies on pa..
            qa = (V**2-pa**2)**0.5
            Jm = scipy.special.jv(m, pa)
            Km = scipy.special.kv(m, qa)
            
            if sign == 'pmp': #plus minus plus
                Jtop =  scipy.special.jv(m+1, pa)
                Ktop = scipy.special.kv(m+1, qa)
                LHS = Jtop/(pa*Jm)
                RHS = -Ktop/(qa*Km)
            else: #minus plus minus
                Jtop =  scipy.special.jv(m-1, pa)
                Ktop = scipy.special.kv(m-1, qa)
                LHS = Jtop/(pa*Jm)
                RHS = Ktop/(qa*Km)
                # DIFFERENT TYPES OF MODE
            
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
    

table_task3.insert(5, 'Precise roots', pa_results)



#pa_array = np.array([pa_results])

beta = []
prop_constant = []

for i in range(len(pa_results)):
    #this loop calculates beta and propagation constant range(len(pa_results)
    #it also checks that it is not a no solution case, in which case it appends n/a
    if isinstance(pa_results[i], np.float64) == True:
        p_value = (pa_results[i])/a
        beta_b = np.sqrt((n1**2)*(k0**2) - p_value**2)
        beta.append(beta_b) 
        prop_constant.append(beta_b/k0)
    
    else:
        beta.append('N/a') 
        prop_constant.append('N/a')
    
    
table_task3.insert(6, 'Beta', beta)
table_task3.insert(7, 'Propagation Constant', prop_constant)
 

table_task3.to_csv('Final_table_Task3.csv')












