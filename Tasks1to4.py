# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:20:08 2022

@author: sophi

Tasks 1-4
See Methods document for equations
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

#Based on Equation 1, calculating V parameter 
V = a* k0 * np.sqrt(n1**2 - n2**2)

#An array of Beta values so that...?
beta = np.linspace(n2*k0, n1*k0, 1000)
p = np.sqrt((n1*k0)**2-beta**2) 
q = np.emath.sqrt((n2*k0)**2 - beta**2)
pa = p*a
qa = q*a


#%% Getting inital guesses (MAYBE CHANGE TO ONLY PLOT RELEVANT PLOTS?)
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
        Jtop =  scipy.special.jv(m+1, pa)
        Ktop = scipy.special.kv(m+1, qa)
        # Jtop =  scipy.special.jvp(m, pa)
        # Ktop = scipy.special.kvp(m, qa)
        RHS = -Ktop/(qa*Km)
        plt.title('{}, pmp'.format(m))
    else:
        Jtop =  scipy.special.jvp(m-1, pa)
        Ktop = scipy.special.kvp(m-1, qa)
        # Jtop =  scipy.special.jvp(m, pa)
        # Ktop = scipy.special.kvp(m, qa)
        RHS = Ktop/(qa*Km)
        plt.title('{}, mpm'.format(m))
    if TE:
        plt.plot(pa, RHS, c='red')
    else:
        plt.plot(pa, (n2**2/n1**2)*RHS, c='red')
    plt.plot(pa, Jtop/(pa*Jm), c='black')
    plt.vlines(V, -3.5, 3.5, linestyles='dashed', color='black')
    plt.xlabel('pa')
    plt.xlim(0, V+0.5)
    plt.ylim(-2.5, 2.5)
    plt.show()



#%%
for i in range(0, 6):
    plotmode(i, True) #TE, EH pmp route
    plotmode(i, False) #TE, HE mpm route
#%%
#This is for ALL modes
for i in range(0, 1):
    plotmode(i, True, False) #TM pmp
    plotmode(i, False, False) #TM mpm
    
#%%
'This function is so that we can find the PRECISE roots, very similar to Plotmode function so maybe should be the same one'
'It takes data from new_table_task3 as an input, which was made based on our graphs' 
'It ouputs a Final table new with all the beta values and progation constant for Task 3'
'Note TE True models for when TE dominates so TE and HE'
'Note TE False models for when TM dominates so TM and EH'


table_task3 = pd.read_csv('new_table_task3.csv')
modes_type = table_task3.loc[:,"Mode"]

pa_results = []


for i in range(len(modes_type)):
    """
    This computes the roots using newton optimize for pmp, TE, TM, EH and HE
    Based on inital guesses from graphical solutions
    NOTE: DO WE NEED TO REMOVE DEGENERATE SOLUTIONS
    
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

'This calculates the beta values and effective index for each mode'
for i in range(len(pa_results)):
    #this loop calculates beta and propagation constant range(len(pa_results)
    #it also checks that it is not a no solution case, in which case it appends n/a
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
 
#This outputs a table with all our values 
table_task3.to_csv('Final_table_Task3_new.csv')

#checking results
# for i in range(len(beta_final)):
#     if beta_final[i] < n2*k0:
#         print('Beta values incorrect')
#     if beta_final > n1*k0:
#         print('Beta values incorrect')
