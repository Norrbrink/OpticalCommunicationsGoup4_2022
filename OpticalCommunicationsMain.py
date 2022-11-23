# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:05:32 2022

@author: Alexander
"""
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

#Effective index is between 1.47 and 1.48 for all modes

#%%
beta = np.linspace(n2*k0, n1*k0, 1000)
p = np.sqrt((n1*k0)**2-beta**2) 
q = np.emath.sqrt((n2*k0)**2 - beta**2)
pa = p*a
qa = q*a
J0 = scipy.special.jv(0, pa)
J1 = scipy.special.jv(1, pa)
K0 = scipy.special.kv(0, qa)
K1 = scipy.special.kv(1, qa)

plt.plot(pa, J1/(pa*J0), c='black')
plt.plot(np.imag(qa), -K1/(np.imag(qa)*K0), c='red')
plt.xlim(0, V)
plt.ylim(-7.5, 7.5)