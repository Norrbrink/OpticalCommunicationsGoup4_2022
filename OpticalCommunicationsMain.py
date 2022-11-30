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

##Question 3

beta = np.linspace(n2*k0, n1*k0, 100000) #I put 100000 instead of 1000 for precision (but takes longer)
p = np.sqrt((n1*k0)**2-beta**2)
q = np.emath.sqrt((n2*k0)**2 - beta**2)
pa = p*a
qa = q*a
J0 = scipy.special.jv(0, pa)
J1 = scipy.special.jv(1, pa)
K0 = scipy.special.kv(0, qa)
K1 = scipy.special.kv(1, qa)


def function_q3(epsilon):
    position=[]
    result=np.array([])
    for k in range(len(beta)):
        diff=(J1[k]/a*p[k]*J0[k])+(K1[k]/a*q[k]*K0[k])
        result = np.append(result,diff)
        if -epsilon < result[k] < epsilon:
            position.append(k)
    return(position)

#So guys the result vary a LOT and it's impossible to get smth near 0 as it will directly go from -10E6 to +10E6 with "only" 100000 different beta. Playing with epsilon around 10E7 I think there are 4 zeros :
Position=[955,14860,44590,83610]

Modes=[n2*k0+((n1-n2)*k0)*k/100000 for k in Position]
#Output : [14661718.167858455, 14675586.055429302, 14705236.706093183, 14744152.56175765]

n_eff=[k/k0 for k in Modes]
#Output : [1.4700955, 1.471486, 1.474459, 1.478361] seems good


#plt.plot(pa, J1/(pa*J0), c='black')
#plt.plot(np.imag(qa), -K1/(np.imag(qa)*K0), c='red')
#plt.xlim(0, V)
#plt.ylim(-7.5, 7.5)
