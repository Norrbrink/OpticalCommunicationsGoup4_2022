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
J2 = scipy.special.jv(2, pa)
J3 = scipy.special.jv(3, pa)
J4 = scipy.special.jv(4, pa)
J5 = scipy.special.jv(5, pa)
J6 = scipy.special.jv(6, pa)
J7 = scipy.special.jv(7, pa)

J11 = scipy.special.jv(-1, pa)



K0 = scipy.special.kv(0, qa)
K1 = scipy.special.kv(1, qa)
K2 = scipy.special.kv(2, qa)
K3 = scipy.special.kv(3, qa)
K4 = scipy.special.kv(4, qa)
K5 = scipy.special.kv(5, qa)
K6 = scipy.special.kv(6, qa)
K7 = scipy.special.kv(7, qa)

K11=scipy.special.kv(-1, qa)



def function_q3(epsilon):
    position=[]
    result=np.array([])
    for k in range(len(beta)):
        diff=(J6[k]/a*p[k]*J7[k])-(K6[k]/a*q[k]*K7[k]) #modify this line for all cases
        result = np.append(result,diff)
        if -epsilon < result[k] < epsilon:
            position.append(k)
    return(position)

#So guys the result vary a LOT and it's impossible to get smth near 0 as it will directly go from -10E6 to +10E6 with "only" 100000 different beta. Playing with epsilon around 10E7/10E6 there are these zeros :

#Case m=0
Position_m0=[955,14860,44590,83610]   #+-+ case
Position_m10=[955,14860,44590,83610] #-+- case

#Case +-+
Position_m1=[7821, 35563, 73537]
Position_m2=[22220,60080]
Position_m3=[37866]
Position_m4=[68383]
Position_m5=[99394]
Position_m6=[]


#Case -+-
Position_m11=[1488,14786,39882,83022]
Position_m12=[9852,37810,83148]
Position_m13=[22979,70535]
Position_m14=[46941]
Position_m15=[67598]
Position_m16=[99394]
Position_m17=[]

#Eliminating redundancies ????????????????????????
Position=[955,7821,9852,14860,22220,35563,37566,39882,44590,60080,68383,73537,83022,83610,99394]


def neff(position):
    Modes=[n2*k0+((n1-n2)*k0)*k/100000 for k in position]
    n_eff=[k/k0 for k in Modes]
    return(n_eff)
    
    
    
#plt.plot(pa, J1/(pa*J0), c='black')
#plt.plot(np.imag(qa), -K1/(np.imag(qa)*K0), c='red')
#plt.xlim(0, V)
#plt.ylim(-7.5, 7.5)

