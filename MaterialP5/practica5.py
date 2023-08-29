#Autor: Miguel Carracedo Rodríguez
#12/05/2022

from cmath import pi
from curses.ascii import FS
from blinker import signal
import numpy as np
import matplotlib.pyplot as plt
import pds
import scipy.signal as signal 


#-----------------------------------------------------------

#Tarea 1

Fs = 8000
Fc =2000
M = 31
wc = np.pi/2

hamming = np.hamming(M)

def filtroFIR():

    h = np.zeros(M)

    for n in range (M):
        if n != (M-1)/2:
            h[n] = (np.sin(wc * ( n - ((M-1)/2) ))) / (np.pi * (n - ((M-1)/2)))
        else:
            h[n] = wc/np.pi

    return h


s = filtroFIR()


# Representa el filtro
plt.figure()
plt.plot(s, label = 'h')
plt.xlabel('Tiempo muestreado')
plt.ylabel('Amplitud')
plt.title('Filtro FIR')
#plt.savefig('Tarea1')
#plt.show()


plt.figure()
pds.plot_freq_resp(s, 1)

#atenuacion de -50 db
s1 = s * hamming

#plt.figure()
pds.plot_freq_resp(s1, 1)



#-----------------------------------------------------------

#Tarea 2

numtaps15 = 15
numtaps31 = 31
numtaps63 = 63
cutoff= 0.5

b15 = signal.firwin(numtaps15, cutoff)
b31 = signal.firwin(numtaps31, cutoff)
b63 = signal.firwin(numtaps63, cutoff)


# Respuesta en frecuencia
pds.plot_freq_resp(b15,1)
pds.plot_freq_resp(b31,1)
pds.plot_freq_resp(b63,1)


#comparar 31 con ventanas de Bartlett y Hamming

#Hamming
b_hamming = b31 * hamming
pds.plot_freq_resp(b_hamming,1)

#Bartlett
bartlett = np.bartlett(31)
b31 = b31 * bartlett
pds.plot_freq_resp(b31,1)

