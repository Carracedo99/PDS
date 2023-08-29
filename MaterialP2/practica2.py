#Autor: Miguel Carracedo Rodríguez
#31/03/2022

from cmath import pi
from blinker import signal
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.signal as ss


Fs, signal = wavfile.read('./digitos.wav')
time = 20
signal = signal.astype(float)
t = np.arange(0, len(signal)/Fs, 1/Fs)

# Representa la señal
plt.figure()
plt.plot(t, signal)  
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Digitos.wav')
plt.savefig('Digitos')
#plt.show()

#-----------------------------------------------------------
#Tarea 1

time = 18.75
t = np.arange(0, time, 1/Fs)

signalOut = 0

for i in range(1,13,2):
    signalOut +=  np.sin(t*i)/i

signalOut = signalOut * (4/pi) 

# Representa la señal
plt.figure()
plt.plot(t, signalOut)  
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Serie Fourier.wav')
plt.savefig('Series_Fourier')
#plt.show()

#--------------------------------------------------------
#Tarea 2

N = 64 * 16
T = 16
f = 1 / T
W = T* f

t = np.arange(0, N)
sqs = ss.square(2*np.pi*f * t)
X = np.fft.fft(sqs)

freqs = np.fft.fftfreq(X.size)
fig, ax = plt.subplots(2,1)

plt.xlabel('Tiempo s')
plt.ylabel('Frecuencia Hz')

ax[0].plot(freqs, np.abs(X))
ax[1].plot(freqs, np.angle(X))

# Representa la señal
#plt.figure()
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Transformada Fourier1.wav')
plt.savefig('Transformada_Fourier1')
#plt.show()

# Representa la cuadrada
#el T será 2*pi
plt.figure()
plt.plot(t, sqs)  
plt.title('Cuadrada Fourier')   #no la queremos para nada pero bueno.
plt.savefig('Cuadrada Fourier')
#plt.show()


#------------------------------------------------------------
#Tarea 3

T = 16
N = 64 * T                 
t = np.arange(0, N)
f = 1/T

w = np.exp(-1j * 2 * np.pi / N)
signal = np.zeros(N, dtype=complex)


def dft_alt(i,signal):
    signalOut = np.zeros_like(signal, dtype=complex)    
    for k in range(0,i-1):
        for n in range(0,i-1):
            signalOut[k] +=  signal[n] * (w**(n*k))    

    return signalOut


signalOut = dft_alt(N,sqs)                


# Representa la señal
plt.figure()
plt.plot(t, abs(signalOut))                  #abs (valor absoluto) para evitar el warning de numeros complejos
plt.xlabel('Tiempo (ms)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Serie Fourier Muestreada.wav')
plt.savefig('Series_Fourier_Muestreada')
#plt.show()


#------------------------------------------------------------
#Tarea 4

Fs, signal = wavfile.read('./digitos.wav')

signalOut = np.zeros_like(signal, dtype=complex)    
digitos = np.zeros_like(signal, dtype=complex)    
tamano = 800        #500-1000, ir probando
umbral = tamano**2

claves_candidatas = []

def buscarpicos(x):
    freqs = np.fft.fftfreq(len(x))
    candidatas = freqs[abs(x) > umbral] * Fs
    candidatas = candidatas[candidatas > 0]
    return candidatas

#dentro del for comprobamos las
#claves_candidatas > 1000 = columna, <1000 = fila y podemos sacar el numero los nun los leemos como que no hay nada

for n in range(0,len(signal),tamano): 
    salida =  np.fft.fft(signal[n : n+tamano])      #no hace falta grafica
    claves_candidatas = buscarpicos(salida)

    print('Las claves candidatas son: ' , claves_candidatas)

#np.argmin(algo) --> EL arumento MINIMO


