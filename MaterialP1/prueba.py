#Autor: Miguel Carracedo Rodríguez
#28/03/2022

from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from cProfile import label
from telnetlib import ECHO
from blinker import signal
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

Fs, signal = wavfile.read('./muneca.wav')
signal = signal.astype(float)
t = np.arange(0, len(signal)/Fs, 1/Fs)

ECHO_DURATION_SEC= 0.1

delay_len_samples = round(ECHO_DURATION_SEC * Fs)
leading_zero_padding_sig = np.zeros(delay_len_samples)
delay_sig = np.concatenate((leading_zero_padding_sig, signal))

#prueba simple del programa
#Fs = 100
#t = np.arange(0, 5000/Fs, 1/Fs)
#signal = np.sin(t)

# Representa la señal
plt.plot(t, signal)  
plt.title('Muneca.wav')
plt.savefig('Muneca')
plt.show()

#-----------------------------------------------------------------------------
#Tarea 1

signalOut = np.zeros_like(signal)

for i in range(len(signal)):
    signalOut[i] = 1/3*(signal[i]+signal[i-1]+signal[i-2])


#Representa la señal
plt.plot(t, signalOut)
plt.title('Muneca_procesada.wav')
plt.savefig('Muneca procesada')
plt.show()

wavfile.write('muneca_procesada.wav', Fs, signalOut)

#---------------------------------------------------------------------------------
#Tarea 2
#ahora crearemos un vector de señal nuevo y crearemos un efecto de eco
#importante que todo tenga el mismo tamaño o dará error

TEMPO_BPM = 250
DELAY_AMPLITUDE = 0.5

delay_len_samples = round(1/TEMPO_BPM * 60 * Fs)
signalOutRe = np.zeros(delay_len_samples)
signalOutRe1 = np.zeros(delay_len_samples)

signalOutRe[0] = 1
signalOutRe[-1] = DELAY_AMPLITUDE

signalOutRe1[0] = 1
signalOutRe1[-1] = DELAY_AMPLITUDE

#le aplicamos convolve a la señal 
eco_signal = np.convolve(signalOutRe, signalOutRe1)

signalEco = np.convolve(signal, eco_signal) 

t = np.arange(0, len(signalEco)/Fs, 1/Fs)

# Representa la señal
plt.plot(t, signalEco)
plt.title('Muneca_eco.wav')
plt.savefig('Muneca Eco')
plt.show()

wavfile.write('muneca_eco.wav', Fs, signalEco)

#Otra manera
#--------------------------------------------------------------------------------------------
#for i in range(len(signalOutRe)):
    #if i < 1800:
        #signalOutRe[i]=signal[i]
    #elif (i>=1800)&(i<len(signal)):
        #signalOutRe[i]=0.9*(signal[i]+0.8*signalOutRe[i-1800])
    #else:
        #signalOutRe[i]= 0.9*0.8*signalOutRe[t-1800]

#------------------------------------------------------------------------------------------------
#Tarea 3 (4.1)
#cuantizacion uniforme

Fs=100
t = np.arange(0, 3, 1/Fs)
signal = np.sin(2*np.pi*t) 

delta16 = 2/16
delta8 = 2/8
delta4 = 2/4

def quantize(signal, delta):
    return delta * np.round(signal/delta)

#saturacion
#max hace que todo lo que sea mayor de 1 te lopone a uno, min todo lo menor que uno
#signalCuant = np.maximum(signalCuant, -1) 
#signalCuant = np.minimum(signalCuant, 1)

signalCuant = quantize(signal ,delta16)
plt.plot(t, signalCuant, label= '4 bits') 

signalCuant = quantize(signalCuant ,delta8)
plt.plot(t, signalCuant, label= '3 bits')

signalCuant = quantize(signalCuant ,delta4)
plt.plot(t, signalCuant, label= '2 bits') 

signalError = signalCuant - signal
plt.plot(t, signalError, label= 'error') 
plt.savefig('Muneca Cuantizada 4-3-2 bits')
plt.legend()
plt.show()


#---------------------------------------------


signalCuant4 = quantize(signal ,delta16)
signalError4 = signalCuant4 - signal
plt.plot(t, signalCuant4, label= '4 bits') 
plt.plot(t, signalError4, label= 'error') 
plt.savefig('Muneca Cuantizada 4 bits')
plt.legend()
plt.show()

signalCuant3 = quantize(signal ,delta8)
signalError3 = signalCuant3 - signal
plt.plot(t, signalCuant3, label= '3 bits') 
plt.plot(t, signalError3, label= 'error') 
plt.savefig('Muneca Cuantizada 3 bits')
plt.legend()
plt.show()

signalCuant2 = quantize(signal ,delta4)
signalError2 = signalCuant2 - signal
plt.plot(t, signalCuant2, label= '2 bits') 
plt.plot(t, signalError2, label= 'error') 
plt.savefig('Muneca Cuantizada 2 bits')
plt.legend()
plt.show()


#Ahora calculamos SNR y SNRsim
A=1 #AMPLITUD 1
Ex=A**2/2

SNR = 10*np.log10(Ex/(delta16**2/12))
print('SNR teorica para 4 bits -> ', SNR , ' db')

SNRsim = 10*np.log10(sum(signalCuant4*signalCuant4)/sum(signalError4*signalError4))
print('SNR de simulacion para 4 bits -> ', SNRsim, ' db')


SNR = 10*np.log10(Ex/(delta8**2/12))
print('SNR teorica para 3 bits -> ', SNR , ' db')

SNRsim = 10*np.log10(sum(signalCuant3*signalCuant3)/sum(signalError3*signalError3))
print('SNR de simulacion para 3 bits -> ', SNRsim, ' db')


SNR = 10*np.log10(Ex/(delta4**2/12))
print('SNR teorica para 2 bits -> ', SNR , ' db')

SNRsim = 10*np.log10(sum(signalCuant2*signalCuant2)/sum(signalError2*signalError2))
print('SNR de simulacion para 2 bits -> ', SNRsim, ' db')



#-----------------------------------------------------------------------------------------------
#Tarea 4 (4.2)
#Cuantizacion Uniforme Voz

Fs_cuant, signalCuantVoz = wavfile.read('./muneca.wav')
#signal = signal.astype(float)

delta16Voz = 16
delta64Voz = 64
delta256Voz = 256

signalCuantVoz16 = quantize(signalCuantVoz ,delta16Voz)
signalCuantVoz64 = quantize(signalCuantVoz ,delta64Voz)
signalCuantVoz256 = quantize(signalCuantVoz ,delta256Voz)

#error
signalErrorVoz16 = signalCuantVoz - signalCuantVoz16
signalErrorVoz64 = signalCuantVoz - signalCuantVoz64
signalErrorVoz256 = signalCuantVoz - signalCuantVoz256

#calculamos el SNR con las señales cuantizadas
SNRsimVoz8 = 10*np.log10(sum(signalCuantVoz16*signalCuantVoz16)/sum(signalErrorVoz16*signalErrorVoz16))
SNRsimVoz6 = 10*np.log10(sum(signalCuantVoz64*signalCuantVoz64)/sum(signalErrorVoz64*signalErrorVoz64))
SNRsimVoz4 = 10*np.log10(sum(signalCuantVoz256*signalCuantVoz256)/sum(signalErrorVoz256*signalErrorVoz256))

print('SNR cuantización señal de voz 8 bits -> ', SNRsimVoz8, ' db')
print('SNR cuantización señal de voz 6 bits -> ', SNRsimVoz6, ' db')
print('SNR cuantización señal de voz 4 bits -> ', SNRsimVoz4, ' db')

wavfile.write('muneca_cuantizada_voz8.wav', Fs_cuant, signalCuantVoz16)
wavfile.write('muneca_cuantizada_voz6.wav', Fs_cuant, signalCuantVoz64)
wavfile.write('muneca_cuantizada_voz4.wav', Fs_cuant, signalCuantVoz256)



#--------------------------------------------------------------------------------------------
#Tarea 5 (4.3)
#Cuantización con ley Mu

Mu = 255
A = 2048

delta8Mu = 16
delta6Mu = 64
delta4Mu = 256

Fs_Mu, signalCuantMu = wavfile.read('./muneca.wav')

LeyMu = A * ( ( np.log(1 + Mu / A) / np.log(1+Mu)))
LeyMuInv = ((1+Mu)**(Mu/A)-1)*(A/Mu)


signalCuantMuLey = quantize(signalCuantMu ,delta8Mu)
signalCuantMu8 = quantize(signalCuantMuLey ,LeyMuInv)

signalCuantMuLey = quantize(signalCuantMu ,delta6Mu)
signalCuantMu6 = quantize(signalCuantMuLey ,LeyMuInv)

signalCuantMuLey = quantize(signalCuantMu ,delta4Mu)
signalCuantMu4 = quantize(signalCuantMuLey ,LeyMuInv)

signalErrorMu8 = signalCuantMu - signalCuantMu8
signalErrorMu6 = signalCuantMu - signalCuantMu6
signalErrorMu4 = signalCuantMu - signalCuantMu4


SNRsimMu8 = 10*np.log10(sum(signalCuantMu8*signalCuantMu8)/sum(signalErrorMu8*signalErrorMu8))
SNRsimMu6 = 10*np.log10(sum(signalCuantMu6*signalCuantMu6)/sum(signalErrorMu6*signalErrorMu6))
SNRsimMu4 = 10*np.log10(sum(signalCuantMu4*signalCuantMu4)/sum(signalErrorMu4*signalErrorMu4))


print('SNR cuantización señal Mu 8 bits -> ', SNRsimMu8, ' db')
print('SNR cuantización señal Mu 6 bits -> ', SNRsimMu6, ' db')
print('SNR cuantización señal Mu 4 bits -> ', SNRsimMu4, ' db')

wavfile.write('muneca_cuantizada_MU8.wav', Fs_Mu, signalCuantMu8)
wavfile.write('muneca_cuantizada_MU6.wav', Fs_Mu, signalCuantMu6)
wavfile.write('muneca_cuantizada_MU4.wav', Fs_Mu, signalCuantMu4)



