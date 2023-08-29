#Autor: Miguel Carracedo Rodríguez
#21/04/2022

from cmath import pi
from blinker import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.signal as signal

#-----------------------------------------------------------
#Tarea 1

#respuesta al impulso 

x = np.zeros(30)
x[0] = 1

a = np.array([1, 0, 0.9])
b = np.array([0.3, 0.6 ,0.3])

x = np.concatenate((np.zeros(3), x))     # pre-pend ceros
y = np.zeros_like(x)
y[:3] = x[:3]                            # inicializamos las 3 primeras muestras

t = np.zeros_like(x)
t[0] = -3
t[1] = -2
t[2] = -1

for n in range(3,len(x)):
    xn = x[n:n-3:-1]                # muestras de x en orden descendente
    yn = y[n:n-3:-1]                # muestras de y en orden descendente
    y[n] = np.dot(-a,yn) + np.dot(b,xn)
    t[n] = n - 3


# Representa el Sistema IIR impulso
plt.figure()
plt.stem ( t, y)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema IIR')
plt.savefig('Tarea1_impulso')
#plt.show()


y2 = signal.lfilter(b,a,x)

# Representa el Sistema IIR  impulso con función scipy
plt.figure()
plt.stem ( t, y2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema IIR')
plt.savefig('Tarea1_impulso_scipy')
#plt.show()

#-----------------------------------------------------------------------------------

#respuesta al escalón


x = np.ones(30)
x[0] = 1

a = np.array([1, 0, 0.9])
b = np.array([0.3, 0.6 ,0.3])


x = np.concatenate((np.zeros(3), x))     # pre-pend ceros
y = np.zeros_like(x)
y[:3] = x[:3]       # inicializamos las 3 primeras muestras

t = np.zeros_like(x)
t[0] = -3
t[1] = -2
t[2] = -1

for n in range(3,len(x)):
    xn = x[n:n-3:-1]        # muestras de x en orden descendente
    yn = y[n:n-3:-1]        # muestras de y en orden descendente
    y[n] = np.dot(-a,yn) + np.dot(b,xn)
    t[n] = n - 3


# Representa el Sistema IIR  escalon
plt.figure()
plt.stem ( t, y)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema IIR')
plt.savefig('Tarea1_escalon')
#plt.show()

y2 = signal.lfilter(b,a,x)

# Representa el Sistema IIR  escalon con funcion scipy
plt.figure()
plt.stem ( t, y2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema IIR')
plt.savefig('Tarea1_escalon_scipy')
#plt.show()