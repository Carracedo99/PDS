#Autor: Miguel Carracedo Rodríguez
#21/04/2022

from cmath import pi
from blinker import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.signal as signal

#-----------------------------------------------------------
#Tarea 3

#a =m[ 1, 0 0 0 0 hasta 10]
#b = [0.1, 0.1 0.1] hasta 10
# y luego meterle 10 ceros delante a la x y a la y

#respuesta al impulso 

x = np.zeros(30)
x[0] = 1

a = np.zeros(10)
a[0] = 1

b = np.ones(10)
b = b/10

x = np.concatenate((np.zeros(10), x))     # pre-pend ceros
y = np.zeros_like(x)
y[:10] = x[:10]       # inicializamos las 3 primeras muestras

t = np.zeros_like(x)

t[0] = -3
t[1] = -2
t[2] = -1
t[3] = -3
t[4] = -2
t[5] = -1
t[6] = -3
t[7] = -2
t[8] = -1
t[9] = -1

for n in range(10,len(x)):
    xn = x[n:n-10:-1]        # muestras de x en orden descendente
    yn = y[n:n-10:-1]        # muestras de y en orden descendente
    y[n] = np.dot(-a,yn) + np.dot(b,xn)
    t[n] = n - 10



# Representa el Sistema FIR impulso
plt.figure()
plt.stem ( t, y)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_impulso')
#plt.show()

y2 = signal.lfilter(b,a,x)

# Representa el Sistema FIR impulso con funcion scipy
plt.figure()
plt.stem ( t, y2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_impuso_scipy')
#plt.show()

#------------------------------------------------------------------

# respuesta al escalon 

x = np.ones(30)
x[0] = 1


a = np.zeros(10)
a[0] = 1

b = np.ones(10)
b = b/10


x = np.concatenate((np.zeros(10), x))     # pre-pend ceros
y = np.zeros_like(x)
y[:10] = x[:10]       # inicializamos las 3 primeras muestras

t = np.zeros_like(x)

t[0] = -3
t[1] = -2
t[2] = -1
t[3] = -3
t[4] = -2
t[5] = -1
t[6] = -3
t[7] = -2
t[8] = -1
t[9] = -1

for n in range(10,len(x)):
    xn = x[n:n-10:-1]        # muestras de x en orden descendente
    yn = y[n:n-10:-1]        # muestras de y en orden descendente
    y[n] = np.dot(-a,yn) + np.dot(b,xn)
    t[n] = n - 10



# Representa el Sistema FIR escalon
plt.figure()
plt.stem ( t, y)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_escalon')
#plt.show()

y2 = signal.lfilter(b,a,x)

# Representa el Sistema FIR escalon con funcion scipy
plt.figure()
plt.stem ( t, y2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_escalon_scipy')
#plt.show()


#-----------------------------------------------------------------------------------

#ultimo apartado de la taera 3

#filter nos va a dar 
#y3 = signal.convolve(y,x)

x = np.zeros(30)
x[0] = 1

a = np.zeros(10)
a[0] = 1

b = np.ones(10)
b = b/10

x = np.concatenate((np.zeros(10), x))     # pre-pend ceros
y = np.zeros_like(x)
y[:10] = x[:10]       # inicializamos las 3 primeras muestras

t = np.zeros_like(x)

t[0] = -3
t[1] = -2
t[2] = -1
t[3] = -3
t[4] = -2
t[5] = -1
t[6] = -3
t[7] = -2
t[8] = -1
t[9] = -1

for n in range(10,len(x)):
    xn = x[n:n-10:-1]        # muestras de x en orden descendente
    yn = y[n:n-10:-1]        # muestras de y en orden descendente
    y[n] = np.dot(-a,yn) + np.dot(b,xn)
    t[n] = n - 10



# Representa salida 1
plt.figure()
plt.stem ( t, y)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_concolve1')
#plt.show()

y2 = signal.lfilter(b,a,x)
h = y2.copy()


# Representa salida 2
plt.figure()
plt.stem ( t, y2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_concolve2')
#plt.show()

y3 = signal.convolve(h,x)

# Representa salida 3
plt.figure()
plt.stem ( y3)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sistema FIR')
plt.savefig('Tarea3_concolve3')
#plt.show()