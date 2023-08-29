#Autor: Miguel Carracedo Rodríguez
#28/04/2022

from cmath import pi
from blinker import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pds

%matplotlib auto    #<-- puede dar problemas fuera de visual studio code

#-----------------------------------------------------------


#Tarea 1
#respuesta al impulso 

x = np.zeros(30)
x[0] = 1

a = np.zeros(10)
a[0] = 1

b = np.ones(10)
b = b/10

pds.plot_freq_resp(b,a=1,worN = None)


#------------------------------------------------------------------

# Tarea 2

a = np.array([1, -1, 0.9])
b = np.array([1, 0, 0])
n = np.linspace(-20, 100,121)

plt.figure()
pds.impz(b,a,n)

#ahora ver si es estable
plt.figure()
pds.zplane(b,a)

#representar la magnitud
plt.figure()
pds.plot_freq_resp(b, a, worN=None)

#el escalon
plt.figure()
pds.stepz(b,a,n)

#-----------------------------------------------------------------------------------

#Tarea 3


a = np.array([1, 0.5, -0.25])
b = np.array([1, 0.5, 0])
n = np.linspace(-20, 100,121)

#respuesta impulsiva
plt.figure()
pds.impz(b,a,n)


#ahora ver si es estable (diagrama de zeros y polos)

plt.figure()
pds.zplane(b,a)


#La salida y(n) cuando la entrada es x(n)= 2 ⋅ 0.9 n u(n) 


u1 = np.zeros(20)
u2 = np.ones(101)
u = np.concatenate((u1, u2))
x = 2 * 0.9**n * u

y = signal.lfilter(b,a,x)


# Representa el Sistema IIR impulso
plt.figure()
plt.stem ( y )
plt.xlabel('Tiempo muestreado')
plt.ylabel('Amplitud')
plt.title('Salida Y')
#plt.savefig('Tarea3')
#plt.show()

#------------------------------------------------------------------

#Tarea 4

a = np.array([1, -0.9, 0.81])
b = np.array([1, 1, 0])
n = np.linspace(-20, 100,121)

#respuesta impultiva
plt.figure()
pds.plot_freq_resp(b, a, worN=None)
plt.axvline(x = np.pi, color='black', label = 'pi/3')
plt.axvline(x = (np.pi / 3), color='orange', label = 'pi')



#ahora vamos a hacer la señal


n = np.linspace(0, 199,200)

x = np.sin((np.pi*n)/3) + 5 * np.cos(np.pi * n)

y = signal.lfilter(b,a,x)

# Representa el Sistema IIR impulso
plt.figure()
plt.plot ( y )
plt.xlabel('Tiempo muestreado')
plt.ylabel('Amplitud')
plt.title('Salida Y')
#plt.savefig('Tarea4')
#plt.show()

#------------------------------------------------------------------

#Tarea 5
import pds

#magnitud y fase
h = np.array([0.0023, 0.0053 ,0.0411, -0.1233 ,-0.2310 ,0.3087 ,0.3087 ,-0.2310 ,-0.1233 ,0.0411 ,0.0053 ,0.0023])
plt.figure()
pds.plot_freq_resp(h)


#respuesta al escalon unitario
plt.figure()
n = np.linspace(-20,40,61) 
pds.stepz(h,1,n)
plt.grid()

#que retardo introduce el filtro?
plt.figure()
pds.plot_group_delay(h,1)







