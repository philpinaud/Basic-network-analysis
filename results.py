
from __future__ import division
import scipy.stats
from scipy import linspace
from math import *
from numpy import *
from matplotlib import rc, rcParams
from pylab import normpdf
from pylab import *
from numpy import poly1d
import numpy as np
import pylab as pyl
import sys						#para argumentos con lineas de comando



###############LECTURA DE ARCHIVO############################
csv 				= 'DATA_ServidorWIFI95RTC0' + '.csv' 		
#EN NETEM ORDENAR LAS 3 PRIMERAS COLUMNAS EN BASE A COLUMNA C				

################SELECCION DE DATOS###########################
####################INTERVALOS###############################
Escala 				= 2																	#3 para 800 datos / 2 para 400 datos / 1 en 1 segundo / 0 en 0.5 segundos
Resolucion			= 0																	#1 para 100 ms / 0 para 10 ms

###############INTERVALOS DE TIEMPO##########################
#(GRAFICO 5) Numeros de intervalos de segundos para muestreo de tiempo entre arribos
Intervalos_seg_A 	= np.array([50, 54, 65, 86]) 
#(GRAFICO 5) Numeros de intervalos de segundos para Numero de eventos por tiempo
Intervalos_seg_B 	= np.array([50, 54, 65, 86]) 

###############INTERVALOS DE DATOS###########################
#(GRAFICO 5) Numeros de intervalos de segundos para muestreo de tiempo entre arribos
Intervalos_seg_C 	= np.array([17000, 17913, 32738, 43000]) 
#(GRAFICO 5) Numeros de intervalos de segundos para Numero de eventos por tiempo
Intervalos_seg_D 	= np.array([17000, 17913, 32738, 43000])  

##############################################################
#Escala de los graficos
opcion_graph		= 2																	#2 real / 1 normalizada / 0 logaritmica 
##############################################################
##############################################################
datospgrupo = 200#800 cableado																		#Defecto 1000
#(GRAFICO 2) Acotar espacio para calculo de histograma GENERAL de eventos por segundo
val_min 			= 5
val_max 			= 200
																

#############CORRECCIONES MANUALES DE LA MEDICION#############
Intv_dly_min		= 1400																#Extraer valores de una zona estable para calcular la media
Intv_dly_max		= 2400 
time_ping 			= 0.560 * (10**-3)   												#WIRE 389 WIFI 560
BW_RED				= 10
offset				= 0.0001375

##############################################################
##############################################################
##############################################################
##############################################################
#SIN USO
#Datos en intervalos para histogramas de la muestra
Intervalos	= 20 																		#ideal=20
Valor_minH	= None 																		#Posicion inicial de las muestras
Valor_maxH 	= None 																		#Posicion final de las muestras
umbral_inf 	= 1#20000 																	#Por defecto=1
umbral_sup 	= 100000#30000   															#Por defecto=200000
ejex_min	= None
ejex_max 	= None
#Busqueda de encolamiento en buffers (valores normalizados de los histogramas)
buscado_min	= 0.0  	#31920 / 31950
buscado_max = 0.0000000003#0.00000000025 	#31950 / 32000

##############################################################
##############################################################
##############################################################
############################################################## 
plt.style.use('ggplot')

###############################################################################################################
###############################################FUNCIONES#######################################################
###############################################################################################################

def bandwidth():
	long_array 	= len(timestamp)	
	datos 		= int(long_array)																				#cantidad de datos para trabajar
	print 'Cantidad de datos 	=', datos

	pkt_rcv		= np.zeros(datos - 1)																			#-1 para evitar ultimo dato sin valores
	time_bw		= np.zeros(datos - 1)
	time_elapsed	= np.zeros(datos - 1)
	stmp_rcv	= np.zeros((datos - 1,3))

	n = 0
	p = 0
	q = 0
	for z in range(1,datos):	
		#separar info
		pkt_rcv[p] 			= (1497*8)																			#(MTU * bits)
		stmp_rcv[p,q] 		= rcvstamp[n]	  																	#columna 1 tiempo de recepcion
		stmp_rcv[p,q+1]		= rcvstamp[n + 1]																	#columna 2 tiempo de recepcion + 1 segundo
		stmp_rcv[p,q+2]		= rcvstamp[0]  																		#columna 3 tiempo de recepcion inicial

		#calcular tiempo 
		time_elapsed[p]		= stmp_rcv[p,q] - stmp_rcv[p,q+2]													#tiempo transcurrido 
		time_bw[p] 			= stmp_rcv[p,q+1] - stmp_rcv[p,q]													#tiempo transcurrido en cada salto

		p += 1
		n += 1
		z += 1

	time_bw			= 1/time_bw																					#Se busca multiplicador para equipararlo a 1 seg (para expresar en per second)
	pkt_rcv			= pkt_rcv * time_bw																			#Se multiplican los paquetes para obtener bps
	real_bw			= pkt_rcv / (10**6)																			#Se dividen los bps para obtener mbps
	real_time		= time_elapsed

	return(real_bw)

###############################################################################################################

def client_bandwidth_estimation(offset):
	tiempo_transcurrido_client_netem	= t_salida - t_salida[0]												#tiempo transcurrido en el cliente

	#Paso 1
	timesleep_recuperado			= timesleep + offset														#Al timesleep se le resta el tiempo de ejecucion del programa
	#Paso 2
	timesleep_recuperado			= timesleep_recuperado**(-1)												#Se transforma de timesleep a bandwidth
	timesleep_recuperado			= (timesleep_recuperado*1497)*8
	bandwidth_recuperado			= (timesleep_recuperado)/(10**6)

	#Adaptacion del largo de los vectores para que coincidan con los vectores de Bandwidth Server
	t_e_c_netem	= np.zeros(long_array)
	t_s_c_netem	= np.zeros(long_array)

	n = 0
	p = 0

	for z in range(0,long_array):	
		t_e_c_netem[p]	= tiempo_transcurrido_client_netem[n]	
		t_s_c_netem[p]	= bandwidth_recuperado[n]  			
		n += 1
		p += 1
		z += 1
	return t_e_c_netem, t_s_c_netem


###############################################################################################################
###############################################LECTURA DE ARCHIVOS#############################################
###############################################################################################################

filecsv = np.loadtxt(csv, delimiter=',')
try:
	t_salida   	= filecsv[0:,0]
	timesleep  	= filecsv[0:,1]
	pkt_salido 	= filecsv[0:,2]
	timestamp  	= filecsv[0:,3]
	rcvstamp   	= filecsv[0:,4]
	totalbytes 	= filecsv[0:,5]
	pktrcv     	= filecsv[0:,6]
	print "Lectura de Datos	= Filas", len(t_salida), "- Columnas 7"
	
except:
	print "Error de Lectura de Archivos"

###############################################################################################################
###############################################CREACION DE ARRAYS##############################################
###############################################################################################################

long_array = len(t_salida)

columna1 = np.zeros([long_array],float)																			#almacena tiempos de llegada al servidor
columna2 = np.zeros([long_array],float)																			#almacena tiempos de llegada entre datagramas al servidor
columna4 = np.zeros([long_array],float)																			#almacena valores de delay
columna5 = np.zeros([long_array],float)																			#almacena paquetes perdidos

try:
	for n in range(0,long_array,1):	
		columna1[n] = rcvstamp[n] - timestamp[n]
		columna4[n] = rcvstamp[n] - t_salida[n]
		columna5[n] = pktrcv[n] - pkt_salido[n]

	columna2[0] = columna1[0]																					#se asigna valor inicial (tiempo de llegada primer datagrama)

	for n in range(1,long_array,1):	
		columna2[n] = rcvstamp[n] - rcvstamp[n-1]
	
except:
	pass

##############################################CORRECCIONES#####################################################
#########################################Sincronia entre equipos###############################################
###############################################################################################################

media_delay = np.mean(columna4[Intv_dly_min:Intv_dly_max])

Dif_tiempo 	= time_ping - media_delay
if (media_delay < 1):
	columna4 = columna4 - (media_delay) + time_ping
else:
	columna4 = columna4 + time_ping - (media_delay)


#################################################FIGURA  1#####################################################
##########################################GRAFICO 1: Throughput################################################
plt.figure(1)##################################################################################################

real_bw 	= bandwidth()

a 			= real_bw[len(real_bw)-1]																			#repetir ultimo elemento del array
real_bw		= np.append(real_bw, a)																				#incorporar al array
																												#calculo de troughput por segundo
residual    = 0																									
sumatoria	= 0
indices		= np.zeros([long_array],float)
for n in range(0, len(real_bw)):																				#revisa datos tiempos entre datagramas, al encontrar suma superior a 1 en varios indices, almacena el valor 1 en esa posicion
	sumatoria	= sumatoria + columna2[n]
	if (sumatoria >= 1):
		residual  	= sumatoria																					#Evita diferencias
		sumatoria	= residual - 1       
		indices[n]	= 1
	else:
		pass
	
conteo 				= 0
prom_bw				= np.copy(real_bw)
indices[len(indices)-1]		= 1																					#se hace el ultimo valor 1 para que no queden datos sin procesar
for n in range(0, len(real_bw)):
	if (indices[n] == 1):
		w 			= np.mean(real_bw[conteo: n])
		prom_bw[conteo: n]	= w
		conteo 			= n
	else:
		pass


WAN_BW 		= np.ones([long_array],float)
b		= np.mean(BW_RED)
WAN_BW 		= WAN_BW * b

plt.subplot(511)
x = np.arange(1, long_array+1, 1)																				#Array con la cantidad total de mediciones realizadas
plt.plot(x, real_bw, color= 'blue', alpha=0.30,)
plt.plot(x, prom_bw, 'k-', linewidth=1.0)
#plt.text(100, b + 8, 'WAN BW')
plt.plot(x, WAN_BW,'r--', linewidth=2.0)
plt.xlabel('Numero de Medicion')
plt.xlim(0,len(x))
plt.ylabel('Troughput de recepcion')
#plt.ylim(0,40)
#plt.title('Grafico Tiempo de Servicio')
plt.grid(True)
#plt.savefig("Cliente.png")

#################################################FIGURA  1#####################################################
#############################################GRAFICO 2: Delay##################################################

plt.subplot(512)
#plt.plot(x, columna4, 'ro')
#plt.plot(x, columna4, 'b*-')
plt.plot(x, columna4, color= 'blue')
plt.xlabel('Numero de Medicion')
plt.xlim(0,len(x))
#plt.ylim(0,0.20)
plt.ylabel('Delay')
#plt.title('Grafico Tiempo de Servicio')
plt.grid(True)

#################################################FIGURA  1#####################################################
#######################################GRAFICO 3: Perdida de Paquetes##########################################

plt.subplot(513)
plt.plot(x, columna5, color= 'blue')
plt.xlabel('Numero de Medicion')
plt.xlim(0,len(x))
plt.ylabel('Perdida de Paquetes')
#plt.ylim(-50000,10000)
#plt.title('Grafico Tiempo de Servicio')
plt.grid(True)
try:
	first_loss 	= int(min(argwhere(columna5)))																	#Localizar primera perdida 
except:
	first_loss 	= 0

if (first_loss == 0):
	print 'Primera Perdida 	= Sin Perdidas'
else:
	print 'Primera Perdida 	=', first_loss

plt.plot(x[first_loss],columna5[first_loss], 'g^',linewidth=10.0)

#################################################FIGURA  1#####################################################
#####################################GRAFICO 4: Tiempo entre datagramas########################################

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

Interpacket_gap = np.ones([long_array],float)																	#Marca para tiempo de procesamiento entre paquetes
Interpacket_gap = Interpacket_gap * 0.96*(10**(-6))

media 	= np.ones([long_array],float)																			#Calculo de la media del tiempo entre datagramas
media 	= media * np.mean(columna2)

y = columna2


plt.subplot(514)
plt.plot(x, y,  color= 'blue')
plt.plot(x, media, 'r--', linewidth=2.0)
plt.plot(x, Interpacket_gap, 'g--', linewidth=2.0)
plt.xlabel('Numero de Medicion')
plt.xlim(0,len(x))
plt.ylabel('Tiempo entre datagramas')
#plt.title('Grafico Tiempo de Servicio')
plt.grid(True)

#################################################FIGURA  1#####################################################
#############################################GRAFICO 5: Load###################################################

plt.subplot(515)

Tiempo_cliente, BW_cliente	= client_bandwidth_estimation(offset)												#Estimacion BW Cliente

#borde = np.ones(100) 
#alto = np.array(np.linspace(0,0.7,100))
#plt.plot(borde, alto, 'r--')

Rho 				= BW_cliente/BW_RED
plt.plot(Rho, columna4,'g^', alpha=0.05)
#plt.plot(Rho[15000:20000], columna4[15000:20000],'b^')
plt.xlabel('Rho')
#plt.xlim(0,len(x))
plt.ylabel('Delay')
#plt.ylim(-50000,10000)
#plt.title('Grafico Tiempo de Servicio')
plt.grid(True)


#################################################FIGURA  2#####################################################
###################################GRAFICO 5: Numero de eventos por tiempo#####################################
#plt.figure(2)#################################################################################################

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

sumatoria_A, sumatoria_B, sumatoria_C	= 0, 0, 0

#######################################CREACION DE INTERVALOS DE TIEMPO########################################
##############################################En 1 segundo#####################################################

residual = 0
indices_A		= np.zeros([long_array],float)
for n in range(0, len(pktrcv)):																					#revisa datos tiempos entre datagramas, al encontrar suma superior a 1 en varios indices, almacena el valor 1 en esa posicion
	sumatoria_A	= sumatoria_A + columna2[n]
	if (sumatoria_A >= 1):		      
		residual  	= sumatoria_A																				#Evita diferencias
		sumatoria_A	= residual - 1       
		indices_A[n]	= 1
	else:
		pass

#Calculo de paquetes por tiempo
pkt_seg_A 			= np.array([])
#interarrival_seg_A 	= np.array([])
conteo 				= 0
indices_A[len(indices_A)-1]		= 1																				#se hace el ultimo valor 1 para que no queden datos sin procesar
for n in range(0, len(columna2)):
	if (indices_A[n]== 1):
		ww 			= len(columna2[conteo: n])																	#parte 1 (cantidad de eventos por tiempo)
		pkt_seg_A 	= np.append(pkt_seg_A, ww)
#		www					= np.mean(columna2[conteo: n])														#parte 2 (tiempo entre eventos en intervalo de tiempo)
#		interarrival_seg_A 	= np.append(interarrival_seg_A, www)

		conteo 		= n
	else :
		pass

#######################################CREACION DE INTERVALOS DE TIEMPO########################################
#############################################En 100 ms=0.1#####################################################

residual = 0
indices_B		= np.zeros([long_array],float)
for n in range(0, len(pktrcv)):					
	sumatoria_B	= sumatoria_B + columna2[n]
	if (sumatoria_B >= 0.1):		
		residual  	= sumatoria_B							
		sumatoria_B	= residual - 0.1       
		indices_B[n]	= 1
	else:
		pass

pkt_seg_B 			= np.array([])
conteo 				= 0
indices_B[len(indices_B)-1]		= 1					
for n in range(0, len(columna2)):
	if (indices_B[n]== 1):
		ww 			= len(columna2[conteo: n]) 						
		pkt_seg_B 	= np.append(pkt_seg_B, ww)
		conteo 		= n
	else :
		pass

#######################################CREACION DE INTERVALOS DE TIEMPO########################################
#############################################En 10 ms=0.01#####################################################

residual = 0
indices_C		= np.zeros([long_array],float)
for n in range(0, len(pktrcv)):					
	sumatoria_C	= sumatoria_C + columna2[n]
	if (sumatoria_C >= 0.01):		  
		residual  	= sumatoria_C							
		sumatoria_C	= residual - 0.01       
		indices_C[n]	= 1
	else:
		pass

pkt_seg_C 			= np.array([])
conteo 				= 0
indices_C[len(indices_C)-1]		= 1					
for n in range(0, len(columna2)):
	if (indices_C[n]== 1):
		ww 			= len(columna2[conteo: n])						
		pkt_seg_C 	= np.append(pkt_seg_C, ww)
		conteo 		= n
	else :
		pass

################################################GRAFICOS#######################################################
##################################NUMERO DE EVENTOS POR TIEMPO (GENERAL)#######################################

import scipy as sp

#plt.subplot(311)
#eje_x_A 	= sp.arange(len(pkt_seg_A))
#plt.bar(eje_x_A,pkt_seg_A, width=0.1)
#plt.ylim(0,max(pkt_seg_A)*1.4)
#plt.xlabel('Tiempo (1 segundo)')
#plt.ylabel('Cantidad de datagramas')

#plt.subplot(312)
#eje_x_B 	= sp.arange(len(pkt_seg_B))
#plt.bar(eje_x_B,pkt_seg_B, width=0.1)
#plt.ylim(0,max(pkt_seg_B)*1.4)
#plt.xlabel('Tiempo (100 milisegundos)')
#plt.ylabel('Cantidad de datagramas')

#plt.subplot(313)
#eje_x_C 	= sp.arange(len(pkt_seg_C))
#plt.bar(eje_x_C,pkt_seg_C)
#plt.ylim(0,max(pkt_seg_C)*1.4)
#plt.xlabel('Tiempo (10 milisegundos)')
#plt.ylabel('Cantidad de datagramas')

#######################################CREACION DE INTERVALOS DE TIEMPO########################################
###########################################EN 0.5 SEGUNDOS#####################################################

sumatoria_D	= 0

residual = 0
indices_D		= np.zeros([long_array],float)
for n in range(0, len(pktrcv)):					
	sumatoria_D	= sumatoria_D + columna2[n]
	if (sumatoria_D >= 0.5):		      
		residual  	= sumatoria_D							
		sumatoria_D	= residual - 0.5       
		indices_D[n]	= 1
	else:
		pass

pkt_seg_D 			= np.array([])
conteo 				= 0
indices_D[len(indices_D)-1]		= 1									
for n in range(0, len(columna2)):
	if (indices_D[n]== 1):
		ww 			= len(columna2[conteo: n])						
		pkt_seg_D 	= np.append(pkt_seg_D, ww)
		conteo 		= n
	else :
		pass


#################################################FIGURA  3#####################################################
###########################GRAFICO 6: Mean packet interarrival time por tiempo#################################

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

#plt.figure(3)

import scipy as sp

#plt.subplot(311)
#eje_x_A 	= sp.arange(len(interarrival_seg_A))
#plt.bar(eje_x_A,interarrival_seg_A, width=0.1)
#plt.ylim(0,max(interarrival_seg_A)*1.4)
#plt.xlabel('Tiempo (1 segundo)')
#plt.ylabel('Cantidad de datagramas')

#plt.subplot(312)
#eje_x_B 	= sp.arange(len(interarrival_seg_B))
#plt.bar(eje_x_B,interarrival_seg_B, width=0.1)
#plt.ylim(0,max(interarrival_seg_B)*1.4)
#plt.xlabel('Tiempo (100 milisegundos)')
#plt.ylabel('Cantidad de datagramas')

#plt.subplot(313)
#eje_x_C 	= sp.arange(len(interarrival_seg_C))
#plt.bar(eje_x_C,interarrival_seg_C)
#plt.ylim(0,max(interarrival_seg_C)*1.4)
#plt.xlabel('Tiempo (10 milisegundos)')
#plt.ylabel('Cantidad de datagramas')

###############################################################################################################
#################################################FIGURA 4######################################################
plt.figure(4)##################################################################################################

###################################PARA ANALISIS EN 1 O 0.5 SEGUNDOS###########################################
Segundos = np.array([])

if (Escala == 1):
	p = 1
	for n in range(0, len(indices_A)):																			#se crea array cuyos elementos corresponden a los segundos en forma discreta
		if (indices_A[n]!=1):
			Segundos = np.append(Segundos, p)
		else:
			Segundos = np.append(Segundos, p)
			p = p + 1

elif (Escala == 0):
	p = 0.5
	for n in range(0, len(indices_D)):																			#se crea array cuyos elementos corresponden a los segundos en forma discreta
		if (indices_D[n]!=1):
			Segundos = np.append(Segundos, p)
		else:
			Segundos = np.append(Segundos, p)
			p = p + 0.5
else:
	pass

######################################GRAFICO 7: Tiempo entre eventos##########################################
##########################################GRUPO 1##############################################################
import scipy.stats as st
distributions = [st.uniform, st.expon, st.norm]																	#Distribuciones continuas mas usadas (solo las continuas tienen fit)


plt.subplot(341)
if (Escala == 0 or Escala == 1):
	itemindex_B1 		= np.nonzero(Segundos==Intervalos_seg_A[0])   											#busca los indices para un valor especifico entregando una matriz
	Muestra_B1 			= columna2[min(itemindex_B1[:][0]): max(itemindex_B1[:][0])]							#identifica el indice min y max que coincide con el valor buscado
elif (Escala == 2):
	itemindex_B1 		= np.nonzero(x==Intervalos_seg_C[0])   											
	Muestra_B1 			= columna2[int(itemindex_B1[:][0])-400: int(itemindex_B1[:][0])]						#identifica el indice min y max que coincide con el valor buscado en un grupo de 400 datos
elif (Escala == 3):
	itemindex_B1 		= np.nonzero(x==Intervalos_seg_C[0])   											
	Muestra_B1 			= columna2[int(itemindex_B1[:][0])-800: int(itemindex_B1[:][0])]						#identifica el indice min y max que coincide con el valor buscado en un grupo de 800 datos
else:
	print 'Error en valor de seleccion de Escala'

mu_H 				= np.mean(Muestra_B1)																		#Media
sigma_H 			= np.std(Muestra_B1)																		#Desviacion estadar
bins_H				= int(np.sqrt(len(Muestra_B1)))
weights 			= np.ones_like(Muestra_B1)/len(Muestra_B1)

if (opcion_graph==0):
	n, bins, patches 	= pyl.hist(Muestra_B1, weights=weights, log=True, facecolor='green', alpha=0.50, label=['Real'])	#Otra opcion: plt.yscale('log', nonposy='clip')
elif (opcion_graph==1):
	n, bins, patches 	= pyl.hist(Muestra_B1, weights=weights, facecolor='green', alpha=0.50, label=['Real'])
elif (opcion_graph==2):
	n, bins, patches 	= pyl.hist(Muestra_B1, bins_H, normed=True, facecolor='green', alpha=0.50, label=['Real'])
else:
	print 'Error en escala de graficos'
plt.xlabel('Tiempo entre eventos en 1 seg (grupo 1)')
plt.ylabel('Frecuencia')
plt.grid(True)

mles_Muestra_B1 = []

for distribution in distributions:
    pars = distribution.fit(Muestra_B1)
    mle_Muestra_B1 = distribution.nnlf(pars, Muestra_B1)
    mles_Muestra_B1.append(mle_Muestra_B1)

results = [(distribution.name, mle_Muestra_B1) for distribution, mle_Muestra_B1 in zip(distributions, mles_Muestra_B1)]
best_fit = sorted(zip(distributions, mles_Muestra_B1), key=lambda d: d[1])[0]
print 'TIEMPO ENTRE EVENTOS'
print 'Grupo 1: Mejor ajuste alcanzado en {}, valor MLE : {}'.format(best_fit[0].name, best_fit[1])

from scipy import stats  
import matplotlib.patches as mpatches

xt = plt.xticks()[0]  																							#entrega informacion del eje x que abarcan los datos
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(Muestra_B1))
a = str(best_fit[0].name) 

if (opcion_graph==2):
	if (a == 'norm'):
		m, s 		= stats.norm.fit(Muestra_B1) 																#get mean and standard deviation  
		pdf_g 		= stats.norm.pdf(lnspc, m, s) 																#now get theoretical values in our interval 
		red_patch 	= mpatches.Patch(color='red', label='normal')
		plt.legend(handles=[red_patch], prop={'size':10})
		coeff 		= str(scipy.stats.variation(Muestra_B1))
		plt.annotate(r'Coeficiente de Variacion = ' + coeff, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		plt.plot(lnspc, pdf_g, color='red') 																	#plot it
		distrib_B1 	= 1

	elif (a == 'uniform'):
		m, s 		= stats.uniform.fit(Muestra_B1)  
		pdf_g 		= stats.uniform.pdf(lnspc, m, s)  
		red_patch 	= mpatches.Patch(color='red', label='uniforme')
		plt.legend(handles=[red_patch])
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B1 	= 2

	elif (a == 'expon'):
		m, s 		= stats.expon.fit(Muestra_B1) 
		pdf_g 		= stats.expon.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='exponencial')
		plt.legend(handles=[red_patch])
		kurtosis 	= str(scipy.stats.kurtosis(Muestra_B1))
		plt.annotate(r'Kurtosis = ' + kurtosis, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		skewness 	= str(scipy.stats.skew(Muestra_B1))
		plt.annotate(r'Skewness = ' + skewness, xy=(0.02, 0.92), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')   
		plt.plot(lnspc, pdf_g, color='red')
		distrib_B1 	= 3

	else: 
		print "error en distribucion grupo 1"
else:
	pass

######################################GRAFICO 7: Tiempo entre eventos##########################################
##########################################GRUPO 2##############################################################

plt.subplot(342)
if (Escala == 0 or Escala == 1):
	itemindex_B2 		= np.nonzero(Segundos==Intervalos_seg_A[1])   											
	Muestra_B2 			= columna2[min(itemindex_B2[:][0]): max(itemindex_B2[:][0])]							
elif (Escala == 2):
	itemindex_B2 		= np.nonzero(x==Intervalos_seg_C[1])   											
	Muestra_B2 			= columna2[int(itemindex_B2[:][0])-400: int(itemindex_B2[:][0])]						
elif (Escala == 3):
	itemindex_B2 		= np.nonzero(x==Intervalos_seg_C[1])   											
	Muestra_B2 			= columna2[int(itemindex_B2[:][0])-800: int(itemindex_B2[:][0])]						
else:
	print 'Error en valor de seleccion de Escala'	

mu_H 				= np.mean(Muestra_B2)		
sigma_H 			= np.std(Muestra_B2)																							
bins_H				= int(np.sqrt(len(Muestra_B2)))
weights 			= np.ones_like(Muestra_B2)/len(Muestra_B2)
if (opcion_graph==0):
	n, bins, patches 	= pyl.hist(Muestra_B2, weights=weights, log=True, facecolor='slateblue', alpha=0.50, label=['Real'])
elif (opcion_graph==1):
	n, bins, patches 	= pyl.hist(Muestra_B2, weights=weights, facecolor='slateblue', alpha=0.50, label=['Real'])
elif (opcion_graph==2):
	n, bins, patches 	= pyl.hist(Muestra_B2, bins_H, normed=True, facecolor='slateblue', alpha=0.50, label=['Real'])
else:
	print 'Error en escala de graficos'
plt.xlabel('Tiempo entre eventos en 1 seg (grupo 2)')
plt.ylabel('Frecuencia')
plt.grid(True)

mles_Muestra_B2 = []

for distribution in distributions:
    pars = distribution.fit(Muestra_B2)
    mle_Muestra_B2 = distribution.nnlf(pars, Muestra_B2)
    mles_Muestra_B2.append(mle_Muestra_B2)

results = [(distribution.name, mle_Muestra_B2) for distribution, mle_Muestra_B2 in zip(distributions, mles_Muestra_B2)]
best_fit = sorted(zip(distributions, mles_Muestra_B2), key=lambda d: d[1])[0]
print 'Grupo 2: Mejor ajuste alcanzado en {}, valor MLE : {}'.format(best_fit[0].name, best_fit[1])
a = str(best_fit[0].name) 

from scipy import stats  
import matplotlib.patches as mpatches

xt = plt.xticks()[0]  			
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(Muestra_B2))

if (opcion_graph==2):
	if (a == 'norm'):
		m, s 		= stats.norm.fit(Muestra_B2) 																		 
		pdf_g 		= stats.norm.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='normal')
		plt.legend(handles=[red_patch], prop={'size':10})
		coeff 		= str(scipy.stats.variation(Muestra_B2))
		plt.annotate(r'Coeficiente de Variacion = ' + coeff, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		plt.plot(lnspc, pdf_g, color='red') 	
		distrib_B2 	= 1													

	elif (a == 'uniform'):
		m, s 		= stats.uniform.fit(Muestra_B2)  
		pdf_g 		= stats.uniform.pdf(lnspc, m, s)  
		red_patch 	= mpatches.Patch(color='red', label='uniforme')
		plt.legend(handles=[red_patch])
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B2 	= 2

	elif (a == 'expon'):
		m, s 		= stats.expon.fit(Muestra_B2) 
		pdf_g 		= stats.expon.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='exponencial')
		plt.legend(handles=[red_patch])
		kurtosis 	= str(scipy.stats.kurtosis(Muestra_B2))
		plt.annotate(r'Kurtosis = ' + kurtosis, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		skewness 	= str(scipy.stats.skew(Muestra_B2))
		plt.annotate(r'Skewness = ' + skewness, xy=(0.02, 0.92), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top') 
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B2 	= 3

	else: 
		print "error en distribucion grupo 2"
else:
	pass

######################################GRAFICO 7: Tiempo entre eventos##########################################
##########################################GRUPO 3##############################################################

plt.subplot(343)
if (Escala == 0 or Escala == 1):
	itemindex_B3 		= np.nonzero(Segundos==Intervalos_seg_A[2])   											
	Muestra_B3 			= columna2[min(itemindex_B3[:][0]): max(itemindex_B3[:][0])]							
elif (Escala == 2):
	itemindex_B3 		= np.nonzero(x==Intervalos_seg_C[2])   											
	Muestra_B3 			= columna2[int(itemindex_B3[:][0])-400: int(itemindex_B3[:][0])]						
elif (Escala == 3):
	itemindex_B3 		= np.nonzero(x==Intervalos_seg_C[2])   											
	Muestra_B3 			= columna2[int(itemindex_B3[:][0])-800: int(itemindex_B3[:][0])]						
else:
	print 'Error en valor de seleccion de Escala'	

mu_H 				= np.mean(Muestra_B3)								
sigma_H 			= np.std(Muestra_B3)								
bins_H				= int(np.sqrt(len(Muestra_B3)))
weights 			= np.ones_like(Muestra_B3)/len(Muestra_B3)
if (opcion_graph==0):
	n, bins, patches 	= pyl.hist(Muestra_B3, weights=weights, log=True, facecolor='red', alpha=0.50, label=['Real'])
elif (opcion_graph==1):
	n, bins, patches 	= pyl.hist(Muestra_B3, weights=weights, facecolor='red', alpha=0.50, label=['Real'])
elif (opcion_graph==2):
	n, bins, patches 	= pyl.hist(Muestra_B3, bins_H, normed=True, facecolor='red', alpha=0.50, label=['Real'])
else:
	print 'Error en escala de graficos'
plt.xlabel('Tiempo entre eventos en 1 seg (grupo 3)')
plt.ylabel('Frecuencia')
plt.grid(True)

mles_Muestra_B3 = []

for distribution in distributions:
    pars = distribution.fit(Muestra_B3)
    mle_Muestra_B3 = distribution.nnlf(pars, Muestra_B3)
    mles_Muestra_B3.append(mle_Muestra_B3)

results = [(distribution.name, mle_Muestra_B3) for distribution, mle_Muestra_B3 in zip(distributions, mles_Muestra_B3)]
best_fit = sorted(zip(distributions, mles_Muestra_B3), key=lambda d: d[1])[0]
print 'Grupo 3: Mejor ajuste alcanzado en {}, valor MLE : {}'.format(best_fit[0].name, best_fit[1])
a = str(best_fit[0].name) 

from scipy import stats  
import matplotlib.patches as mpatches

xt = plt.xticks()[0]  			
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(Muestra_B3))

if (opcion_graph==2):
	if (a == 'norm'):
		m, s 		= stats.norm.fit(Muestra_B3) 																		 
		pdf_g 		= stats.norm.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='normal')
		plt.legend(handles=[red_patch], prop={'size':10})
		coeff 		= str(scipy.stats.variation(Muestra_B3))
		plt.annotate(r'Coeficiente de Variacion = ' + coeff, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B3 	= 1														

	elif (a == 'uniform'):
		m, s 		= stats.uniform.fit(Muestra_B3)  
		pdf_g 		= stats.uniform.pdf(lnspc, m, s)  
		red_patch 	= mpatches.Patch(color='red', label='uniforme')
		plt.legend(handles=[red_patch])
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B3 	= 2

	elif (a == 'expon'):
		m, s 		= stats.expon.fit(Muestra_B3) 
		pdf_g 		= stats.expon.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='exponencial')
		plt.legend(handles=[red_patch])
		kurtosis 	= str(scipy.stats.kurtosis(Muestra_B3))
		plt.annotate(r'Kurtosis = ' + kurtosis, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		skewness 	= str(scipy.stats.skew(Muestra_B3))
		plt.annotate(r'Skewness = ' + skewness, xy=(0.02, 0.92), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top') 
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B3 	= 3

	else: 
		print "error en distribucion grupo 3"
else:
	pass

######################################GRAFICO 7: Tiempo entre eventos##########################################
##########################################GRUPO 4##############################################################

plt.subplot(344)
if (Escala == 0 or Escala == 1):
	itemindex_B4 		= np.nonzero(Segundos==Intervalos_seg_A[3])   											
	Muestra_B4 			= columna2[min(itemindex_B4[:][0]): max(itemindex_B4[:][0])]							
elif (Escala == 2):
	itemindex_B4 		= np.nonzero(x==Intervalos_seg_C[3])   											
	Muestra_B4 			= columna2[int(itemindex_B4[:][0])-400: int(itemindex_B4[:][0])]						
elif (Escala == 3):
	itemindex_B4 		= np.nonzero(x==Intervalos_seg_C[3])   											
	Muestra_B4 			= columna2[int(itemindex_B4[:][0])-800: int(itemindex_B4[:][0])]						
else:
	print 'Error en valor de seleccion de Escala'		

mu_H 				= np.mean(Muestra_B4)								
sigma_H 			= np.std(Muestra_B4)								
bins_H				= int(np.sqrt(len(Muestra_B4)))
weights 			= np.ones_like(Muestra_B4)/len(Muestra_B4)
if (opcion_graph==0):
	n, bins, patches 	= pyl.hist(Muestra_B4, weights=weights, log=True, facecolor='black', alpha=0.50, label=['Real'])
elif (opcion_graph==1):
	n, bins, patches 	= pyl.hist(Muestra_B4, weights=weights, facecolor='black', alpha=0.50, label=['Real'])
elif (opcion_graph==2):
	n, bins, patches 	= pyl.hist(Muestra_B4, bins_H, normed=True, facecolor='black', alpha=0.50, label=['Real'])
else:
	print 'Error en escala de graficos'
plt.xlabel('Tiempo entre eventos en 1 seg (grupo 4)')
plt.ylabel('Frecuencia')
plt.grid(True)

mles_Muestra_B4 = []

for distribution in distributions:
    pars = distribution.fit(Muestra_B4)
    mle_Muestra_B4 = distribution.nnlf(pars, Muestra_B4)
    mles_Muestra_B4.append(mle_Muestra_B4)

results = [(distribution.name, mle_Muestra_B4) for distribution, mle_Muestra_B4 in zip(distributions, mles_Muestra_B4)]
best_fit = sorted(zip(distributions, mles_Muestra_B4), key=lambda d: d[1])[0]
print 'Grupo 4: Mejor ajuste alcanzado en {}, valor MLE : {}'.format(best_fit[0].name, best_fit[1])
a = str(best_fit[0].name) 

from scipy import stats  
import matplotlib.patches as mpatches

xt = plt.xticks()[0]  			
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(Muestra_B4))

if (opcion_graph==2):
	if (a == 'norm'):
		m, s 		= stats.norm.fit(Muestra_B4) 																		 
		pdf_g 		= stats.norm.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='normal')
		plt.legend(handles=[red_patch], prop={'size':10})
		coeff 		= str(scipy.stats.variation(Muestra_B4))
		plt.annotate(r'Coeficiente de Variacion = ' + coeff, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		plt.plot(lnspc, pdf_g, color='red') 	
		distrib_B4 	= 1													

	elif (a == 'uniform'):
		m, s 		= stats.uniform.fit(Muestra_B4)  
		pdf_g 		= stats.uniform.pdf(lnspc, m, s)  
		red_patch 	= mpatches.Patch(color='red', label='uniforme')
		plt.legend(handles=[red_patch])
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B4 	= 2

	elif (a == 'expon'):
		m, s 		= stats.expon.fit(Muestra_B4) 
		pdf_g 		= stats.expon.pdf(lnspc, m, s) 
		red_patch 	= mpatches.Patch(color='red', label='exponencial')
		plt.legend(handles=[red_patch])
		kurtosis 	= str(scipy.stats.kurtosis(Muestra_B4))
		plt.annotate(r'Kurtosis = ' + kurtosis, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')      
		skewness 	= str(scipy.stats.skew(Muestra_B4))
		plt.annotate(r'Skewness = ' + skewness, xy=(0.02, 0.92), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top') 
		plt.plot(lnspc, pdf_g, color='red') 
		distrib_B4 	= 3

	else: 
		print "error en distribucion grupo 4"
else:
	pass

#y 					= pyl.mlab.normpdf(bins, mu_H, sigma_H)	#Linea de ajuste
#plt.plot(bins, y, 'r--', linewidth=2)


#################################PARA ANALISIS EN 100 O 10 MILISEGUNDOS########################################

if (Resolucion == 1):
	indices_D = indices_B

elif (Resolucion == 0):
	indices_D = indices_C

else:
	print "Error en escala en milisegundos"


###################################GRAFICO 8: Num de Eventos por segundo#######################################
##########################################GRUPO 1##############################################################


pkt_seg_Fig5A 		= np.array([])

if (Escala == 0 or Escala == 1):
	itemindex_C1 		= np.nonzero(Segundos==Intervalos_seg_B[0])   											#busca los indices para un valor especifico entregando una matriz
	miv	 				= min(itemindex_C1[:][0])
	mav 				= max(itemindex_C1[:][0])
	conteo 				= miv
elif (Escala == 2):
	itemindex_C1 		= np.nonzero(x==Intervalos_seg_D[0])   
	miv	 				= int(itemindex_C1[:][0])-400
	mav 				= int(itemindex_C1[:][0])
	conteo 				= miv											
elif (Escala == 3):
	itemindex_C1 		= np.nonzero(x==Intervalos_seg_D[0])   
	miv	 				= int(itemindex_C1[:][0])-800
	mav 				= int(itemindex_C1[:][0])
	conteo 				= miv						
else:
	print 'Error en valor de seleccion de Escala (Num de Eventos por tiempo)'	


for n in range(miv, mav):
	if (indices_D[n]== 1):
		ww 				= len(columna2[conteo: n]) 																#parte 1 (cantidad de eventos por tiempo)
		pkt_seg_Fig5A 	= np.append(pkt_seg_Fig5A, ww)
		conteo 			= n
	else :
		pass

###################################GRAFICO 8: Num de Eventos por segundo#######################################
##########################################GRUPO 2##############################################################

pkt_seg_Fig5B 		= np.array([])

if (Escala == 0 or Escala == 1):
	itemindex_C2 		= np.nonzero(Segundos==Intervalos_seg_B[1])   											#busca los indices para un valor especifico entregando una matriz
	miv	 				= min(itemindex_C2[:][0])
	mav 				= max(itemindex_C2[:][0])
	conteo 				= miv
elif (Escala == 2):
	itemindex_C2 		= np.nonzero(x==Intervalos_seg_D[1])   
	miv	 				= int(itemindex_C2[:][0])-400
	mav 				= int(itemindex_C2[:][0])
	conteo 				= miv											
elif (Escala == 3):
	itemindex_C2 		= np.nonzero(x==Intervalos_seg_D[1])   
	miv	 				= int(itemindex_C2[:][0])-800
	mav 				= int(itemindex_C2[:][0])
	conteo 				= miv						
else:
	print 'Error en valor de seleccion de Escala (Num de Eventos por tiempo)'

for n in range(miv, mav):
	if (indices_D[n]== 1):
		ww 				= len(columna2[conteo: n]) 						
		pkt_seg_Fig5B 	= np.append(pkt_seg_Fig5B, ww)
		conteo 			= n
	else :
		pass

###################################GRAFICO 8: Num de Eventos por segundo#######################################
##########################################GRUPO 3##############################################################

pkt_seg_Fig5C 		= np.array([])

if (Escala == 0 or Escala == 1):
	itemindex_C3 		= np.nonzero(Segundos==Intervalos_seg_B[2])   											#busca los indices para un valor especifico entregando una matriz
	miv	 				= min(itemindex_C3[:][0])
	mav 				= max(itemindex_C3[:][0])
	conteo 				= miv
elif (Escala == 2):
	itemindex_C3 		= np.nonzero(x==Intervalos_seg_D[2])   
	miv	 				= int(itemindex_C3[:][0])-400
	mav 				= int(itemindex_C3[:][0])
	conteo 				= miv											
elif (Escala == 3):
	itemindex_C3 		= np.nonzero(x==Intervalos_seg_D[2])   
	miv	 				= int(itemindex_C3[:][0])-800
	mav 				= int(itemindex_C3[:][0])
	conteo 				= miv						
else:
	print 'Error en valor de seleccion de Escala (Num de Eventos por tiempo)'

for n in range(miv, mav):
	if (indices_D[n]== 1):
		ww 				= len(columna2[conteo: n]) 						
		pkt_seg_Fig5C 	= np.append(pkt_seg_Fig5C, ww)
		conteo 			= n
	else :
		pass

###################################GRAFICO 8: Num de Eventos por segundo#######################################
##########################################GRUPO 4##############################################################

pkt_seg_Fig5D 		= np.array([])

if (Escala == 0 or Escala == 1):
	itemindex_C4 		= np.nonzero(Segundos==Intervalos_seg_B[3])   											#busca los indices para un valor especifico entregando una matriz
	miv	 				= min(itemindex_C4[:][0])
	mav 				= max(itemindex_C4[:][0])
	conteo 				= miv
elif (Escala == 2):
	itemindex_C4 		= np.nonzero(x==Intervalos_seg_D[3])   
	miv	 				= int(itemindex_C4[:][0])-400
	mav 				= int(itemindex_C4[:][0])
	conteo 				= miv											
elif (Escala == 3):
	itemindex_C4 		= np.nonzero(x==Intervalos_seg_D[3])   
	miv	 				= int(itemindex_C4[:][0])-800
	mav 				= int(itemindex_C4[:][0])
	conteo 				= miv						
else:
	print 'Error en valor de seleccion de Escala (Num de Eventos por tiempo)'

for n in range(miv, mav):
	if (indices_D[n]== 1):
		ww 				= len(columna2[conteo: n]) 						
		pkt_seg_Fig5D 	= np.append(pkt_seg_Fig5D, ww)
		conteo 			= n
	else :
		pass

###################################GRAFICO 8: Num de Eventos por segundo#######################################
############################################GRAFICOS###########################################################

plt.subplot(345)
#eje_x_C 	= sp.arange(len(pkt_seg_Fig5A))
#plt.bar(eje_x_C,pkt_seg_Fig5A)
mu_H 				= np.mean(pkt_seg_Fig5A)																	#Media
sigma_H 			= np.std(pkt_seg_Fig5A)																		#Desviacion estadar
#bins_H				= len(pkt_seg_Fig5A)#int(np.sqrt(len(Muestra_B)))
weights 			= np.ones_like(pkt_seg_Fig5A)/len(pkt_seg_Fig5A)
n, bins, patches 	= pyl.hist(pkt_seg_Fig5A, weights=weights, facecolor='green', alpha=0.50, label=['Real'])
plt.xlabel('Numero de eventos por segundo (grupo 1)')
plt.ylabel('Frecuencia')
plt.grid(True)

import statsmodels.api as sm
res = sm.Poisson(pkt_seg_Fig5A,np.ones_like(pkt_seg_Fig5A)).fit()
print 'NUMERO DE EVENTOS POR INTERVALO DE TIEMPO'
print 'Grupo 1:', res.summary()


plt.subplot(346)
mu_H 				= np.mean(pkt_seg_Fig5B)								
sigma_H 			= np.std(pkt_seg_Fig5B)								
#bins_H				= len(pkt_seg_Fig5B)#int(np.sqrt(len(Muestra_B)))
weights 			= np.ones_like(pkt_seg_Fig5B)/len(pkt_seg_Fig5B)
n, bins, patches 	= pyl.hist(pkt_seg_Fig5B, weights=weights, facecolor='slateblue', alpha=0.50, label=['Real'])
plt.xlabel('Numero de eventos por segundo (grupo 2)')
plt.ylabel('Frecuencia')
plt.grid(True)

plt.subplot(347)
mu_H 				= np.mean(pkt_seg_Fig5C)								#Media
sigma_H 			= np.std(pkt_seg_Fig5C)								#Desviacion estadar
#bins_H				= len(pkt_seg_Fig5C)#int(np.sqrt(len(Muestra_B)))
weights 			= np.ones_like(pkt_seg_Fig5C)/len(pkt_seg_Fig5C)
n, bins, patches 	= pyl.hist(pkt_seg_Fig5C, weights=weights, facecolor='red', alpha=0.50, label=['Real'])
plt.xlabel('Numero de eventos por segundo (grupo 3)')
plt.ylabel('Frecuencia')
plt.grid(True)

plt.subplot(348)
mu_H 				= np.mean(pkt_seg_Fig5D)								#Media
sigma_H 			= np.std(pkt_seg_Fig5D)								#Desviacion estadar
#bins_H				= len(pkt_seg_Fig5D)#int(np.sqrt(len(Muestra_B)))
weights 			= np.ones_like(pkt_seg_Fig5D)/len(pkt_seg_Fig5D)
n, bins, patches 	= pyl.hist(pkt_seg_Fig5D, weights=weights, facecolor='black', alpha=0.50, label=['Real'])
plt.xlabel('Numero de eventos por segundo (grupo 4)')
plt.ylabel('Frecuencia')
plt.grid(True)

#####################################GRAFICO 9: Delay general con marcas#######################################
############################################GRAFICOS###########################################################

plt.subplot(3,4,(9,12))

if (Escala==1 or Escala==0):
	plt.plot(columna1, columna4, color= 'blue')
	plt.plot(columna1[min(itemindex_C1[:][0]): max(itemindex_C1[:][0])], columna4[min(itemindex_C1[:][0]): max(itemindex_C1[:][0])], color= 'green', marker = 'd')
	plt.plot(columna1[min(itemindex_C2[:][0]): max(itemindex_C2[:][0])], columna4[min(itemindex_C2[:][0]): max(itemindex_C2[:][0])], color= 'slateblue', marker = 'd')
	plt.plot(columna1[min(itemindex_C3[:][0]): max(itemindex_C3[:][0])], columna4[min(itemindex_C3[:][0]): max(itemindex_C3[:][0])], color= 'red', marker = 'd')
	plt.plot(columna1[min(itemindex_C4[:][0]): max(itemindex_C4[:][0])], columna4[min(itemindex_C4[:][0]): max(itemindex_C4[:][0])], color= 'black', marker = 'd')
	plt.xlim(0,max(columna1)*1.001)
	plt.xlabel('Tiempo Transcurrido')
	plt.ylabel('Delay')
	plt.grid(True)

elif (Escala==2): 
	plt.plot(x, columna4, color= 'blue')
	plt.plot(x[int(itemindex_B1[:][0])-400: int(itemindex_B1[:][0])], columna4[int(itemindex_B1[:][0])-400: int(itemindex_B1[:][0])], color= 'green', marker = 'd')
	plt.plot(x[int(itemindex_B2[:][0])-400: int(itemindex_B2[:][0])], columna4[int(itemindex_B2[:][0])-400: int(itemindex_B2[:][0])], color= 'slateblue', marker = 'd')
	plt.plot(x[int(itemindex_B3[:][0])-400: int(itemindex_B3[:][0])], columna4[int(itemindex_B3[:][0])-400: int(itemindex_B3[:][0])], color= 'red', marker = 'd')
	plt.plot(x[int(itemindex_B4[:][0])-400: int(itemindex_B4[:][0])], columna4[int(itemindex_B4[:][0])-400: int(itemindex_B4[:][0])], color= 'black', marker = 'd')
	plt.xlabel('Numero de Datagrama')
	plt.ylabel('Delay')
	plt.xlim(0,max(x))
	plt.grid(True)

elif (Escala==3): 
	plt.plot(x, columna4, color= 'blue')
	plt.plot(x[int(itemindex_B1[:][0])-800: int(itemindex_B1[:][0])], columna4[int(itemindex_B1[:][0])-800: int(itemindex_B1[:][0])], color= 'green', marker = 'd')
	plt.plot(x[int(itemindex_B2[:][0])-800: int(itemindex_B2[:][0])], columna4[int(itemindex_B2[:][0])-800: int(itemindex_B2[:][0])], color= 'slateblue', marker = 'd')
	plt.plot(x[int(itemindex_B3[:][0])-800: int(itemindex_B3[:][0])], columna4[int(itemindex_B3[:][0])-800: int(itemindex_B3[:][0])], color= 'red', marker = 'd')
	plt.plot(x[int(itemindex_B4[:][0])-800: int(itemindex_B4[:][0])], columna4[int(itemindex_B4[:][0])-800: int(itemindex_B4[:][0])], color= 'black', marker = 'd')
	plt.xlabel('Numero de Datagrama')
	plt.ylabel('Delay')
	plt.xlim(0,max(x))
	plt.grid(True)

else:
	print 'Error en el Grafico de Delay (figura 4)'


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
#################################################FIGURA  5####################################################
#########################GRAFICO 9: Num de Eventos por intervalo de tiempo####################################

#rc('text',usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern']})

#plt.figure(5)
#plt.subplot(231)

#Muestra_A 			= pkt_seg_A[val_min:val_max]

#mu_H 				= np.mean(Muestra_A)								#Media
#sigma_H 			= np.std(Muestra_A)								#Desviacion estadar
###bins_H				= int(np.sqrt(len(Muestra_A)))
#bins_H				= len(Muestra_A)
#n, bins, patches 	= pyl.hist(Muestra_A, bins_H, normed=True, facecolor='green', alpha=0.50, label=['Real'])

#print 'Informacion Num eventos por intervalo de tiempo'
#print 'Mu =', mu_H, '/ Sigma =', sigma_H, '/ Bins =', bins_H
#print 'valor max=', max(bins),'/ max frecuencia =', max(n)

#from scipy.stats.kde import gaussian_kde
#from numpy import linspace

#kde = gaussian_kde( Muestra_A )												#Create the kernel, given an array it will estimate the probability over that values
#dist_space = linspace( min(Muestra_A), max(Muestra_A), 100 ) 	#these are the values over wich your kernel will be evaluated
#plt.plot( dist_space, kde(dist_space),'r--', linewidth=1 )   									#Plot the results

#print max(kde (dist_space))
#plt.xlabel('Nun eventos por intervalo de tiempo (por segundo)')
#plt.ylabel('Frecuencia')
#plt.grid(True)

###########################################Simulacion Poisson##################################################

#lamda			= np.mean(Muestra_A)
#s 				= np.random.poisson(lamda, len(Muestra_A))
#count, bins, ignored 	= plt.hist(s, bins_H, normed=True, histtype='bar', color=['blue'], alpha=0.50, label=['Poisson Sim'])
#plt.legend()
#plt.grid(True)




########################################Simulacion Exponencial#################################################

#lamda_B			= np.mean(Muestra_B)

#def func(xp, lamda_B):
#    return lamda_B * np.exp(-lamda_B * xp)

#xp= np.linspace(min(Muestra_B),max(Muestra_B),len(Muestra_B))
#y = func(xp, lamda_B)

#plt.legend()
#plt.grid(True)
#plt.plot(xp,y,'b--', linewidth=2)




##################################GRAFICO 11: Analisis Estadistico############################################

#plt.subplot(2,3,(2,3))
#plt.grid(True)

#plt.subplot(2,3,(5,6))
#plt.grid(True)



#################################################FIGURA 6######################################################
##################################GRAFICO 12: Comparacion de Histogramas#######################################

#from numpy import array_split
#from itertools import cycle

#plt.figure(6)

#datos_hist 	= columna2[Valor_minH: Valor_maxH]
#jj 			= np.array_split(datos_hist, Intervalos)

#cota_inf 	= 1
#colores 	= matplotlib.cm.rainbow(np.linspace(0, 1, Intervalos))
#lines 		= ["-","--"]
#linecycler 	= cycle(lines)

#for h in range(0,Intervalos):
#		kde 		= gaussian_kde( jj[h] )
#		pp 			= jj[h]
#		cota_sup 	= cota_inf + len(jj[h]) - 1
#		dist_space 	= linspace( min(pp), max(pp), 100 )
#####		if (max(kde(dist_space))>=umbral_inf and max(kde(dist_space)<umbral_sup)):
#		if (umbral_inf <= int(max(kde(dist_space))) < umbral_sup):
#			plt.plot( dist_space, kde(dist_space), linewidth=1, color=colores[h], linestyle=next(linecycler), label=['Intervalo N =', h, cota_inf, cota_sup, 'Valor max =', int(max(kde(dist_space)))])
#		else:
#			pass
#		try:		
#			cota_inf 	= cota_sup + 1
#		except:
#			pass


#plt.xlim(ejex_min,ejex_max)
#plt.legend()
#plt.grid(True)
######plt.show()

#######[x for x in a if x.size > 0]

#################################################FIGURA  7#####################################################
##################################GRAFICO 13: Deteccion de encolamiento########################################

from numpy import array_split

plt.figure(7)

Intervalos_A= int(len(columna2)/datospgrupo)
jjj 		= np.array_split(columna2, Intervalos_A)
print 'GRAFICO ANALISIS:'
print 'Cantidad de grupos= ',len(jjj)
cant_elementos = np.array([])
for n in range(0, len(jjj)):
	cant_elementos = np.append(cant_elementos, len(jjj[n]))
print 'Cantidad de datos por grupo / Min =', min(cant_elementos), ' / Max =', max(cant_elementos)  

datos_coeff 		= np.array([])
datos_kurtosis		= np.array([])
datos_skewness 		= np.array([])

for h in range(0,Intervalos_A):
		coeff 				= scipy.stats.variation(jjj[h])
		kurtosis 			= scipy.stats.kurtosis(jjj[h])
		skewness 			= scipy.stats.skew(jjj[h])
		datos_coeff 		= np.append(datos_coeff, coeff)	
		datos_kurtosis		= np.append(datos_kurtosis, kurtosis)	
		datos_skewness 		= np.append(datos_skewness, skewness)		

Intervalos_B= int(len(columna4)/datospgrupo)
ppp 		= np.array_split(columna4, Intervalos_A)											

datos_delay 		= np.array([])   
for h in range(0,Intervalos_B):
		x_delay 	= np.mean(ppp[h])
		datos_delay = np.append(datos_delay, x_delay)	

if (distrib_B1 == 1 and distrib_B2 == 1 and distrib_B3 == 1 and distrib_B4 == 1):
	#plt.subplot(221)
	from scipy.stats import gaussian_kde
	import matplotlib.pyplot as plt

	fig, ax1 = plt.subplots()

	from scipy.optimize import curve_fit

	x = datos_coeff[datos_coeff<0.019]
	y = datos_delay[datos_coeff<0.019]
	xy = np.vstack([x,y])																#Calculo de los puntos de densidad
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()											#ordenar puntos de densidad (mas densos ploteados al final)
	x, y, z = x[idx], y[idx], z[idx]
	im = ax1.scatter(x, y, c=z, s=50, edgecolor='k', cmap='Reds')
	fig.colorbar(im)

	x = datos_coeff[datos_coeff>0.019]
	y = datos_delay[datos_coeff>0.019]
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	im = ax1.scatter(x, y, c=z, s=50, edgecolor='k', cmap='Greens')
	fig.colorbar(im)
	plt.ylabel('Promedio Delay')
	plt.xlabel('Coeficiente de variacion')

	def func(x, a, b, c):
	    return a * np.exp(-b * x) + c

	xx = np.sort(datos_coeff)[::-1]
	yy = datos_delay
	popt, pcov = curve_fit(func, xx, yy)
	left, bottom, width, height = [0.3, 0.5, 0.3, 0.3]
	ax2 	= fig.add_axes([left, bottom, width, height])
	ax2.plot(xx, func(xx, *popt), 'r-', label="Fitted Curve")
	#plt.ylim(0, 0.02)


elif (distrib_B1 == 3 and distrib_B2 == 3 and distrib_B3 == 3 and distrib_B4 == 3):
	from scipy.stats import gaussian_kde
	import matplotlib.pyplot as plt

	plt.subplot(211)

	x = datos_skewness[datos_delay<=0.025]
	y = datos_delay[datos_delay<=0.025]
	xy = np.vstack([x,y])																#Calculo de los puntos de densidad
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()											#ordenar puntos de densidad (mas densos ploteados al final)
	x, y, z = x[idx], y[idx], z[idx]
	plt.scatter(x, y, c=z, s=50, edgecolor='k', cmap='Greens')
	plt.colorbar()

	x = datos_skewness[datos_delay>0.025]
	y = datos_delay[datos_delay>0.025]
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	plt.scatter(x, y, c=z, s=50, edgecolor='k', cmap='Reds')
	plt.colorbar()
	plt.ylabel('Promedio Delay')
	plt.xlabel('Skewness')

	plt.subplot(212)

	x = datos_kurtosis[datos_delay<=0.025]
	y = datos_delay[datos_delay<=0.025]
	xy = np.vstack([x,y])																#Calculo de los puntos de densidad
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()											#ordenar puntos de densidad (mas densos ploteados al final)
	x, y, z = x[idx], y[idx], z[idx]
	plt.scatter(x, y, c=z, s=50, edgecolor='k', cmap='Greens')
	plt.colorbar()

	x = datos_kurtosis[datos_delay>0.025]
	y = datos_delay[datos_delay>0.025]
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	plt.scatter(x, y, c=z, s=50, edgecolor='k', cmap='Reds')
	plt.colorbar()
	plt.ylabel('Promedio Delay')
	plt.xlabel('Kustosis')





else:
	pass

#ind_delay_arreglo 	= np.array([])
#p = 0
#for i in range(0,len(prueba)):
#	m = len(jjj[i]) 
#	for n in range(0, m): 
#		ind_delay_arreglo = np.append(ind_delay_arreglo, prueba[p])
#	p = p + 1

#ind_delay 	= np.array([])
#for n in range(0, len(ind_delay_arreglo)):		#revisa datos del array y los compara a un valor, almacenando el valor 1 en esa posicion del indice encontrado
#####	if (prueba[n] >= buscado_min and prueba[n] < buscado_max):
#	if (buscado_min <= ( ind_delay_arreglo[n] ) < buscado_max):
#		ind_delay 	= np.append(ind_delay, columna4[n])
#	else:
#		ind_delay 	= np.append(ind_delay, [0])



#plt.plot(x, columna4)
#plt.plot(x[ind_delay>0], ind_delay[ind_delay>0],'ro')


#plt.xlabel('Numero de Medicion')
#plt.xlim(0,len(x))
#####plt.ylim(0,0.20)
#plt.ylabel('Delay')
#####plt.title('Grafico Tiempo de Servicio')
#plt.grid(True)

import matplotlib as mpl
print 'Version de Matplotlib =', mpl.__version__


plt.show()



###############################################################################################################
###############################################EXPORTACION DE ARRAYS###########################################
import csv

archivo 	= csv.writer(open("ARCHIVO_Tiempo_entre_datagramas.csv","w"), delimiter=',')	#,quoting=csv.QUOTE_ALL)
rows = zip(columna2)								#writerow escribe por defecto en filas, esto permite dejar los datos en columnas
for row in rows:
	archivo.writerow(row)

#np.savetxt('Arrays_Servidor.csv', columna4, delimiter=',') 			#presenta problemas en la lectura

print 'Archivo Creado'


archivo 	= csv.writer(open("ARCHIVO_Varianzas.csv","w"), delimiter=',')	#,quoting=csv.QUOTE_ALL)
rows = zip(prueba)								#writerow escribe por defecto en filas, esto permite dejar los datos en columnas
for row in rows:
	archivo.writerow(row)

#np.savetxt('Arrays_Servidor.csv', columna4, delimiter=',') 			#presenta problemas en la lectura

print 'Archivo Creado'



#CREAR ARCHIVOS DE TEXTO CON LOS DATOS
#np.savetxt('Servidor-TBP.txt', t_entre_llegadas, delimiter=' ') 		#presenta problemas en la lectura
#np.savetxt('Servidor-Delay.txt', columna4, delimiter=' ') 				#presenta problemas en la lectura

#print 'Archivo txt Creado'

