import socket as so
import time
import struct
import numpy as np
import binascii
import threading
import Queue 
import csv

############################################################

#HOST 	= '192.168.1.3
HOST 	= '192.168.0.1'
#HOST 	= 'localhost'
PORT 	= 10677
BW_MIN 	= 1
BW_MAN	= 12
BW_MAX	= 12									#MAX 45mbps


q = Queue.Queue()  

#####TIEMPOS OFFSET
### 0.000135 	- 45Mbps
### 0.0001375 	- 18Mbps

###################GENERADOR DE TIMESLEEP################################

Vector_a 	= np.array(np.linspace(BW_MIN*(10**6), BW_MAN*(10**6), 20))	#Anchos de banda deseados 100
Vector_b	= np.array(np.linspace(BW_MAN*(10**6), BW_MAX*(10**6), 100))	#Anchos de banda deseados que se desean remarcar al final 70
BW		= np.hstack((Vector_a,Vector_b))
SLEEPTIME 	= ((BW/8)/1497)**(-1)						#Trasformacion a timesleep
#offset		= 0.000263							#REGULAR TIEMPO OFFSET 0.000265     /con 0 a 50 MAX se transmitem 30 mbps exactos (wireshark)
offset		= 0.0001375							#Si sube, sube tambien el ancho de banda
SLEEPTIME	= SLEEPTIME - offset						#se resta el offset que genera la aplicacion
#sleeptime 	= sleeptime[::-1]						#GENERA UN ANCHO DE BANDA CRECIENTE(COMENTARLO GENERA UNO DECRECIENTE)

###################HILO TRANSMISOR DE PAQUETES UDP########################

class Transmision(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.args 		= args
		self.kwargs 		= kwargs
		self.HOST		= args[0]
		self.PORT		= args[1]
		self.SLEEPTIME		= args[2]
		self.q			= args[3]
		self.s 			= so.socket(so.AF_INET, so.SOCK_DGRAM)
		self.pkt_num 		= 1		
		self.totalbytes 	= 0				#bytes iniciales enviados
		self.totalsnt 		= 0				#paquetes iniciales enviados
		self.avance_array	= 0
		self.t_actual		= time.time()

	def pack_data(self, time_gen, sleep_time, pkt_num):
		self.relleno		= 'X'*1410
		self.info 		= time_gen, sleep_time, pkt_num, self.relleno  	#no mayor de 1410 debido a los 1500 MTU max (1497 SEGUN WIRESHARK)
		self.packer 		= struct.Struct('26s 17s I 1410s')
		self.packed_data 	= self.packer.pack(*self.info)			#*incluye todos los parametros de info
		#print "%s" % binascii.hexlify(packed_data)
		######################################################
		#unpacker	= struct.Struct('26s 17s I 1410s')
		#normal	 	= unpacker.unpack(packed_data)
		#print normal
		######################################################
		return self.packed_data

	def run(self):
		q 	= self.q
		s	= self.s
		while True:
			if (self.totalsnt == 0):	
       	       			self.timestamp 	= time.time()
				self.udp	= "XX"
				s.sendto(str(self.udp),(HOST,PORT))

			self.t_avance 	= time.time()
			if ((self.t_avance - self.t_actual) > 0.5):	  			
				self.avance_array += 1	
				self.t_actual	= self.t_avance

			if (self.avance_array == (len(self.SLEEPTIME)-1)):
				udp 	= "X"
				s.sendto(str(udp),(HOST,PORT))
				s.close()
				print "Finished"
				break	

			self.time_gen_pkt 	= time.time()
			self.time_gen		= (str("{0:.15f}".format(self.time_gen_pkt)))
			self.sleep_time		= (str("{0:.15f}".format(SLEEPTIME[self.avance_array])))

			self.packed_data	= Transmision.pack_data(self, self.time_gen, self.sleep_time, self.pkt_num)
			self.n_bytes 		= s.sendto(self.packed_data, (self.HOST,self.PORT))
	
			self.pktlen 		= (self.n_bytes + 42) 	#se anade 42 correspondiente a la cabecera ip
			self.pkt_num	 	+= 1
			self.sntstamp		= (str("{0:.15f}".format(time.time())))	
			self.totalbytes 	+= self.pktlen
			self.totalsnt 		+= 1

			######################TIMESLEEP#######################
			time.sleep(SLEEPTIME[self.avance_array])	
	
			######################CSV QUEUE#######################
			#d = Decimal(datos[n]) 
			#print d.as_tuple().exponent
			archivar 		= (self.timestamp, self.sntstamp, self.totalbytes, self.totalsnt, SLEEPTIME[self.avance_array])  			
			q.put(archivar)


###################HILO GENERADOR DE ARCHIVO##############################

class GenArchivo(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.q	 	= args
		self.kwargs 	= kwargs
		self.out 	= csv.writer(open("DATA_Cliente.csv","w"), delimiter=',')	#,quoting=csv.QUOTE_ALL)

	def run(self):
		q 		= self.q
        	while True:   
	            	try:  
				info = q.get(True) 
				self.out.writerow(info)
	            	except Queue.Empty:
				#pass
				if (udp.isAlive):
					pass
				else:
					print "cerrando"
					break


###################HILO RECEPTOR DE PAQUETES TCP###########################

class Aviso(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.args 	= args
		self.kwargs 	= kwargs
		self.HOST 	= args[0]
		self.PORT 	= args[1]
		self.BUFFER_SIZE= 1024
		self.s		= so.socket(so.AF_INET, so.SOCK_STREAM)
		####DEBE SER CAMBIADO POR UNA ALERTA O BANDERA GLOBAL
		self.s.connect((self.HOST, self.PORT+1))

	def run(self):
		while True:  
      			mensaje = str(1) 
      			self.s.send(mensaje)  

########################INICIAR THREADS####################################
#PORT 	= 10565
udp 	= Transmision(args=(HOST, PORT, SLEEPTIME, q), kwargs=())
arch 	= GenArchivo(args=(q), kwargs=())
#tcp 	= Aviso(args=(HOST, PORT, q), kwargs=())

udp.start()
#tcp.start()
arch.start()




		


	


	
	
	





