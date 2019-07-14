import socket as so
import time
import struct
import threading
import Queue 
import csv

############################################################

HOST 		= '192.168.0.1'
#HOST 		= "localhost"
PORT 		= 10685
THRESHOLD	= 0.0015
buffer 		= 102400					#819200 bit = 102400 byte = 100 KB o Kilobyte  
								##por defecto cat /proc/sys/net/ipv4/udp_mem, R=93222 124298 186444

q = Queue.Queue()   

###################HILO RECEPTOR DE PAQUETES UDP###########################

class Recepcion(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):		#target = funcion
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)		#Constructor
		self.args 	= args
		self.kwargs 	= kwargs
		self.HOST	= args[0]
		self.PORT	= args[1]
		self.q		= args[2]
		self.s 		= so.socket(so.AF_INET,so.SOCK_DGRAM)			#ADDR = (HOST,PORT)
		self.s.bind((self.HOST, self.PORT))					#bind((ADDR))
		self.totalbytes	= 0 							#bytes iniciales recibidos
		self.totalrcvs 	= 0 							#paquetes iniciales arribados

	def extract_data(self, data):
		############################################
		self.data	= data[0:48]
		self.unpacker	= struct.Struct('26s 17s I')
		self.normal 	= self.unpacker.unpack(self.data)
		############################################	
		self.t_salida 	= float(self.normal[0])
		self.timesleep	= float(self.normal[1])
		self.pkt_salido	= self.normal[2]
		return self.t_salida, self.timesleep, self.pkt_salido

	def run(self):
		q 		= self.q
		while True:
			self.data, self.addr  	= self.s.recvfrom(buffer)		#funcion recvfrom(BUFFER SIZE) para datagrama UDP ##llega (data,(host,port))
			self.len_data 		= len(self.data)
			self.rcvstamp   	= time.time()				#hora UTC expresada en segundos como flotante, mejor en Linux
	
			if (self.len_data == 2):	
				self.timestamp 	= self.rcvstamp
			elif (self.len_data == 1):
				self.s.close()
				signal		= 1
				break
			else:
				self.extraer 	= Recepcion.extract_data(self, self.data)						
				self.pktlen     = (self.len_data + 42)			#se anade 42 correspondiente a la cabecera ip (data es solo el paquete UDP)
				self.totalbytes	+= self.pktlen
				self.totalrcvs 	+= 1
				###################MUTEX#############################
				self.t_llegada	= (str("{0:.15f}".format(self.rcvstamp)))
				self.t_salida	= (str("{0:.15f}".format(self.extraer[0])))

				archivar 	= (self.t_salida, self.extraer[1], self.extraer[2], self.timestamp, self.t_llegada, self.totalbytes, self.totalrcvs)
				q.put(archivar)


###################HILO GENERADOR DE ARCHIVO##############################

class GenArchivo(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.q	 	= args
		self.kwargs 	= kwargs
		self.out 	= csv.writer(open("DATA_Servidor.csv","w"), delimiter=',')	#,quoting=csv.QUOTE_ALL)

	def run(self):
		q 	= self.q
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


###################HILO TRANSMISOR DE PAQUETES TCP###########################

class Aviso(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.umbral = args
		self.kwargs = kwargs
		self.HOST	= args[0]
		self.PORT	= args[1]
		self.umbral	= args[2]
		self.q		= args[3]
		self.BUFFER_SIZE= 1024
		self.s 		= so.socket(so.AF_INET, so.SOCK_STREAM)	#Conexion IP para asegurar la llegada del paquete
		self.s.setsockopt(so.SOL_SOCKET, so.SO_REUSEADDR, 1)
		self.s.bind((self.HOST, self.PORT+1))
		self.s.listen(1)
		self.conn, self.addr	= self.s.accept()

	def run(self):
		while True:  
      			recibido = self.conn.recv(1024)  		####NOTA: EN EL SERVIDOR, RECV/SEND VAN LIGADOS A CONN
			print recibido


########################INICIAR THREADS####################################
#PORT 		= 10565
udp 	= Recepcion(args=(HOST, PORT, q), kwargs=())
arch 	= GenArchivo(args=(q), kwargs=())
#tcp 	= Aviso(args=(HOST, PORT, THRESHOLD, q), kwargs=())

udp.start()
#tcp.start()
arch.start()

#while (1):
#	if arch.isAlive():
#		print "VIVA ARCH"
#	else:
#		print "MUERTA ARCH"

#	if udp.isAlive():
#		print "VIVA UDP"
#	else:
#		print "MUERTA UDP"


#tcp 	= Transmision(args=(HOST, PORT, THRESHOLD), kwargs=())
#udp.join()	#se enganche el programa hasta que termine el hilo
#tcp.start()

							

######################################################



