from osa import Client
coils_db_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBProxy?wsdl')
extender_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')

my_config = extender_client.types.MagneticConfiguration()
my_config.name = 'w7x'

coilIds = coils_db_client.service.getConfigData([3])
coils_data = coils_db_client.service.getCoilData(coilIds[0].coils)

currents = [14000.0, 14000.0, 13160.0, 12950.0, 12390.0, -9660.0, -9660.0]
circuits = []

for i in range(7):	
	sc = extender_client.types.SerialCircuit()
	sc.name = 'coil group ' + str(i+1)
	sc.current = currents[i]
	sc.currentCarrier = []
	for j in range(10):
		if i < 5:
			index = i + j * 5
			c = coils_data[index]
			c.numWindings = 108
		elif i == 5: 
			index = 50 + j * 2
			c = coils_data[index]
			c.numWindings = 36
		else:
			index = 51 + j * 2
			c = coils_data[index]
			c.numWindings = 36
			
		pf1 = extender_client.types.PolygonFilament()
		pf1.vertices = extender_client.types.Points3D()
		pf1.vertices.x1 = c.vertices.x1 
		pf1.vertices.x2 = c.vertices.x2 
		pf1.vertices.x3 = c.vertices.x3
		coil = extender_client.types.Coil()
		coil.name = c.name
		print(str(index) + ' ' + c.name)

		coil.windingDirection = 1.
		coil.currentCarrierPrimitive = [pf1]	
		sc.currentCarrier.append(c)

	circuits.append(sc)

import pdb; pdb.set_trace()
my_config.circuit = circuits
