from osa import Client
coils_db_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBProxy?wsdl')
extender_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')

''' get full configuration from db: '''
#my_config = coils_db_client.service.getConfigData(1)

''' ... or build from single coils: '''

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
                coil1 = extender_client.types.Coil()
                                
		if i < 5:
			index = i + j * 5
			c = coils_data[index]
			coil1.numWindings = 108
		elif i == 5: 
			index = 50 + j * 2
			c = coils_data[index]
			coil1.numWindings = 36
		else:
			index = 51 + j * 2
			c = coils_data[index]
			coil1.numWindings = 36
					
		pf1 = extender_client.types.PolygonFilament()
		pf1.vertices = extender_client.types.Points3D()
		pf1.vertices.x1 = c.vertices.x1 
		pf1.vertices.x2 = c.vertices.x2 
		pf1.vertices.x3 = c.vertices.x3
		coil1.name = c.name

		coil1.windingDirection = 1.
		coil1.currentCarrierPrimitive = [pf1]	
		sc.currentCarrier.append(coil1)

	circuits.append(sc)
	
	
my_config.circuit = circuits


from osa import Client
coils_db_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBProxy?wsdl')
extender_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')

''' get full configuration from db: '''

cl = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')
vmecURL = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/run/test42/wout.nc'


points = cl.types.Points3D()
points.x1 = [5.5, 5.6, 5.6, 5.6, 5.6, 5.6, 4.6, 4.6, 4.6, 4.6, 4.6]
points.x2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.628, 0.628, 0.628, 0.628, 0.628]
points.x3 = [0.0, -0.2, -0.1, 0.0, 0.1, 0.2, -0.2, -0.1, 0.0, 0.1, 0.2]

plasmaField = cl.service.getExtendedField(None, vmecURL, my_config, None, points, None)
import pdb; pdb.set_trace()
print plasmaField
