# from osa import Client

# # coils_db_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBProxy?wsdl')
# cl = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')
# tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')
# config = tracer.types.MagneticConfig()
# config.configIds = 15

# # ''' get full configuration from db: '''
# # my_config = coils_db_client.service.getMagneticConfiguration(15)


# vmecURL = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/run/w7x.1000_1000_1000_1000_+0500_+0500.03.09ll_fixed2aaa_8.48/wout.nc' 
# points = cl.types.Points3D()
# points.x1 = [5.5, 5.6, 5.6, 5.6, 5.6, 5.6, 4.6, 4.6, 4.6, 4.6, 4.6]
# points.x2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.628, 0.628, 0.628, 0.628, 0.628]
# points.x3 = [0.0, -0.2, -0.1, 0.0, 0.1, 0.2, -0.2, -0.1, 0.0, 0.1, 0.2]

# coils = cl.types.MagneticConfiguration(15)
# print config
# # .. create MagneticConfiguration object e.g. with coil data from CoilsDB, please see Data Types section..

# plasmaField = cl.service.getExtendedField(None, vmecURL, config, None, points, None)

# print plasmaField

# from osa import Client
# coils_db_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBProxy?wsdl')
# extender_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')

# ''' get full configuration from db: '''
# my_config = coils_db_client.types.getMagneticConfiguration(1)
# vmecURL = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/run/w7x.1000_1000_1000_1000_+0500_+0500.03.09ll_fixed2aaa_8.48/wout.nc' 
# points = cl.types.Points3D()
# points.x1 = [5.5, 5.6, 5.6, 5.6, 5.6, 5.6, 4.6, 4.6, 4.6, 4.6, 4.6]
# points.x2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.628, 0.628, 0.628, 0.628, 0.628]
# points.x3 = [0.0, -0.2, -0.1, 0.0, 0.1, 0.2, -0.2, -0.1, 0.0, 0.1, 0.2]
# plasmaField = cl.service.getExtendedField(None, vmecURL, my_config, None, points, None)

from osa import Client
coils_db_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBProxy?wsdl')
extender_client = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')

''' get full configuration from db: '''
my_config = coils_db_client.service.getConfigData(15)[0]

cl = Client('http://esb.ipp-hgw.mpg.de:8280/services/Extender?wsdl')
vmecURL = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/run/test42/wout.nc'


points = cl.types.Points3D()
points.x1 = [5.5, 5.6, 5.6, 5.6, 5.6, 5.6, 4.6, 4.6, 4.6, 4.6, 4.6]
points.x2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.628, 0.628, 0.628, 0.628, 0.628]
points.x3 = [0.0, -0.2, -0.1, 0.0, 0.1, 0.2, -0.2, -0.1, 0.0, 0.1, 0.2]

plasmaField = cl.service.getExtendedField(None, vmecURL, my_config, None, points, None)

import pdb; pdb.set_trace()
print plasmaField
