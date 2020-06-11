from osa import Client
import matplotlib
import numpy as np
import boututils.calculus as calc

def field_line_curvature(xin,yin,zin, configuration=1):

    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    pos = tracer.types.Points3D()

    config = tracer.types.MagneticConfig()
    config.configIds = configuration

    lineTask = tracer.types.LineTracing()
    lineTask.numSteps = 512

    task = tracer.types.Task()
    task.step = 2*np.pi*(np.sqrt(xin**2+yin**2))/512
    task.lines = lineTask

    pos.x1 = xin
    pos.x2 = yin
    pos.x3 = zin

    res_forward = tracer.service.trace(pos, config, task, None, None)
    
    x = np.asarray(res_forward.lines[0].vertices.x1)
    y = np.asarray(res_forward.lines[0].vertices.x2)
    z = np.asarray(res_forward.lines[0].vertices.x3)

    # config.inverseField=1

    # res_backward = tracer.service.trace(pos, config, task, None, None)

    # x = np.insert(x,0,res_backward.lines[0].vertices.x1[:])
    # y = np.insert(y,0,res_backward.lines[0].vertices.x2[:])
    # z = np.insert(z,0,res_backward.lines[0].vertices.x3[:])
    
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    dphi = phi[1]-phi[0]
    
    drdphi = calc.deriv(r)/dphi
    d2rdphi2 = calc.deriv(drdphi)/dphi
    k = (r**2 + 2*drdphi**2 - r*d2rdphi2)/ ((r**2 + drdphi**2)**(1.5))

    # dzdphi = calc.deriv(z)/dphi
    # d2zdphi2 = calc.deriv(dzdphi)/dphi

    # Ktot = np.sqrt(d2zdphi2**2 + (d2rdphi2*dzdphi - d2zdphi2*drdphi)**2 - (d2rdphi2)**2) / (drdphi**2+dzdphi**2)**1.5

    # kz = Ktot - k 
    
    return r,phi,x,y,z,k

