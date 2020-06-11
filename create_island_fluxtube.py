from osa import Client
import matplotlib
import numpy as np
import boututils.calculus as calc

def island_fluxtube_grid(nx, ny, nz, turns=1, configuration=1):
    ### Copy GPI Code:
    # Generates a nx by nz rectangular grid at the location of the
    # outboard midplane island in standard configuration
    ####
    bottom_left = [6.1,0,-0.5]
    bottom_right = [6.2,0,-0.5] 
    top_left = [6.1,0,0.5] 
    top_right = [6.2,0,0.5]
    x_array = np.zeros((nx,ny,nz))
    z_array = np.zeros((nx,ny,nz))
    y_array = np.zeros((nx,ny,nz))
    for y in np.arange(0,ny):
        left_edge_x  = np.linspace(bottom_left[0],top_left[0], nz)
        left_edge_z  = np.linspace(bottom_left[2],top_left[2], nz)
        left_edge_y  = np.linspace(bottom_left[1], top_left[1], nz)
        
        right_edge_x = np.linspace(bottom_right[0],top_right[0], nz)
        right_edge_z = np.linspace(bottom_right[2],top_right[2], nz)
        right_edge_y = np.linspace(bottom_right[1],top_right[1], nz)
        
        for z in np.arange(0,nz):
            line_x = np.linspace(left_edge_x[z],right_edge_x[z],nx)
            line_z = np.linspace(left_edge_z[z],right_edge_z[z],nx)
            line_y = np.linspace(left_edge_y[z],right_edge_y[z],nx)
            z_array[:,y,z] = line_z
            x_array[:,y,z] = line_x
            y_array[:,y,z] = line_y


    ### Have grid, must calculate curvature ####
    
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    pos = tracer.types.Points3D()

    config = tracer.types.MagneticConfig()
    config.configIds = configuration

    lineTask = tracer.types.LineTracing()
    lineTask.numSteps = turns*ny

    task = tracer.types.Task()
    task.step = turns*2*np.pi*np.mean(x_array)/lineTask.numSteps
    task.lines = lineTask

    pos.x1 = x_array.flatten()
    pos.x2 = y_array.flatten()
    pos.x3 = z_array.flatten()


    res = tracer.service.trace(pos, config, task, None, None)

    x = np.asarray(res.lines[0].vertices.x1)
    y = np.asarray(res.lines[0].vertices.x2)
    z = np.asarray(res.lines[0].vertices.x3)

    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    dphi = phi[1]-phi[0]
    
    drdphi = calc.deriv(r)/dphi
    d2rdphi2 = calc.deriv(drdphi)/dphi
    k = (r**2 + 2*drdphi**2 - r*d2rdphi2)/ ((r**2 + drdphi**2)**(1.5))

    k = np.ndarray.reshape(k,(nx,ny,nz))
    # dzdphi = calc.deriv(z)/dphi
    # d2zdphi2 = calc.deriv(dzdphi)/dphi

    # Ktot = np.sqrt(d2zdphi2**2 + (d2rdphi2*dzdphi - d2zdphi2*drdphi)**2 - (d2rdphi2)**2) / (drdphi**2+dzdphi**2)**1.5

    # kz = Ktot - k 
    
    return r,phi,x,y,z,k
