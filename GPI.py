import numpy as np

def GPI_view(nx=132,nz=128, extended=False, return_corners=False, extension_factor=0):
    '''
    Generates a nx by nz rectangular grid at the location of the GPI view
    
    returns: 2D arrays 'x,y,z' corresponding to W7-X machine coordinates
    '''
    if not (extended):
        bottom_left = [.4484,-5.9987,-.2948]
        bottom_right = [.4516,-6.0417,-.3106] 
        top_left = [.4503,-6.0246,-.2234] 
        top_right = [.4536,-6.0677,-.2392]
    else:
        bottom_left = [.4484,-5.9987,-.2948]
        bottom_right = [.4516,-6.0417,-.3106] 
        top_left = [.4503,-6.0246,-.2234] 
        top_right = [.4536,-6.0677,-.2392]

        # rename variables for each corner
        rtl = np.sqrt(top_left[0]**2 + top_left[1]**2)
        ztl = top_left[-1]

        rtr = np.sqrt(top_right[0]**2 + top_right[1]**2)
        ztr = top_right[-1]

        rbl = np.sqrt(bottom_left[0]**2 + bottom_left[1]**2)
        zbl = bottom_left[-1]

        rbr = np.sqrt(bottom_right[0]**2 + bottom_right[1]**2)
        zbr = bottom_right[-1]

        # line from top left to bottom right
        m_tlbr = (ztl-zbr)/ (rtl-rbr)
        b_tlbr = ztl - m_tlbr*rtl
        d_tlbr = np.sqrt((rtl-rbr)**2 + (ztl-zbr)**2)

        # line from bottom left to top right
        m_bltr = (ztr-zbl)/ (rtr-rbl)
        b_bltr = ztr - m_bltr*rtr
        d_bltr = np.sqrt((rbl-rtr)**2 + (zbl-ztr)**2)

        # distance to expand (proportion of how diagonals)
        d_new_bltr = d_bltr*extension_factor
        d_new_tlbr = d_tlbr*extension_factor

        corners_tlbr = [[rtl,ztl], [rbr,zbr]]
        corners_bltr = [[rbl,zbl], [rtr,ztr]]
        new_corners = [[0,0],[0,0],[0,0],[0,0]]
        for point,i in zip(corners_tlbr,[0,1]):
            x = point[0]
            z = point[1]
            a_qe = (m_tlbr**2 + 1)
            b_qe = 2*(m_tlbr*b_tlbr - m_tlbr*z - x)
            c_qe = (z**2 + b_tlbr**2 - 2*z*b_tlbr - d_new_tlbr**2 + x**2)

            x_p = (-b_qe + np.sqrt(b_qe**2 - 4*a_qe*c_qe)) / (2*a_qe)
            x_m = (-b_qe - np.sqrt(b_qe**2 - 4*a_qe*c_qe)) / (2*a_qe)
            
            z_p = m_tlbr*x_p + b_tlbr
            z_m = m_tlbr*x_m + b_tlbr

            d_p = np.sqrt((x_p-corners_tlbr[np.abs(i-1)][0])**2 + (z_p-corners_tlbr[np.abs(i-1)][1])**2)
            d_m = np.sqrt((x_m-corners_tlbr[np.abs(i-1)][0])**2 + (z_m-corners_tlbr[np.abs(i-1)][1])**2)
            if (d_p > d_tlbr):
                new_corners[i][0] = x_p
                new_corners[i][1] = z_p
            else:
                new_corners[i][0] = x_m
                new_corners[i][1] = z_m

            
        for point,i in zip(corners_bltr,[2,3]):
            i -= 2
            x = point[0]
            z = point[1]
            a_qe = (m_bltr**2 + 1)
            b_qe = 2*(m_bltr*b_bltr - m_bltr*z - x)
            c_qe = (z**2 + b_bltr**2 - 2*z*b_bltr - d_new_bltr**2 + x**2)

            x_p = (-b_qe + np.sqrt(b_qe**2 - 4*a_qe*c_qe)) / (2*a_qe)
            x_m = (-b_qe - np.sqrt(b_qe**2 - 4*a_qe*c_qe)) / (2*a_qe)

            z_p = m_bltr*x_p + b_bltr
            z_m = m_bltr*x_m + b_bltr

            d_p = np.sqrt((x_p-corners_bltr[np.abs(i-1)][0])**2 + (z_p-corners_bltr[np.abs(i-1)][1])**2)
            d_m = np.sqrt((x_m-corners_bltr[np.abs(i-1)][0])**2 + (z_m-corners_bltr[np.abs(i-1)][1])**2)
            if (d_p > d_bltr):
                new_corners[i+2][0] = x_p
                new_corners[i+2][1] = z_p
            else:
                new_corners[i+2][0] = x_m
                new_corners[i+2][1] = z_m    

        phi = 4.787
        top_left = [new_corners[0][0]*np.cos(phi), new_corners[0][0]*np.sin(phi), new_corners[0][-1]]
        bottom_right = [new_corners[1][0]*np.cos(phi), new_corners[1][0]*np.sin(phi), new_corners[1][-1]]
        bottom_left = [new_corners[2][0]*np.cos(phi), new_corners[2][0]*np.sin(phi), new_corners[2][-1]]
        top_right = [new_corners[3][0]*np.cos(phi), new_corners[3][0]*np.sin(phi), new_corners[3][-1]]
        
    left_edge_x  = np.linspace(bottom_left[0],top_left[0], nz)
    left_edge_z  = np.linspace(bottom_left[2],top_left[2], nz)
    left_edge_y  = np.linspace(bottom_left[1], top_left[1], nz)

    right_edge_x = np.linspace(bottom_right[0],top_right[0], nz)
    right_edge_z = np.linspace(bottom_right[2],top_right[2], nz)
    right_edge_y = np.linspace(bottom_right[1],top_right[1], nz)

    x_array = np.zeros((nx,nz))
    z_array = np.zeros((nx,nz))
    y_array = np.zeros((nx,nz))

    for z in np.arange(0,nz):
        line_x = np.linspace(left_edge_x[z],right_edge_x[z],nx)
        line_z = np.linspace(left_edge_z[z],right_edge_z[z],nx)
        line_y = np.linspace(left_edge_y[z],right_edge_y[z],nx)
        z_array[:,z] = line_z
        x_array[:,z] = line_x
        y_array[:,z] = line_y

    if not (return_corners):
        return x_array, y_array, z_array
    else:
        return x_array, y_array, z_array, bottom_left, top_left, top_right, bottom_right 

def GPI_lpar(x,y,z, configuration=1, limit=5000):
    '''
    Calculates Parallel connection length given:
    input: 
          - 2D arrays of W7-X machine coordinates (x,y,z)
          - Configuration corresponding to the Webservices pre-built configurations
          - A limit for the maximum connection length (closed field line regions)

    returns: 2D array of connection lengths (same size as x) 

    NOTE: x,y,z must be same dimensions

    '''
    ### Use webservices to get connection length ###
    from osa import Client
    tracer = Client('http://esb:8280/services/FieldLineProxy?wsdl')
    points = tracer.types.Points3D()
    points.x1 = x.flatten()
    points.x2 = y.flatten()
    points.x3 = z.flatten()


    ### copied from webservices...#
    task = tracer.types.Task()
    task.step = 6.5e-3
    con = tracer.types.ConnectionLength()
    con.limit = limit
    con.returnLoads = False
    task.connection = con

    # diff = tracer.types.LineDiffusion()
    # diff.diffusionCoeff = .00
    # diff.freePath = 0.1
    # diff.velocity = 5e4
    # task.diffusion = diff

    config = tracer.types.MagneticConfig()

    config.configIds = configuration  ## Just use machine IDs instead of coil currents because it's easier.

    ### This bit doesn't work when called as a function.  
    # config.coilsIds = range(160,230)
    # config.coilsIdsCurrents = [1.43e6,1.43e6,1.43e6,1.43e6,1.43e6]*10
    # # config.coilsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)


    # # config.coilsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)
    # config.coilsIdsCurrents = [1.43e6,1.43e6,1.43e6,1.43e6,1.43e6]*10
    # # config.coilsIdsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)
    
    grid = tracer.types.Grid()
    grid.fieldSymmetry = 5

    cyl = tracer.types.CylindricalGrid()
    cyl.numR, cyl.numZ, cyl.numPhi = 181, 181, 481
    cyl.RMin, cyl.RMax, cyl.ZMin, cyl.ZMax = 4.05, 6.75, -1.35, 1.35
    grid.cylindrical = cyl

    machine = tracer.types.Machine(1)
    machine.grid.numX, machine.grid.numY, machine.grid.numZ = 500,500,100
    machine.grid.ZMin, machine.grid.ZMax = -1.5,1.5
    machine.grid.YMin, machine.grid.YMax = -7, 7
    machine.grid.XMin, machine.grid.XMax = -7, 7
    # machine.meshedModelsIds = [164]
    machine.assemblyIds = [12,14,8,9,13,21]

    config.grid = grid

    config.inverseField = False

    res_fwd = tracer.service.trace(points, config, task, machine)

    config.inverseField = True

    res_bwd = tracer.service.trace(points, config, task, machine)

    ###### end of copied code #######

    nx = x.shape[0]
    nz = x.shape[-1]
    
    lengths = np.zeros((nx*nz))
    
    for n in np.arange(0,nx*nz):
        lengths[n] = 0.5*(res_fwd.connection[n].length + res_bwd.connection[n].length)

    lengths = np.ndarray.reshape(lengths,(nx,nz))

    return lengths

def GPI_curvature(x,y,z):
    '''
    Calculates curvature in a 2D plane given:
    input: 2D W7-X real-space coordinates
    returns: 2D curvature drives for BOUT++ in x,y,z directions
    '''


    phi0 = 4.787
    
    from osa import Client
    from boututils import calculus as calc
    
    nx = x.shape[0]
    nz = x.shape[-1]
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    config = tracer.types.MagneticConfig()
    config.configIds = [1]


    #### Expand in phi to take derivatives
    phi_list = np.linspace(.99*phi0, 1.01*phi0, 9)
    r = x*np.cos(phi0) + y*np.sin(phi0)

    dz = z[0,1]-z[0,0]
    dx = x[1,0]-x[1,1]
    dphi = phi_list[1]-phi_list[0]

    r_extended = np.zeros((nx,9,nz))
    z_extended = np.zeros((nx,9,nz))
    phi_extended = np.zeros((nx,9,nz))
    
    
    for j in np.arange(0,phi_list.shape[0]):
        r_extended[:,j,:] = x*np.cos(phi_list[j]) + y*np.sin(phi_list[j])
        z_extended[:,j,:] = z
        phi_extended[:,j,:] = np.ones((nx,nz))*phi_list[j]
        
    points = tracer.types.Points3D()
    points.x1 = (r_extended*np.cos(phi_extended)).flatten()
    points.x2 = (r_extended*np.sin(phi_extended)).flatten()
    points.x3 = z_extended.flatten()

    res = tracer.service.magneticField(points, config)
    
    ## Reshape to 3d array
    Bx = np.ndarray.reshape(np.asarray(res.field.x1),(nx,9,nz))
    By = np.ndarray.reshape(np.asarray(res.field.x2),(nx,9,nz))
    Bz = np.ndarray.reshape(np.asarray(res.field.x3),(nx,9,nz))

    br = np.sqrt(Bx**2 + By**2)
    bphi = -Bx*np.sin(phi_extended) + By*np.cos(phi_extended)
    # bphi = np.arctan(By/Bx)
        
    b2 = (Bx**2 + By**2 + Bz**2)
    
    Bx /= b2
    By /= b2
    Bz /= b2
    bphi /= b2
    br /= b2

    dbphidz = np.zeros((nx,nz))
    dbrdz = np.zeros((nx,nz))
    dbzdphi = np.zeros((nx,nz))
    dbzdr = np.zeros((nx,nz))
    dbphidr = np.zeros((nx,nz))
    dbrdphi = np.zeros((nx,nz)) 
    curlbr = np.zeros((nx,nz))
    curlbphi = np.zeros((nx,nz))
    curlbz = np.zeros((nx,nz))
    
    for i in np.arange(0,nx):
        dbphidz[i,:] = calc.deriv(bphi[i,4,:])/dz
        dbrdz[i,:] = calc.deriv(br[i,4,:])/dz

    for k in np.arange(0,nz):
        dbzdr[:,k] = calc.deriv(Bz[:,4,k])/dx
        dbphidr[:,k] =  calc.deriv(bphi[:,4,k])/dx
        for i in np.arange(0,nx):
            dbzdphi[i,k] = calc.deriv(Bz[i,:,k])[4]/dphi
            dbrdphi[i,k] = calc.deriv(br[i,:,k])[4]/dphi
            curlbr[i,k]   = (dbzdphi[i,k] - dbphidz[i,k])  
            curlbphi[i,k] = (dbrdz[i,k] - dbzdr[i,k])
            curlbz[i,k]   = (dbphidr[i,k] - dbrdphi[i,k])

    curlbz   = 1/r

    return curlbr, curlbphi, curlbz

def GPI_Poincare(x,y,z, configuration=1, show_plot=False):
    '''
    Plots poincare plot given:
    input: 2D W7-X real-space coordinates (x,y,z), configuration
    returns: Field line tracer object 
    '''


    from osa import Client
    import matplotlib.pyplot as plt
    
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    config = tracer.types.MagneticConfig()
    config.configIds = configuration
    pos = tracer.types.Points3D()

    pos.x1 = np.linspace(0.95*np.mean(x[0,:]), 1.05*np.mean(x[-1,:]),80)
    pos.x2 = np.linspace(0.95*np.mean(y[0,:]), 1.05*np.mean(y[-1,:]),80)
    pos.x3 = np.linspace(0.95*np.mean(z[:,0]), 1.05*np.mean(z[:,-1]),80)
    
    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints = 400
    poincare.phi0 = [4.787] ## This is where the poincare plane is (bean=0, triangle = pi/5.)
    
    task = tracer.types.Task()
    task.step = 0.2
    task.poincare = poincare
    
    res = tracer.service.trace(pos, config, task, None, None)
    if show_plot:
        for i in range(0, len(res.surfs)):
            plt.scatter(res.surfs[i].points.x1, res.surfs[i].points.x3, color="black", s=0.5)

        plt.xlim(x[0,0],x[-1,-1])
        plt.ylim(z[0,0],z[-1,-1])
        plt.show()    

    return res

def plot_GPI_lpar(nx=132,nz=128, everything=False, specific_configuration=1):
    '''
    Plots connection length in a GPI-view 2D plane given:
    input: 2D resolution
    returns: saved file location
    '''
    import matplotlib.pyplot as plt
#    plt.ioff()  ## turns off window display after each plot
    
    configurations = np.arange(0,19)
    names = ['Standard (0)', 'Standard (1)', 'Standard (2)', 'Low iota', 'High iota', 'Low mirror', 'High mirror', 'Low shear', 'Inward Shift', 'Outward Shift', 'Limiter (10)', 'Standard(11)', 'Kamin', 'Standard (13)', 'Limiter (14)','Limiter (15)','Limiter (16)','Limiter (17)', 'Limiter (18)']
    


    if everything:
        for configuration, name in zip(configurations,names):
            x,y,z = GPI_view(nx,nz)
            r = x*np.cos(4.787) + y*np.sin(4.787)
            lengths = GPI_lpar(x,y,z, configuration=configuration, limit=500)
            filename = 'Lpar_plots/GPI_Lpar_configuration_'+str(configuration)+'.png'
            plt.figure(configuration)
            plt.contourf(r,z,lengths,100)
            plt.title(str(name), fontsize=18)
            plt.colorbar().set_label(r'$L_\parallel$ (m)', fontsize=16)
            plt.set_cmap('viridis')
            plt.ylabel('Z(m)', fontsize=16)
            plt.xlabel('R (m)', fontsize=16)
            plt.tick_params('both',labelsize=14)
            plt.tight_layout()
            plt.axis("equal")
            plt.savefig(filename, dpi=300)
            plt.close("all")

    else:
        x,y,z = GPI_view(nx,nz)
        r = x*np.cos(4.787) + y*np.sin(4.787)
        lengths = GPI_lpar(x,y,z, configuration=specific_configuration, limit=500)
        filename = 'Lpar_plots/GPI_Lpar_configuration_'+str(specific_configuration)+'.png'
        plt.contourf(r,z,lengths,100)
        plt.set_cmap('viridis')
        plt.colorbar().set_label(r'$L_\parallel$ (m)', fontsize=16)
        plt.ylabel('Z(m)', fontsize=16)
        plt.xlabel('R (m)', fontsize=16)
        plt.title(str(names[specific_configuration]), fontsize=18)
        plt.tick_params('both',labelsize=14)
        plt.tight_layout()
        plt.axis("equal")
        plt.savefig(filename, dpi=300)

    plt.show()
        
    return filename
