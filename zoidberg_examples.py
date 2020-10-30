import zoidberg as zb
import numpy as np
from boututils.datafile import DataFile
import boututils.calculus as calc
import matplotlib.pyplot as plt
import random

def screwpinch(nx=68,ny=16,nz=128, xcentre=1.5, fname='screwpinch.fci.nc', a=0.2, npoints=421, show_maps=False):
    yperiod = 2*np.pi
    field = zb.field.Screwpinch(xcentre = xcentre, yperiod = yperiod, shear=0*2e-1)
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre+a/2.
    start_z = 0.
    print ("Making curvilinear poloidal grid")
    inner = zb.rzline.shaped_line(R0=xcentre, a=a/2., elong=0, triang=0.0, indent=0, n=npoints)
    outer = zb.rzline.shaped_line(R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints)
    poloidal_grid = zb .poloidal_grid.grid_elliptic(inner, outer,
                                                   nx, nz)
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps =  zb.make_maps(grid, field)
    zb.write_maps(grid,field,maps,str(fname),metric2d=False)
    calc_curvilinear_curvature(fname, field, grid, maps)

    if show_maps:
        zb.plot.plot_forward_map(grid, maps, yslice=0)

def rotating_ellipse(nx=68,ny=16,nz=128,xcentre=5.5,I_coil=0.01,curvilinear=True,rectangular=False, fname='rotating-ellipse.fci.nc', a=0.4, curvilinear_inner_aligned=True, curvilinear_outer_aligned=True, npoints=421, Btor=2.5, show_maps=False, calc_curvature=True, smooth_curvature=False, return_iota=True, write_iota=False):
    yperiod = 2*np.pi/5.
    field = zb.field.RotatingEllipse(xcentre = xcentre, I_coil=I_coil, radius = 2*a, yperiod = yperiod, Btor=Btor)
    # Define the y locations
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre+a/2.
    start_z = 0.

    if rectangular:
        print ("Making rectangular poloidal grid")
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(nx, nz, 1.0, 1.0, Rcentre=xcentre)
    elif curvilinear:
        print ("Making curvilinear poloidal grid")
        inner = zb.rzline.shaped_line(R0=xcentre, a=a/2., elong=0, triang=0.0, indent=0, n=npoints)
        outer = zb.rzline.shaped_line(R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints)

        if curvilinear_inner_aligned:
            print ("Aligning to inner flux surface...")
            inner_lines = get_lines(field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints)
        if curvilinear_outer_aligned:
            print ("Aligning to outer flux surface...")
            outer_lines = get_lines(field, xcentre+a, start_z, ycoords, yperiod=yperiod, npoints=npoints)
            
        print ("creating grid...")
        if curvilinear_inner_aligned:
            if curvilinear_outer_aligned:
                poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner, outer in zip(inner_lines, outer_lines) ]
            else:
                poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner in inner_lines ]
        else:
            poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer,
                                              nx, nz)
    
    # Create the 3D grid by putting together 2D poloidal grids
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps =  zb.make_maps(grid, field)
    zb.write_maps(grid,field,maps,str(fname),metric2d=False)

    if (curvilinear and calc_curvature):
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid, maps)

    if (calc_curvature and smooth_curvature):
        smooth_metric(fname, write_to_file=True, return_values=False, smooth_metric=True)

    if (return_iota or write_iota):
        rindices = np.linspace(start_r, xcentre+a, nx)
        zindices = np.zeros((nx))
        iota_bar = calc_iota(field, start_r, start_z)
        if (write_iota):
            f = DataFile(str(fname), write=True)
            f.write('iota_bar', iota_bar)
            f.close()
        else:
            print ("Iota_bar = ", iota_bar)

def dommaschk(nx=68,ny=16,nz=128, C=None, xcentre=1.0, Btor=1.0, a=0.1, curvilinear=True,rectangular=False, fname='Dommaschk.fci.nc', curvilinear_inner_aligned=True, curvilinear_outer_aligned=True, npoints=421, show_maps=False, calc_curvature=True, smooth_curvature=False, return_iota=True, write_iota=False):

    if C is None:
        C = np.zeros((6,5,4))
        C[5,2,1] = .4 
        C[5,2,2] = .4 
        # C[5,4,1] = 19.25
        
    yperiod = 2*np.pi/5.
    field = zb.field.DommaschkPotentials(C, R_0=xcentre, B_0=Btor)
    # Define the y locations
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre+a/2.
    start_z = 0.

    if rectangular:
        print ("Making rectangular poloidal grid")
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(nx, nz, 1.0, 1.0, Rcentre=xcentre)
    elif curvilinear:
        print ("Making curvilinear poloidal grid")
        inner = zb.rzline.shaped_line(R0=xcentre, a=a/2., elong=0, triang=0.0, indent=0, n=npoints)
        outer = zb.rzline.shaped_line(R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints)

        if curvilinear_inner_aligned:
            print ("Aligning to inner flux surface...")
            inner_lines = get_lines(field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints)
        if curvilinear_outer_aligned:
            print ("Aligning to outer flux surface...")
            outer_lines = get_lines(field, xcentre+a, start_z, ycoords, yperiod=yperiod, npoints=npoints)
            
        print ("creating grid...")
        if curvilinear_inner_aligned:
            if curvilinear_outer_aligned:
                poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner, outer in zip(inner_lines, outer_lines) ]
            else:
                poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner in inner_lines ]
        else:
            poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer,
                                              nx, nz)
    
    # Create the 3D grid by putting together 2D poloidal grids
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps =  zb.make_maps(grid, field)
    zb.write_maps(grid,field,maps,str(fname),metric2d=False)

    if (curvilinear and calc_curvature):
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid, maps)

    if (calc_curvature and smooth_curvature):
        smooth_metric(fname, write_to_file=True, return_values=False, smooth_metric=False)

    if (return_iota or write_iota):
        rindices = np.linspace(start_r, xcentre+a, nx)
        zindices = np.zeros((nx))
        iota_bar = calc_iota(field, start_r, start_z)
        if (write_iota):
            f = DataFile(str(fname), write=True)
            f.write('iota_bar', iota_bar)
            f.close()
        else:
            print ("Iota_bar = ", iota_bar)
def W7X(nx=68,ny=32,nz=256,fname='W7-X.fci.nc', vmec_file='w7-x.wout.nc', inner_VMEC=False, inner_vacuum=False, outer_VMEC=False, outer_vacuum=False, outer_vessel=False, npoints=100, a=2.5, show_maps=False, calc_curvature=True, smooth_curvature=False, plasma_field=False, configuration=0, vmec_url='http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/w7x_ref_171/wout.nc'):

    yperiod = 2*np.pi/5.
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    if outer_VMEC:
        print ("Aligning to outer VMEC flux surface...")
        outer_lines = get_VMEC_surfaces(phi=ycoords,s=1,npoints=nz, w7x_run=vmec_url)
    elif outer_vessel:
        print ("Aligning to plasma vessel (EXPERIMENTAL) ...")
        outer_lines = get_W7X_vessel(phi=ycoords, nz=nz)
    elif (outer_vacuum or inner_vacuum):
        xmin = 4.05
        xmax = 4.05+2.5
        zmin = -1.35
        zmax = -zmin
        field = zb.field.W7X_vacuum(phimax=2*np.pi/5., x_range=[xmin,xmax], z_range=[zmin,zmax], include_plasma_field=plasma_field)
        xcentre, zcentre = field.magnetic_axis(phi_axis=ycoords[0],configuration=configuration)
        start_r = xcentre+a/2.
        start_z = zcentre
        if inner_vacuum:
            print ("Aligning to inner vacuum flux surface...")
            inner_lines = get_lines(field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints)
        if outer_vacuum:
            print ("Aligning to outer vacuum flux surface...")
            outer_lines = get_lines(field, start_r+a, start_z, ycoords, yperiod=yperiod, npoints=npoints)
            
    xmin = np.min([min(outer_lines[i].R) for i in range(ny)])
    xmax = np.max([max(outer_lines[i].R) for i in range(ny)]) 
    zmin = np.min([min(outer_lines[i].Z) for i in range(ny)])
    zmax = np.max([max(outer_lines[i].Z) for i in range(ny)])

    if inner_VMEC:
        print ("Aligning to inner VMEC flux surface...")
        inner_lines = get_VMEC_surfaces(phi=ycoords,s=0.67,npoints=nz, w7x_run=vmec_url)

    print ("creating grid...")
    poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner, outer in zip(inner_lines, outer_lines) ]

    # Create the 3D grid by putting together 2D poloidal grids
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)

    field = zb.field.W7X_vacuum(phimax=2*np.pi/5., x_range=[xmin,xmax], z_range=[zmin,zmax], include_plasma_field=plasma_field)
    
    maps =  zb.make_maps(grid, field)
    zb.write_maps(grid,field,maps,str(fname),metric2d=False)

    if calc_curvature:
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid)

    if (calc_curvature and smooth_curvature):
        smooth_metric(fname, write_to_file=True, return_values=False, smooth_metric=True)

    if show_maps:
        zb.plot.plot_forward_map(grid, maps, yslice=-1)

def get_lines(field, start_r, start_z, yslices, yperiod=2*np.pi, npoints=150, smoothing=False):
    rzcoord, ycoords = zb.fieldtracer.trace_poincare(field, start_r, start_z, yperiod, y_slices=yslices, revs=npoints)

    lines = []
    for i in range(ycoords.shape[0]):
        r = rzcoord[:,i,0,0]
        z = rzcoord[:,i,0,1]
        line = zb.rzline.line_from_points(r,z)
        # Re-map the points so they're approximately uniform in distance along the surface
        # Note that this results in some motion of the line
        line = line.equallySpaced()
        lines.append(line)
            
    return lines


### return a VMEC flux surface as a RZline object
def get_VMEC_surfaces(phi=[0],s=0.75,w7x_run='w7x_ref_1',npoints=100):
    from osa import Client
    client =  Client("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl")
    points = client.service.getFluxSurfaces(str(w7x_run), phi, s, npoints)

    lines = []
    for y in range(len(phi)):
        r, z = points[y].x1, points[y].x3
        line = zb.rzline.line_from_points(r,z)
        line = line.equallySpaced()
        lines.append(line)

    return lines

### Return the W7X PFC as a RZline object
def get_W7X_vessel(phi=[0],nz=256):
    from osa import Client
    import matplotlib.path as path
    
    srv2 = Client("http://esb.ipp-hgw.mpg.de:8280/services/MeshSrv?wsdl")
    #describe geometry
    mset = srv2.types.SurfaceMeshSet() # a set of surface meshes
    mset.references = []

    #add references to single components, in this case ports
    w1 = srv2.types.SurfaceMeshWrap()
    ref = srv2.types.DataReference()
    ref.dataId = "371" # component id
    w1.reference = ref
    # w2 = srv2.types.SurfaceMeshWrap()
    # ref = srv2.types.DataReference()
    # ref.dataId = "341" # component id
    # w2.reference = ref
    mset.meshes = [w1]#, w2]
    
    lines = []
    for y in range(phi.shape[0]):
        # intersection call for phi_vals=phi
        result = srv2.service.intersectMeshPhiPlane(phi[y], mset)
        all_vertices = np.zeros((len(result),2))
        R = np.zeros((len(result)))
        z = np.zeros((len(result)))

        # plotting
        for s,i in zip(result, np.arange(0,len(result))): #loop over non-empty triangle intersections
            xyz = np.array((s.vertices.x1, s.vertices.x2, s.vertices.x3)).T
            R[i] = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)[0] # major radius
            z[i] = xyz[:,2][0]
            all_vertices[i,:]= ((R[i], z[i]))


        # now have all vertices of vessel, but need to put them in sensible order...

        # find magnetic axis
        # axis_line = get_VMEC_surfaces(phi=[phi[y]],s=0,npoints=20)
        # axis = [axis_line[0].R[0], axis_line[0].Z[0]]

        # loc = np.asarray([all_vertices[:,0] - axis[0], all_vertices[:,1] - axis[1]])
    
        # theta = np.arctan2(loc[1,:],loc[0,:])

        # sorted_indices = sorted(range(len(theta)), key=lambda k: theta[k])

        # r, z = [all_vertices[sorted_indices][::4,0], all_vertices[sorted_indices][::4,1]]

        Path = path.Path(all_vertices, closed=True)
        r, z = [all_vertices[::3,0], all_vertices[::3,1]]
        line = zb.rzline.line_from_points(r,z)
        line = line.equallySpaced(n=nz)
        lines.append(line)
    
    return lines

## calculate curvature for curvilinear grids
def calc_curvilinear_curvature(fname, field, grid, maps):
        from scipy.signal import savgol_filter

        f = DataFile(str(fname), write=True)
        B = f.read("B")

        dx = grid.metric()["dx"]
        dz = grid.metric()["dz"]
        g_11 = grid.metric()["g_xx"]
        g_22 = grid.metric()["g_yy"]
        g_33 = grid.metric()["g_zz"]
        g_12 = 0.0
        g_13 = grid.metric()["g_xz"]
        g_23 = 0.0

        GR = np.zeros(B.shape)
        GZ = np.zeros(B.shape)
        Gphi = np.zeros(B.shape)
        dRdz = np.zeros(B.shape)
        dZdz = np.zeros(B.shape)
        dRdx = np.zeros(B.shape)
        dZdx = np.zeros(B.shape)
        
        for y in np.arange(0,B.shape[1]):
            pol,_ = grid.getPoloidalGrid(y)
            R = pol.R
            Z = pol.Z
            # G = \vec{B}/B, here in cylindrical coordinates
            GR[:,y,:] = field.Bxfunc(R,y,Z)/((B[:,y,:])**2)
            GZ[:,y,:] = field.Bzfunc(R,y,Z)/((B[:,y,:])**2)
            Gphi[:,y,:] = field.Byfunc(R,y,Z)/((B[:,y,:])**2)
            for x in np.arange(0,B.shape[0]):
                dRdz[x,y,:] = calc.deriv(R[x,:])/dz[x,y,:]
                dZdz[x,y,:] = calc.deriv(Z[x,:])/dz[x,y,:]
            for z in np.arange(0,B.shape[-1]):
                dRdx[:,y,z] = calc.deriv(R[:,z])/dx[:,y,z]
                dZdx[:,y,z] = calc.deriv(Z[:,z])/dx[:,y,z]

        R = f.read("R")
        Z = f.read("Z")
        dy = f.read("dy")

        ## calculate Jacobian and contravariant terms in curvilinear coordinates
        J = R * (dZdz * dRdx - dZdx * dRdz )
        Gx = (GR*dZdz - GZ*dRdz)*(R/J)
        Gz = (GZ*dRdx - GR*dZdx)*(R/J)
        
        G_x = Gx*g_11 + Gphi*g_12 + Gz*g_13
        G_y = Gx*g_12 + Gphi*g_22 + Gz*g_23
        G_z = Gx*g_13 + Gphi*g_23 + Gz*g_33

        dG_zdy = np.zeros(B.shape)
        dG_ydz = np.zeros(B.shape)
        dG_xdz = np.zeros(B.shape)
        dG_zdx = np.zeros(B.shape)
        dG_ydx = np.zeros(B.shape)
        dG_xdy = np.zeros(B.shape)
        for y in np.arange(0,B.shape[1]):
            for x in np.arange(0,B.shape[0]):
                dG_ydz[x,y,:] = calc.deriv(G_y[x,y,:])/dz[x,y,:]
                dG_xdz[x,y,:] = calc.deriv(G_x[x,y,:])/dz[x,y,:]
            for z in np.arange(0,B.shape[-1]):
                dG_ydx[:,y,z] = calc.deriv(G_y[:,y,z])/dx[:,y,z]
                dG_zdx[:,y,z] = calc.deriv(G_z[:,y,z])/dx[:,y,z]

        #this should really use the maps...
        for x in np.arange(0,B.shape[0]):
            for z in np.arange(0,B.shape[-1]):
                dG_zdy[x,:,z] = calc.deriv(G_z[x,:,z])/dy[x,:,z]
                dG_xdy[x,:,z] = calc.deriv(G_x[x,:,z])/dy[x,:,z]

        bxcvx = (dG_zdy - dG_ydz)/J
        bxcvy = (dG_xdz - dG_zdx)/J
        bxcvz = (dG_ydx - dG_xdy)/J
        bxcv = g_11*(bxcvx**2) + g_22*(bxcvy**2) + g_33*(bxcvz**2) + 2*(bxcvz*bxcvx*g_13) 
        f.write('bxcvx', bxcvx)
        f.write('bxcvy', bxcvy)
        f.write('bxcvz', bxcvz)
        f.write('J', J)
        f.close()

## smooth the metric tensor components
def smooth_metric(fname, write_to_file=False, return_values=False, smooth_metric=True, order=7):
    from scipy.signal import savgol_filter
    f = DataFile(str(fname),write=True)
    B = f.read('B')
    bxcvx = f.read('bxcvx')
    bxcvz = f.read('bxcvz')
    bxcvy = f.read('bxcvy')
    J = f.read('J')

    bxcvx_smooth = np.zeros(bxcvx.shape)
    bxcvy_smooth = np.zeros(bxcvy.shape)
    bxcvz_smooth = np.zeros(bxcvz.shape)
    J_smooth = np.zeros(J.shape)

    if smooth_metric:
        g13 = f.read('g13')
        g_13 = f.read('g_13')
        g11 = f.read('g11')
        g_11 = f.read('g_11')
        g33 = f.read('g33')
        g_33 = f.read('g_33')

        g13_smooth = np.zeros(g13.shape)
        g_13_smooth = np.zeros(g_13.shape)
        g11_smooth = np.zeros(g11.shape)
        g_11_smooth = np.zeros(g_11.shape)
        g33_smooth = np.zeros(g33.shape)
        g_33_smooth = np.zeros(g_33.shape)

    
    
    for y in np.arange(0,bxcvx.shape[1]):
        for x in np.arange(0,bxcvx.shape[0]):
            bxcvx_smooth[x,y,:] = savgol_filter(bxcvx[x,y,:],np.int(np.ceil(bxcvx.shape[-1]/2)//2*2+1),order)
            bxcvz_smooth[x,y,:] = savgol_filter(bxcvz[x,y,:],np.int(np.ceil(bxcvz.shape[-1]/2)//2*2+1),order)
            bxcvy_smooth[x,y,:] = savgol_filter(bxcvy[x,y,:],np.int(np.ceil(bxcvy.shape[-1]/2)//2*2+1),order)
            J_smooth[x,y,:] = savgol_filter(J[x,y,:],np.int(np.ceil(J.shape[-1]/2)//2*2+1),order)
            if smooth_metric:
                g11_smooth[x,y,:] = savgol_filter(g11[x,y,:],np.int(np.ceil(g11.shape[-1]/2)//2*2+1),order)
                g_11_smooth[x,y,:] = savgol_filter(g_11[x,y,:],np.int(np.ceil(g_11.shape[-1]/2)//2*2+1),order)
                g13_smooth[x,y,:] = savgol_filter(g13[x,y,:],np.int(np.ceil(g13.shape[-1]/2)//2*2+1),order)
                g_13_smooth[x,y,:] = savgol_filter(g_13[x,y,:],np.int(np.ceil(g_13.shape[-1]/2)//2*2+1),order)
                g33_smooth[x,y,:] = savgol_filter(g33[x,y,:],np.int(np.ceil(g33.shape[-1]/2)//2*2+1),order)
                g_33_smooth[x,y,:] = savgol_filter(g_33[x,y,:],np.int(np.ceil(g_33.shape[-1]/2)//2*2+1),order)
                

    if(write_to_file):
        # f.write('bxcvx',bxcvx_smooth)
        # f.write('bxcvy',bxcvy_smooth)
        # f.write('bxcvz',bxcvz_smooth)
        f.write('J',J_smooth)

        if smooth_metric:
            f.write('g11',g11_smooth)
            f.write('g_11',g_11_smooth)
            f.write('g13',g13_smooth)
            f.write('g_13',g_13_smooth)
            f.write('g33',g33_smooth)
            f.write('g_33',g_33_smooth)
            
    f.close()
    if(return_values):
        return bxcvx_smooth, bxcvy_smooth, bxcvz_smooth, bxcvx, bxcvy, bxcvz

def plot_RE_poincare(xcentre=3, I_coil=0.005, a=0.5, start_r = 3.25, start_z=0.0, npoints=100):
    yperiod = 2*np.pi
    field = zb.field.RotatingEllipse(xcentre = xcentre, I_coil=I_coil, radius = 2*a, yperiod = yperiod)
    zb.plot.plot_poincare(field, start_r, start_z, yperiod, revs=npoints)

def calc_iota(field, start_r, start_z):
    from scipy.signal import argrelextrema
    toroidal_angle = np.linspace(0.0, 400*np.pi, 10000, endpoint=False)
    result = zb.fieldtracer.FieldTracer.follow_field_lines(field,start_r,start_z,toroidal_angle)
    peaks = argrelextrema(result[:,0,0], np.greater, order=10)[0]
    peak_locations = [result[i,0,0] for i in peaks]
    # print (peak_locations, peaks)
    iota_bar = 2*np.pi/(toroidal_angle[peaks[1]]-toroidal_angle[peaks[0]])
    # plt.plot(toroidal_angle, result[:,0,0]); plt.show()
    return iota_bar

def plot_maps(field, grid, maps, yslice=0):
    pol, ycoord = grid.getPoloidalGrid(yslice)
    pol_next, ycoord_next = grid.getPoloidalGrid(yslice+1)

    plt.plot(pol.R, pol.Z, 'x')

    import pdb; pdb.set_trace()
    # Get the coordinates which the forward map corresponds to
    R_next, Z_next = pol_next.getCoordinate(maps['forward_xt_prime'][:,yslice,:], maps['forward_zt_prime'][:,yslice,:] )
    
    plt.plot(R_next, Z_next, 'o')
    
    plt.show()

    
if __name__ == '__main__':
    main()

    
