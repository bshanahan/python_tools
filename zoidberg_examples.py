import zoidberg as zb
import numpy as np
from boututils.datafile import DataFile
import boututils.calculus as calc
import matplotlib.pyplot as plt
import random

def rotating_ellipse(nx=132,ny=16,nz=128,xcentre=3,I_coil=0.005,curvilinear=True,rectangular=False, fname='rotating-ellipse.fci.nc', a=0.5, curvilinear_inner_aligned=True, curvilinear_outer_aligned=True, npoints=100, Btor=2.5, show_maps=False, calc_curvature=True, smooth_curvature=False):
    yperiod = 2*np.pi
    field = zb.field.RotatingEllipse(xcentre = xcentre, I_coil=I_coil, radius = 2*a, yperiod = yperiod)
    # Define the y locations
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre+a/2.
    start_z = 0.
    R2 = np.ones((ny))
    iteration = 0
    if rectangular:
        print "Making rectangular poloidal grid"
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(nx, nz, 1.0, 1.0, Rcentre=xcentre)
    elif curvilinear:
        print "Making curvilinear poloidal grid"
        inner = zb.rzline.shaped_line(R0=xcentre, a=a/2., elong=0, triang=0.0, indent=0, n=npoints)
        outer = zb.rzline.shaped_line(R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints)

        if curvilinear_inner_aligned:
            print "Aligning to inner flux surface..."
            inner_lines = get_lines(field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints)
        if curvilinear_outer_aligned:
            print "Aligning to outer flux surface..."
            outer_lines = get_lines(field, start_r+a, start_z, ycoords, yperiod=yperiod, npoints=npoints)
        print "creating_grid..."
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
        calc_curvilinear_curvature(fname, field, grid)

    if (calc_curvature and smooth_curvature):
        smooth_field(fname, write_to_file=True, return_values=False)
        
def get_lines(field, start_r, start_z, yslices, yperiod=2*np.pi, npoints=150, smoothing=False):
    rzcoord, ycoords = zb.fieldtracer.trace_poincare(field, start_r, start_z, yperiod, y_slices=yslices, revs=npoints)

    # if smoothing:
    #     angles = np.linspace(0,2*np.pi,npoints/10) ##some equally-spaced angles for re-interpolating the lines.

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
            
def calc_curvilinear_curvature(fname, field, grid):
        f = DataFile(str(fname), write=True)
        B = f.read("B")
        dBydz = np.zeros(np.shape(B))
        dBydx = np.zeros(np.shape(B))
        dBxdz = np.zeros(np.shape(B))
        dBzdx = np.zeros(np.shape(B))
        for y in np.arange(0,B.shape[1]):
            pol,_ = grid.getPoloidalGrid(y)
            R = pol.R
            Z = pol.Z
            dx = grid.metric()["dx"]
            dz = grid.metric()["dz"]
            for x in np.arange(0,B.shape[0]):
                dBydz[x,y,:] = calc.deriv(field.Byfunc(R[x,:],y,Z[x,:]))/dz[x,y,:]
                dBxdz[x,y,:] = calc.deriv(field.Bxfunc(R[x,:],y,Z[x,:]))/dz[x,y,:]
            for z in np.arange(0,B.shape[-1]):
                dBzdx[:,y,z] = calc.deriv(field.Bzfunc(R[:,z],y,Z[:,z]))/dx[:,y,z]
                dBydx[:,y,z] = calc.deriv(field.Byfunc(R[:,z],y,Z[:,z]))/dx[:,y,z]
        bxcvx = dBydz
        bxcvy = dBxdz - dBzdx
        bxcvz = dBydx

        f.write('bxcvz', bxcvz)
        f.write('bxcvx', bxcvx)
        f.write('bxcvy', bxcvy)
        f.close()

def smooth_field(fname, write_to_file=False, return_values=False, order=7):
    from scipy.signal import savgol_filter
    f = DataFile(str(fname),write=True)
    B = f.read('B')
    bxcvx = f.read('bxcvx')
    bxcvz = f.read('bxcvz')
    bxcvy = f.read('bxcvy')
    bxcvx_smooth = np.zeros(bxcvx.shape)
    bxcvy_smooth = np.zeros(bxcvy.shape)
    bxcvz_smooth = np.zeros(bxcvz.shape)
    
    for y in np.arange(0,bxcvx.shape[1]):
        for x in np.arange(0,bxcvx.shape[0]):
            bxcvx_smooth[x,y,:] = savgol_filter(bxcvx[x,y,:],np.int(np.ceil(bxcvx.shape[-1]/2)//2*2+1),order)
            bxcvz_smooth[x,y,:] = savgol_filter(bxcvz[x,y,:],np.int(np.ceil(bxcvz.shape[-1]/2)//2*2+1),order)
            bxcvy_smooth[x,y,:] = savgol_filter(bxcvy[x,y,:],np.int(np.ceil(bxcvy.shape[-1]/2)//2*2+1),order)

    if(write_to_file):
        f.write('bxcvx',bxcvx_smooth)
        f.write('bxcvy',bxcvy_smooth)
        f.write('bxcvz',bxcvz_smooth)
    f.close()
    if(return_values):
        return bxcvx_smooth, bxcvy_smooth, bxcvz_smooth, bxcvx, bxcvy, bxcvz

def plot_RE_poincare(xcentre=3, I_coil=0.005, a=0.5, start_r = 3.25, start_z=0.0, npoints=100):
    yperiod = 2*np.pi
    field = zb.field.RotatingEllipse(xcentre = xcentre, I_coil=I_coil, radius = 2*a, yperiod = yperiod)
    zb.plot.plot_poincare(field, start_r, start_z, yperiod, revs=npoints)

if __name__ == '__main__':
    main()

    
