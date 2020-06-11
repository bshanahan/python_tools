from boutdata.collect import collect
from boututils.datafile import DataFile
import numpy as np
from boututils import calculus as calc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def calc_com_velocity(path = "." ,fname="rot_ell.curv.68.16.128.Ic_02.nc", tmax=-1):

    n  = collect("ne", path=path, tind=[0,tmax])
    t_array = collect("t_array", path=path, tind=[0,tmax])
    wci = collect("Omega_ci", path=path, tind=[0,tmax])
    dt = (t_array[1]-t_array[0])/wci
    
    nt = n.shape[0]
    nx = n.shape[1]
    ny = n.shape[2]
    nz = n.shape[3]

    fdata = DataFile(fname)

    R = fdata.read("R")
    Z = fdata.read("Z")

    dx = R[1,0,0]-R[0,0,0]
    max_ind = np.zeros((nt,ny))
    fwhd = np.zeros((nt,nx,ny,nz))
    xval = np.zeros((nt,ny),dtype='int')
    zval = np.zeros((nt,ny),dtype='int')
    xpeakval = np.zeros((nt,ny))
    zpeakval = np.zeros((nt,ny))
    Rpos = np.zeros((nt,ny))
    Zpos = np.zeros((nt,ny))
    pos =  np.zeros((nt,ny))
    x =  np.zeros((nt,ny))
    vr = np.zeros((nt,ny))
    vz = np.zeros((nt,ny))
    vtot = np.zeros((nt,ny))
    pos_fit = np.zeros((nt,ny))
    v_fit = np.zeros((nt,ny))
    Zposfit = np.zeros((nt,ny))
    RZposfit = np.zeros((nt,ny))
    
    for y in np.arange(0,ny):
        for t in np.arange(0,nt):
            # max_ind[t,y] = np.where(n[t,:,y,:] == np.max(n[t,:,y,:]))
            # R_max = 
            data = n[t,:,y,:]
            nmax,nmin = np.amax((data[:,:])),np.amin((data[:,:]))
            data[data<0.95*nmax] = 0.0
            fwhd[t,:,y,:]=data
            ntot = np.sum(data[:,:])
            zval_float = np.sum(np.sum(data[:,:],axis=0)*(np.arange(nz)))/ntot
            xval_float = np.sum(np.sum(data[:,:],axis=1)*(np.arange(nx)))/ntot
            xval[t,y] = int(round(xval_float))
            zval[t,y] = int(round(zval_float))

            xpos,zpos = np.where(data[:,:]==nmax)		
            xpeakval[t,y] = xpos[0]
            zpeakval[t,y] = zpos[0]
            # import pdb;pdb.set_trace()
            Rpos[t,y] = R[xval[t,y],y,zval[t,y]]
            Zpos[t,y] = Z[xval[t,y],y,zval[t,y]]
        
        pos[:,y] = np.sqrt((Rpos[:,y]-Rpos[0,y])**2 + (Zpos[:,y]-Zpos[0,y])**2)
        z1 = np.polyfit(t_array[:],pos[:,y],3)
        f = np.poly1d(z1)
        pos_fit[:,y] = f(t_array[:])

        t_cross = np.where(pos_fit[:,y]>pos[:,y])[0]
        t_cross = 0 #t_cross[0]

        pos_fit[:t_cross,y] = pos[:t_cross,y]

        x[:,y] = dx*(xval[:,y]-xval[0,y])

        z1 = np.polyfit(t_array[:],pos_fit[:,y],5)
        f = np.poly1d(z1)
        pos_fit[:,y] = f(t_array[:])
        # pos_fit[:t_cross,y] = pos[:t_cross,y]

        v_fit[:,y] = calc.deriv(pos_fit[:,y])/dt

        # hole_fill = interp1d(t_array[::t_cross+2], v_fit[::t_cross+2,y] )
        
        # v_fit[:t_cross+1,y] = hole_fill(t_array[:t_cross+1])

        # pos_index =1+ np.where(pos[:-1,y] != pos[1:,y])[0]
        # # posunique, pos_index = np.unique(pos[:,y],return_index=True)
        # pos_index = np.sort(pos_index)
        # XX = np.vstack(( t_array[:]**3,t_array[:]**2,t_array[:], pos[pos_index[0],y]*np.ones_like(t_array[:]))).T

        # pos_fit_no_offset = np.linalg.lstsq(XX[pos_index,:-2],pos[pos_index,y])[0]
        # pos_fit[:,y] = np.dot(pos_fit_no_offset,XX[:,:-2].T)
        # v_fit[:,y] = calc.deriv(pos_fit[:,y])/dt

        ### Take fit of raw velocity calculation
        v  = calc.deriv(Rpos[:,y]-Rpos[0,y])/dt
        vr = calc.deriv(x[:,y])/dt
        z1 = np.polyfit(t_array[:],vr,5)
        f = np.poly1d(z1)
        v_fit = f(t_array[:])

    return v_fit, pos_fit[:,0], pos[:,0], Rpos[:,0], Zpos[:,0], vr

def analyze(path = "." ,fname="rot_ell.curv.68.16.128.Ic_02.nc", plot=True):
    v, pos_fit, pos,r,z,t = calc_com_velocity(path,fname)
    t_array = collect("t_array", path=path)
    wci = collect("Omega_ci",path=path)
    t_array /= wci*1e-6
    ny = v.shape[1]
    v /= 1e3
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    tor_angle = np.linspace(0,2*np.pi, ny, endpoint=False)

    for y in np.arange(0,5):
        if plot:
            ax1.plot(t_array[:], v[:,y], label = '$\phi$ = %0.2f' % tor_angle[y], linewidth=3)
            ax1.set_xlabel('Time ($\mu$s)', fontsize=22)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel("Speed (km/s)", fontsize=22)
            ax2.plot(t_array[:], 1e2*pos_fit[:,y], '--', linewidth=3)
            ax2.set_ylabel("Position from origin (cm)", fontsize=22)


    ax1.legend(loc='best')
    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig('na_comp.eps')
    plt.show()
