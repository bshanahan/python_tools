from boutdata.collect import collect
from boututils.datafile import DataFile
import numpy as np
from boututils import calculus as calc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def vary_inclination(path='.', t_range=[100,150], thetas=np.linspace(-0.5,0.5,20), show_plot=True, save_plot=False, fname='inclination.png'):
    sigma_d = np.zeros(thetas.shape)
    sigma_v = np.zeros(thetas.shape)
    for i,theta in zip(np.arange(0,thetas.shape[0]),thetas):
        d_m, d_s, vr_CA, v, I, sigma_d[i], sigma_v[i] = synthetic_probe(path=path, t_range=t_range, inclination=theta, return_error = True)

    if show_plot:
        plt.rc('font', family='Serif')
        plt.figure(figsize=(8,4.5))
        plt.plot(thetas, sigma_v, 'o')
        plt.grid(alpha=0.5)
        plt.xlabel(r'Inclination angle [rad]', fontsize=18)
        plt.ylabel(r'Error [%]', fontsize=18)
        plt.tick_params('both', labelsize=14)
        plt.tight_layout()
        if save_plot:
            plt.savefig(str(fname), dpi=300)
        else:
            plt.show()
    else:
        return sigma_v
    

def synthetic_probe(path='.',t_range=[100,150], detailed_return=False, return_error=False, t_min=50, pin_sep=5e-3, inclination=0., quiet=True):
    """ Synthetic MPM probe for 2D blob simulations
    Follows same conventions as Killer, Shanahan et al., PPCF 2020.

    input: 
    path:  to BOUT++ dmp files 
    t_range: time range for measurements (index)
    detailed_return (bool) -- whether or not extra information is returned.
    t_min: minimum time index for valid measurement
    pin_sep: vertical separation of V_fl pins to I_sat pin.
    inclination: misalignment of the probe wrt. poloidal motion. 

    returns: 
    delta_measured -- measured blob size
    delta_real_mean -- real blob size
    vr_CA -- radial velocity (conditionally averaged)
    v -- COM velocity
    I_CA -- conditionally-averaged Ion saturation current

    optionally returns: 
    len(event_indices) -- number of events measured
    t_e -- time-width of I_sat peak
    events -- locations of measurements
    size_error -- error in size measurement
    velocity_error -- error in velocity measurement
    """
    n  = collect("Ne", path=path, info=False)
    Pe = collect("Pe", path=path, info=False)
    n0 = collect("Nnorm", path=path, info=False)
    T0 = collect("Tnorm", path=path, info=False)
    phi = collect("phi",path=path, info=False)*T0
    phi -= phi[0,0,0,0]
    t_array = collect("t_array", path=path, info=False)
    wci = collect("Omega_ci", path=path, info=False)
    dt = (t_array[1]-t_array[0])/wci 
    rhos = collect('rho_s0', path=path, info=False)
    R0 = collect('R0',path=path, info=False)*rhos
    B0 = collect('Bnorm',path=path, info=False)
    dx = collect('dx', path=path, info=False)*rhos*rhos
    dz = collect('dz',path=path, info=False)
    Lx = ((dx.shape[0]-4)*dx[0,0])/(R0)
    Lz = dz * R0 * n.shape[-1]

    # print ("Synthetic Measurement Frequency: {} MHz".format(np.around(1e-6/dt,decimals=3)))

    tsample_size = t_range[-1]-t_range[0]
    trange = np.linspace(t_range[0],t_range[-1]-1, tsample_size, dtype='int')
    nt = n.shape[0]
    nx = n.shape[1]
    ny = n.shape[2]
    nz = n.shape[3]
    probe_offset_z = int((pin_sep/Lz)*nz)
    pin_sep_r = pin_sep*np.tan(inclination)
    probe_offset_r = int(((pin_sep_r)/Lx)*nx)

    Epol = np.zeros((nt,nx,nz))
    vr = np.zeros((nt,nx,nz))
    events = np.zeros((nx,nz))
    trise = np.zeros((nx,nz))

    n = n[:,:,0,:]
    Pe = Pe[:,:,0,:]
    
    Isat = get_Isat(n*n0,Pe*n0*T0, path=path)

    nmax,nmin = np.amax((n[0,:,:])),np.amin((n[0,:,:]))
    Imax,Imin = np.amax((Isat[0,:,:])),np.amin((Isat[0,:,:]))

    # Get measured rise time, calculated radial velocity
    for k in np.arange(0,nz):
        for i in np.arange(0,nx):
            if(np.any(n[t_min:,i,k] >  nmin+0.368*(nmax-nmin))):
                trise[i,k] = int(np.argmax(Isat[t_min:,i,k] > Imin+0.368*(Imax-Imin))+t_min)
                events[i,k] = 1
                Epol[:,i,k] = (phi[:,(i+probe_offset_r)%(nx-1),0, (k+probe_offset_z)%(nz-1)] -  phi[:,(i-probe_offset_r)%(nx-1),0, (k-probe_offset_z)%(nz-1)])/(2*np.sqrt(pin_sep**2 + pin_sep_r**2))
                vr[:,i,k] = Epol[:,i,k] / B0

    trise_flat = trise.flatten()
    events_flat = events.flatten()
    Epol_flat = Epol.reshape(nt,nx*nz)
    vr_flat = vr.reshape(nt,nx*nz)
    I_flat = Isat.reshape(nt,nx*nz)
    
    vr_offset = np.zeros((250,nx*nz))
    Isat_offset = np.zeros((250,nx*nz))
    event_indices = []
    # get measured velocity and density with same t==0, for CA.
    for count in np.arange(0,nx*nz):
        for t in np.arange(trange[0],trange[-1]):
            if (t==np.int(trise_flat[count])):
                event_indices.append(count)
                t_measurement = [min(max(np.int(trise_flat[count])-50,0),250), min(np.int(trise_flat[count])+200,500)]
                vr_offset[:,count] = vr_flat[t_measurement[0]:t_measurement[-1], count]
                Isat_offset[:,count] = I_flat[t_measurement[0]:t_measurement[-1], count]

    # Conditionally average.
    vr_CA = np.mean(vr_offset[:,event_indices], axis=-1)
    I_CA = np.mean(Isat_offset[:,event_indices], axis=-1)
    I_CA_max,I_CA_min = np.amax(I_CA),np.amin(I_CA)
    twindow = np.linspace(-50*dt, 200*dt, 250)
    tmin, tmax = np.min(twindow[I_CA > Imin+0.368*(Imax-Imin)]), np.max(twindow[I_CA > Imin+0.368*(Imax-Imin)])
    t_e = tmax-tmin

    ## get real velocity.
    v,pos_fit,pos,r,z,tcross = calc_com_velocity(path=path,fname=None,tmax=-1)

    ## The next 2 lines calculate the measured size, and could be more elegant...
    v_pol = np.mean(calc.deriv(z[50:int(nt/2)])/(dt))
    delta_measured = t_e*v_pol

    print (v_pol)
    # plt.plot(twindow, I_CA); plt.hlines(Imin+0.368*(Imax-Imin), 0, t_e); plt.show()
    ## calculate the actual size of the filament -- averaged over R/Z.
    n_real = n[trange,:,:]
    n_real_max = np.zeros((tsample_size))
    n_real_min = np.zeros((tsample_size))
    Rsize = np.zeros((tsample_size))
    Zsize = np.zeros((tsample_size))
    delta_real = np.zeros((tsample_size))
    for tind in np.arange(0,tsample_size):
        n_real_max[tind], n_real_min[tind] = np.amax((n[trange[tind],:,:])),np.amin((n[trange[tind],:,:]))
        n_real[tind,n_real[tind] < (n_real_min[tind]+0.368*(n_real_max[tind]-n_real_min[tind]))] = 0.0
        R = np.linspace(0,Lx,n.shape[1])
        Z = np.linspace(0,Lz,n.shape[-1])
        RR,ZZ = np.meshgrid(R,Z,indexing='ij')
        Rsize[tind] = np.max(RR[np.nonzero(n_real[tind])]) - np.min(RR[np.nonzero(n_real[tind])])
        Zsize[tind] = np.max(ZZ[np.nonzero(n_real[tind])]) - np.min(ZZ[np.nonzero(n_real[tind])])
        delta_real[tind] = np.mean([Rsize[tind],Zsize[tind]])

    delta_real_mean = np.mean(delta_real[0])
    size_error = np.around(100*np.abs(delta_measured-delta_real_mean)/delta_real_mean,decimals=2)
    velocity_error = np.around(100*np.abs(np.max(vr_CA)-np.max(v[trange]))/np.max(v[trange]),decimals=2)

    if not quiet:
        print ("Number of events: {} ".format(np.around(len(event_indices),decimals=2)))
        print ("Size measurement error: {}% ".format(size_error))
        print ("Velocity measurement error: {}% ".format(velocity_error))

    if not detailed_return:
        if return_error:
            return  delta_measured, delta_real_mean, vr_CA, v, I_CA , size_error, velocity_error 
        else:
            return  delta_measured, delta_real_mean, vr_CA, v, I_CA
    else:
        if return_error:
            return  delta_measured, delta_real_mean, vr_CA, v, I_CA, size_error, velocity_error, len(event_indices), t_e, trise
        else:
            return  delta_measured, delta_real_mean, vr_CA, v, I_CA, len(event_indices), t_e, trise

def get_Isat(n, Pe, path='.'):
    # Returns ion saturation current density in mA/mm**2
    # NOTE: Pin area is set.
    
    qe = 1.602176634e-19
    m_i = 1.672621898e-27

    Te = np.divide(Pe, n)

    Isat = n * 0.49 * qe * np.sqrt((qe * Te) / (m_i)) * (np.pi/2)* 1e-3

    T0 = collect('Tnorm', path=path, info=False) 
    n0 = collect('Nnorm', path=path, info=False)
    cs = collect('Cs0', path=path, info=False)
    J0 = n0 * 0.49 * qe * np.sqrt((qe * T0) / (m_i)) * (np.pi/2) * 1e-3
    # Isat = 0.49*n*cs*qe*8e-3
    return Isat-J0

def calc_com_velocity(path = "." ,fname=None, tmax=-1, track_peak=False):

    n  = collect("Ne", path=path, tind=[0,tmax], info=False)
    t_array = collect("t_array", path=path, tind=[0,tmax], info=False)
    wci = collect("Omega_ci", path=path, tind=[0,tmax],info=False)
    dt = (t_array[1]-t_array[0])/wci
    
    nt = n.shape[0]
    nx = n.shape[1]
    ny = n.shape[2]
    nz = n.shape[3]

    if fname is not None:
        fdata = DataFile(fname)

        R = fdata.read("R")
        Z = fdata.read("Z")

    else:
        R = np.zeros((nx,ny,nz))
        Z = np.zeros((nx,ny,nz))
        rhos = collect('rho_s0', path=path, tind=[0,tmax])
        Rxy = collect("R0", path=path, info=False)*rhos
        dx = (collect('dx', path=path, tind=[0,tmax],info=False)*rhos*rhos/(Rxy))[0,0]
        dz = (collect('dz', path=path, tind=[0,tmax],info=False)*Rxy)
        for i in np.arange(0,nx):
            for j in np.arange(0,ny):
                R[i,j,:] = dx*i
                for k in np.arange(0,nz):
                    Z[i,j,k] = dz*k
                    
        
    max_ind = np.zeros((nt,ny))
    fwhd = np.zeros((nt,nx,ny,nz))
    xval = np.zeros((nt,ny),dtype='int')
    zval = np.zeros((nt,ny),dtype='int')
    xpeakval = np.zeros((nt,ny))
    zpeakval = np.zeros((nt,ny))
    Rpos = np.zeros((nt,ny))
    Zpos = np.zeros((nt,ny))
    pos =  np.zeros((nt,ny))
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
            data[data < (nmin+0.368*(nmax-nmin))] = 0
            fwhd[t,:,y,:]=data
            ntot = np.sum(data[:,:])
            zval_float = np.sum(np.sum(data[:,:],axis=0)*(np.arange(nz)))/ntot
            xval_float = np.sum(np.sum(data[:,:],axis=1)*(np.arange(nx)))/ntot

            xval[t,y] = int(np.round(xval_float))
            zval[t,y] = int(np.round(zval_float))

            xpos,zpos = np.where(data[:,:]==nmax)		
            xpeakval[t,y] = xpos[0]
            zpeakval[t,y] = zpos[0]

            # # import pdb;pdb.set_trace()
            if track_peak:
                Rpos[t,y] = R[int(xpeakval[t,y]),y,int(zpeakval[t,y])]
                Zpos[t,y] = Z[int(xpeakval[t,y]),y,int(zpeakval[t,y])]
            else:
                Rpos[t,y] = R[xval[t,y],y,zval[t,y]]
                Zpos[t,y] = Z[xval[t,y],y,zval[t,y]]
            
        pos[:,y] = np.sqrt((Rpos[:,y]-Rpos[0,y])**2)# + (Zpos[:,y]-Zpos[0,y])**2)
        z1 = np.polyfit(t_array[:],pos[:,y],5)
        f = np.poly1d(z1)
        pos_fit[:,y] = f(t_array[:])

        t_cross = np.where(pos_fit[:,y]>pos[:,y])[0]
        t_cross = 0 #t_cross[0]

        pos_fit[:t_cross,y] = pos[:t_cross,y]

        z1 = np.polyfit(t_array[:],pos[:,y],5)
        f = np.poly1d(z1)
        pos_fit[:,y] = f(t_array[:])
        # pos_fit[:t_cross,y] = pos[:t_cross,y]

        v_fit[:,y] = calc.deriv(pos_fit[:,y])/dt

        # hole_fill = interp1d(t_array[::t_cross+2], v_fit[::t_cross+2,y] )
        
        # v_fit[:t_cross+1,y] = hole_fill(t_array[:t_cross+1])

        # # pos_index =1+ np.where(pos[:-1,y] != pos[1:,y])[0]
        # posunique, pos_index = np.unique(pos[:,y],return_index=True)
        # pos_index = np.sort(pos_index)
        # XX = np.vstack(( t_array[:]**5, t_array[:]**4,t_array[:]**3,t_array[:]**2,t_array[:], pos[pos_index[0],y]*np.ones_like(t_array[:]))).T

        # pos_fit_no_offset = np.linalg.lstsq(XX[pos_index,:-2],pos[pos_index,y],rcond=None)[0]
        # pos_fit[:,y] = np.dot(pos_fit_no_offset,XX[:,:-2].T)
        # v_fit[:,y] = calc.deriv(pos_fit[:,y])/dt

        #        # Take fit of raw velocity calculation
        # v  = calc.deriv(pos[:,y])/dt
        # z1 = np.polyfit(t_array[:],v,5)
        # f = np.poly1d(z1)
        # v_fit[:,y] = f(t_array[:])

    return v_fit[:,0], pos_fit[:,0], pos[:,0], Rpos[:,0], Zpos[:,0], t_cross

    
