from boutdata.collect import collect
from boututils.datafile import DataFile
import numpy as np
from boututils import calculus as calc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def synthetic_probe(path='.', t_range=[100, 150], detailed_return=False, t_min=50, pin_distance=np.array((0.005, 0.005)),
                    inclination=0,distSecondProbe=0):
    """ Synthetic MPM probe for 2D blob simulations
    Follows same conventions as Killer, Shanahan et al., PPCF 2020.

    input: 
    path to BOUT++ dmp files 
    time range for measurements (index)
    detailed_return (bool) -- whether or not extra information is returned.
    pin_distance --distance between pins:center pin and upper pin; center and lower pin
    inclination -- inclination of probe head to magnetic surfaces
    returns: 
    delta_measured -- measured blob size
    delta_real_mean -- real blob size
    vr_CA -- radial velocity (conditionally averaged)
    v -- COM velocity
    optionally returns: 
    I_CA-J0 -- conditionally-averaged Saturation current density
    len(event_indices) -- number of events measured
    t_e -- time-width of I_sat peak
    events -- locations of measurements
    """


    n, Pe,n0,T0,trange,Lx,Lz,B0,phi,tsample_size,dt, t_array =loading_data (path, t_range)
    
    I, J0, Imax, Imin = finding_Isat(n, Pe, n0,T0)

    trise, Epol,vr, event, sinclinationAngle =geting_messurments(I, phi, B0, Lx, Lz, Imin, Imax,pin_distance,inclination,t_min)

    t_e, vr_CA, I_CA, twindow,event_indices = average_messurments(trise,Epol,vr,events,I,Imin, Imax,trange,dt,t_A)
    
    vr_2Probes, delta_t_measured, twindow, delta_t, events_2Probes = second_prob( t_array, Lx, trise,t_range, distSecondProbe)


    delta_real_mean= real_size(n, trange, tsample_size, Lx, Lz)

    v, pos_fit, pos, r, z, t = calc_com_velocity(path=path, fname=None, tmax=t_range[1] - 1)

    

    v_pol = (np.max(z) - z[t_range[0]]) / (dt * len(trange))
    delta_measured = t_e * v_pol
    velocity_error_1Probe = (np.max(vr_CA) - np.max(v)) / np.max(v)
    velocity_error_2Probe = (np.max(vr_2Probes) - np.max(v)) / np.max(v)
    blob_size_error = (delta_measured - delta_real_mean) / delta_real_mean

    print("Number of events: {} ".format(np.around(len(event_indices), decimals=2)))
    print("Size measurement error: {}% ".format(np.around(blob_size_error, decimals=2)))
    print("Velocity measurement error 1 Probe: {}% ".format(np.around(velocity_error_1Probe, decimals=2)))
    print("Velocity measurement error 2 Probe: {}% ".format(np.around(velocity_error_2Probe, decimals=2)))
    if not detailed_return:
        return delta_measured, delta_real_mean, vr_CA, v[t_range[0]:]
    else:
        return delta_measured, delta_real_mean, vr_CA, v[t_range[0]:], I_CA - J0, len(event_indices), t_e, events, v_pol, vr_2Probes


def inclinationOfProbe(path,t_range=[150, 200],inclin = np.arange(-0.003, 0.0031, 0.0005), pin_distance=np.array((0.005, 0.005)),t_min=50,t_A=np.array((100,150))):

    n, Pe,n0,T0,trange,Lx,Lz,B0,phi,tsample_size,dt, t_array =loading_data (path, t_range)
    
    I, J0, Imax, Imin = finding_Isat(n, Pe, n0,T0)
    vr_CA=np.zeros((t_A[1]+t_A[0],inclin.shape[0]))
    inclinationAngle=np.zeros(inclin.shape[0])
    for ii in range(inclin.shape[0]):
        trise, Epol,vr, events,inclinationAngle[ii] =geting_messurments(I, phi, B0, Lx, Lz, Imin, Imax,pin_distance,inclin[ii],t_min)

        t_e, vr_CA[:,ii], I_CA, twindow,event_indices = average_messurments(trise,Epol,vr,events,I,Imin, Imax,trange,dt,t_A)


    delta_real_mean= real_size(n, trange, tsample_size, Lx, Lz)

    v, pos_fit, pos, r, z, t = calc_com_velocity(path=path, fname=None, tmax=t_range[1] - 1)

    v_pol = (np.max(z) - z[t_range[0]]) / (dt * len(trange))
    delta_measured = t_e * v_pol
    velocity_error_1Probe = (np.max(vr_CA, axis=0) - np.max(v)) / np.max(v)

    twindow*=1e6
    np.savez(path+"/inclination2",velocity_error_1Probe=velocity_error_1Probe,twindow=twindow, vr_CA=vr_CA, inclin=inclin,inclinationAngle=inclinationAngle)

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( twindow,vr_CA[:,0],'v', label=r'Inclination = '+ str(inclin[0])+'mm')
    plt.plot( twindow,vr_CA[:,3],'s', label=r'Inclination = '+ str(inclin[3])+'mm')
    plt.plot( twindow,vr_CA[:,6],'o', label=r'Inclination = '+ str(inclin[6])+'mm')
    plt.plot( twindow[50:100],v[t_range[0]:],'k--', label=r'Real velocity')

    plt.grid(alpha=0.5)
    plt.xlim(-100, 100)
    plt.xlabel(r't [$\mu s$]', fontsize=18)
    plt.ylabel(r'$\mathrm{v}$ [m/s]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'v_inclination2.png', dpi=300)
    plt.show()



    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(inclinationAngle,velocity_error_1Probe[:],'v', label=r'Velocity Error from Inclination')
    
    plt.grid(alpha=0.5)     
    plt.xlabel(r'Inclination Angle ', fontsize=18)
    plt.ylabel(r'Velocity error [$\%$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'inclination_error2.png', dpi=300)
    plt.show()
    return


def loading_data(path,t_range):
    n = collect("Ne", path=path, info=False)
    Pe = collect("Pe", path=path, info=False)
    n0 = collect("Nnorm", path=path, info=False)
    T0 = collect("Tnorm", path=path, info=False)
    phi = collect("phi", path=path, info=False) * T0
    phi -= phi[0, 0, 0, 0]
    wci = collect("Omega_ci", path=path, info=False)
    t_array = collect("t_array", path=path, info=False)/wci
    dt = (t_array[1] - t_array[0]) 
    rhos = collect('rho_s0', path=path, info=False)
    R0 = collect('R0', path=path, info=False) * rhos
    B0 = collect('Bnorm', path=path, info=False)
    dx = collect('dx', path=path, info=False) * rhos * rhos
    dz = collect('dz', path=path, info=False)
    Lx = ((dx.shape[0] - 4) * dx[0, 0]) / (R0)
    Lz = dz * R0 * n.shape[-1]

    tsample_size = t_range[-1] - t_range[0]
    trange = np.linspace(t_range[0], t_range[-1] - 1, tsample_size, dtype='int')
    nx = n.shape[1]
    

    n = n[:, :, 0, :]
    Pe = Pe[:, :, 0, :]
    return n, Pe,n0,T0,trange,Lx,Lz,B0,phi,tsample_size,dt, t_array

def second_prob(t_array,Lx,trise, t_range,distSecondProbe):
    """a second Probe is used to determine the radial velocity """
    nx = trise.shape[0]
    nz = trise.shape[1]
    dist_probeheads= int((distSecondProbe / Lx) * nx)
    delta_t=np.zeros((nx-dist_probeheads, nz))
    events_SecProbe=np.zeros((nx-dist_probeheads, nz), dtype=int)

    for k in np.arange(0, nz):
        for i in np.arange(0, nx-dist_probeheads):
            if (t_array[int(trise[i,k])]>0) and (t_array[int(trise[i+dist_probeheads,k])]>0):
                events_SecProbe[i, k] = 1
                delta_t[i,k]=t_array[int(trise[i+dist_probeheads,k])]-t_array[int(trise[i,k])]


    twindow1=np.zeros((nx-dist_probeheads, nz))
    twindow2=np.zeros((nx-dist_probeheads, nz))
    for k in np.arange(0, nz):
        for i in np.arange(0, nx-dist_probeheads):
            for t in np.arange(t_range[0], t_range[-1]):
                if (trise[i,k]>=t_range[0]) and (trise[i,k]<t_range[-1]):
                    twindow1[i,k]=1
                if (trise[i+dist_probeheads,k]>=t_range[0]) and (trise[i+dist_probeheads,k]<t_range[-1]):
                    twindow2[i,k]=1
    twindow=twindow1*twindow2

    delta_t_measured=np.mean(delta_t[twindow==1])   
    events_2Probes=len(delta_t[twindow==1])
    vr_2Probs=distSecondProbe/delta_t_measured

    return vr_2Probs, delta_t_measured, twindow, delta_t, events_2Probes


def finding_Isat(n, Pe, n0, T0):
    """ calcualtion the ion saturation current"""
    qe = 1.602176634e-19
    m_i = 1.672621898e-27

    P = Pe * T0 * n0
    Te = np.divide(P, n * n0)

    J_factor = n0 * 0.49 * qe * np.sqrt((qe * Te) / (m_i)) * 1e-3

    J0 = n0 * 0.49 * qe * np.sqrt((qe * T0) / (m_i)) * 1e-3
    I = np.multiply(n, J_factor)

    Imax, Imin = np.amax((I[0, :, :])), np.amin((I[0, :, :]))

    return I, J0, Imax, Imin


def geting_messurments(I, phi, B0, Lx, Lz, Imin, Imax,pin_distance,inclination,t_min ):

    nt = I.shape[0]
    nx = I.shape[1]
    nz = I.shape[2]
    

    probe_offset = [int((pin_distance[0] / Lz) * nz), int((pin_distance[1] / Lz) * nz)]
    probe_misalignment = int((inclination / Lx) * nx)
    # distance between outer probs
    d = np.sqrt((np.sum(pin_distance)) ** 2 + (2 * inclination) ** 2)
    inclinationAngle=np.arcsin(inclination/d)
    Epol = np.zeros((nt, nx, nz))
    vr = np.zeros((nt, nx, nz))
    events = np.zeros((nx, nz))
    trise = np.zeros((nx, nz))
    for k in np.arange(0, nz):
        for i in np.arange(0, nx):
            if (np.any(I[t_min:, i, k] > Imin + 0.368 * (Imax - Imin))):
                trise[i, k] = int(np.argmax(I[t_min:, i, k] > Imin + 0.368 * (Imax - Imin)) + t_min)
                events[i, k] = 1
                Epol[:, i, k] = (phi[:, (i + probe_misalignment), 0, (k + probe_offset[0]) % (nz - 1)] - phi[:, (i - probe_misalignment), 0, (k - probe_offset[1]) % (nz - 1)]) / d
                vr[:, i, k] = Epol[:, i, k] / B0

    return trise, Epol,vr, events,inclinationAngle


def average_messurments(trise,Epol,vr,events,I, Imin, Imax,trange,dt,t_A):

    nt = I.shape[0]
    nx = I.shape[1]
    nz = I.shape[2]
    
    trise_flat = trise.flatten()
    events_flat = events.flatten()
    Epol_flat = Epol.reshape(nt, nx * nz)
    vr_flat = vr.reshape(nt, nx * nz)
    I_flat = I.reshape(nt, nx * nz)

    vr_offset = np.zeros((t_A[1]+t_A[0], nx * nz))
    I_offset = np.zeros((t_A[1]+t_A[0], nx * nz))
    event_indices = []
    for count in np.arange(0, nx * nz):
        for t in np.arange(trange[0], trange[-1]):
            if (t == np.int(trise_flat[count])):
                event_indices.append(count)
                vr_offset[:, count] = vr_flat[np.int(trise_flat[count]) - t_A[0]:np.int(trise_flat[count] + t_A[1]), count]
                I_offset[:, count] = I_flat[np.int(trise_flat[count]) - t_A[0]:np.int(trise_flat[count] + t_A[1]), count]

    vr_CA = np.mean(vr_offset[:, event_indices], axis=-1)
    I_CA = np.mean(I_offset[:, event_indices], axis=-1)
    twindow = np.linspace(-t_A[0] * dt, t_A[1] * dt, t_A[1]+t_A[0])
    tmin, tmax = np.min(twindow[I_CA > Imin + 0.368 * (Imax - Imin)]), np.max(
        twindow[I_CA > Imin + 0.368 * (Imax - Imin)])
    t_e = tmax - tmin


    return t_e, vr_CA, I_CA, twindow,event_indices


def real_size(n, trange, tsample_size, Lx, Lz):
    n_real = n[trange, :, :]
    n_real_max = np.zeros((tsample_size))
    n_real_min = np.zeros((tsample_size))
    Rsize = np.zeros((tsample_size))
    Zsize = np.zeros((tsample_size))
    delta_real = np.zeros((tsample_size))
    for tind in np.arange(0, tsample_size):
        n_real_max[tind], n_real_min[tind] = np.amax((n[trange[tind], :, :])), np.amin((n[trange[tind], :, :]))
        n_real[tind, n_real[tind] < (n_real_min[tind] + 0.368 * (n_real_max[tind] - n_real_min[tind]))] = 0.0
        R = np.linspace(0, Lx, n.shape[1])
        Z = np.linspace(0, Lz, n.shape[-1])
        RR, ZZ = np.meshgrid(R, Z, indexing='ij')
        Rsize[tind] = np.max(RR[np.nonzero(n_real[tind])]) - np.min(RR[np.nonzero(n_real[tind])])
        Zsize[tind] = np.max(ZZ[np.nonzero(n_real[tind])]) - np.min(ZZ[np.nonzero(n_real[tind])])
        delta_real[tind] = np.mean([Rsize[tind], Zsize[tind]])

    delta_real_mean = np.mean(delta_real)


    return delta_real_mean


def calc_com_velocity(path=".", fname="rot_ell.curv.68.16.128.Ic_02.nc", tmax=-1, track_peak=False):
    n = collect("Ne", path=path, tind=[0, tmax], info=False)
    t_array = collect("t_array", path=path, tind=[0, tmax], info=False)
    wci = collect("Omega_ci", path=path, tind=[0, tmax], info=False)
    dt = (t_array[1] - t_array[0]) / wci

    nt = n.shape[0]
    nx = n.shape[1]
    ny = n.shape[2]
    nz = n.shape[3]

    if fname is not None:
        fdata = DataFile(fname)

        R = fdata.read("R")
        Z = fdata.read("Z")

    else:
        R = np.zeros((nx, ny, nz))
        Z = np.zeros((nx, ny, nz))
        rhos = collect('rho_s0', path=path, tind=[0, tmax])
        Rxy = collect("R0", path=path, info=False) * rhos
        dx = (collect('dx', path=path, tind=[0, tmax], info=False) * rhos * rhos / (Rxy))[0, 0]
        dz = (collect('dz', path=path, tind=[0, tmax], info=False) * Rxy)
        for i in np.arange(0, nx):
            for j in np.arange(0, ny):
                R[i, j, :] = dx * i
                for k in np.arange(0, nz):
                    Z[i, j, k] = dz * k

    max_ind = np.zeros((nt, ny))
    fwhd = np.zeros((nt, nx, ny, nz))
    xval = np.zeros((nt, ny), dtype='int')
    zval = np.zeros((nt, ny), dtype='int')
    xpeakval = np.zeros((nt, ny))
    zpeakval = np.zeros((nt, ny))
    Rpos = np.zeros((nt, ny))
    Zpos = np.zeros((nt, ny))
    pos = np.zeros((nt, ny))
    vr = np.zeros((nt, ny))
    vz = np.zeros((nt, ny))
    vtot = np.zeros((nt, ny))
    pos_fit = np.zeros((nt, ny))
    v_fit = np.zeros((nt, ny))
    Zposfit = np.zeros((nt, ny))
    RZposfit = np.zeros((nt, ny))

    for y in np.arange(0, ny):
        for t in np.arange(0, nt):

            data = n[t, :, y, :]
            nmax, nmin = np.amax((data[:, :])), np.amin((data[:, :]))
            data[data < (nmin + 0.368 * (nmax - nmin))] = 0
            fwhd[t, :, y, :] = data
            ntot = np.sum(data[:, :])
            zval_float = np.sum(np.sum(data[:, :], axis=0) * (np.arange(nz))) / ntot
            xval_float = np.sum(np.sum(data[:, :], axis=1) * (np.arange(nx))) / ntot

            xval[t, y] = int(np.round(xval_float))
            zval[t, y] = int(np.round(zval_float))

            xpos, zpos = np.where(data[:, :] == nmax)
            xpeakval[t, y] = xpos[0]
            zpeakval[t, y] = zpos[0]


            if track_peak:
                Rpos[t, y] = R[int(xpeakval[t, y]), y, int(zpeakval[t, y])]
                Zpos[t, y] = Z[int(xpeakval[t, y]), y, int(zpeakval[t, y])]
            else:
                Rpos[t, y] = R[xval[t, y], y, zval[t, y]]
                Zpos[t, y] = Z[xval[t, y], y, zval[t, y]]

        pos[:, y] = np.sqrt((Rpos[:, y] - Rpos[0, y]) ** 2)  
        z1 = np.polyfit(t_array[:], pos[:, y], 5)
        f = np.poly1d(z1)
        pos_fit[:, y] = f(t_array[:])

        t_cross = np.where(pos_fit[:, y] > pos[:, y])[0]
        t_cross = 0  # t_cross[0]

        pos_fit[:t_cross, y] = pos[:t_cross, y]

        z1 = np.polyfit(t_array[:], pos[:, y], 5)
        f = np.poly1d(z1)
        pos_fit[:, y] = f(t_array[:])


        v_fit[:, y] = calc.deriv(pos_fit[:, y]) / dt


        posunique, pos_index = np.unique(pos[:, y], return_index=True)
        pos_index = np.sort(pos_index)
        XX = np.vstack((t_array[:] ** 5, t_array[:] ** 4, t_array[:] ** 3, t_array[:] ** 2, t_array[:],
                        pos[pos_index[0], y] * np.ones_like(t_array[:]))).T

        pos_fit_no_offset = np.linalg.lstsq(XX[pos_index, :-2], pos[pos_index, y])[0]
        pos_fit[:, y] = np.dot(pos_fit_no_offset, XX[:, :-2].T)
        v_fit[:, y] = calc.deriv(pos_fit[:, y]) / dt



    return v_fit[:, 0], pos_fit[:, 0], pos[:, 0], Rpos[:, 0], Zpos[:, 0], t_cross
