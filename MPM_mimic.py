from boutdata.collect import collect
from boututils.datafile import DataFile
import numpy as np
from boututils import calculus as calc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def synthetic_probe(path='.', t_range=[100, 150], detailed_return=False, t_min=50, pin_distance=[0.005, 0.005],
                    inclination=0):
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
    n = collect("Ne", path=path, info=False)
    Pe = collect("Pe", path=path, info=False)
    n0 = collect("Nnorm", path=path, info=False)
    T0 = collect("Tnorm", path=path, info=False)
    phi = collect("phi", path=path, info=False) * T0
    phi -= phi[0, 0, 0, 0]
    t_array = collect("t_array", path=path, info=False)
    wci = collect("Omega_ci", path=path, info=False)
    dt = (t_array[1] - t_array[0]) / wci
    rhos = collect('rho_s0', path=path, info=False)
    R0 = collect('R0', path=path, info=False) * rhos
    B0 = collect('Bnorm', path=path, info=False)
    dx = collect('dx', path=path, info=False) * rhos * rhos
    dz = collect('dz', path=path, info=False)
    Lx = ((dx.shape[0] - 4) * dx[0, 0]) / (R0)
    Lz = dz * R0 * n.shape[-1]

    tsample_size = t_range[-1] - t_range[0]
    trange = np.linspace(t_range[0], t_range[-1] - 1, tsample_size, dtype='int')


    n = n[:, :, 0, :]
    Pe = Pe[:, :, 0, :]
    
    I, J0, Imax, Imin = finding_Isat(n, Pe, n0,T0)

    trise, Epol,vr, events =geting_messurments(I, phi, B0, Lx, Lz, Imin, Imax,pin_distance,inclination,t_min)
    t_e, vr_CA, I_CA, twindow,event_indices = average_messurments(trise,Epol,vr,events,I,Imin, Imax,trange,dt)
    delta_real_mean= real_size(n, trange, tsample_size, Lx, Lz)
    v, pos_fit, pos, r, z, t = calc_com_velocity(path=path, fname=None, tmax=t_range[1] - 1)

    v_pol = (np.max(z) - z[t_range[0]]) / (dt * len(trange))
    delta_measured = t_e * v_pol
    velocity_error = (np.max(vr_CA) - np.max(v)) / np.max(v)
    blob_size_error = (delta_measured - delta_real_mean) / delta_real_mean

    print("Number of events: {} ".format(np.around(len(event_indices), decimals=2)))
    print("Size measurement error: {}% ".format(np.around(blob_size_error, decimals=2)))
    print("Velocity measurement error: {}% ".format(np.around(velocity_error, decimals=2)))

    if not detailed_return:
        return delta_measured, delta_real_mean, vr_CA, v[t_range[0]:]
    else:
        return delta_measured, delta_real_mean, vr_CA, v[t_range[0]:], I_CA - J0, len(event_indices), t_e, events, v_pol


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

    return trise, Epol,vr, events


def average_messurments(trise,Epol,vr,events,I, Imin, Imax,trange,dt):

    nt = I.shape[0]
    nx = I.shape[1]
    nz = I.shape[2]
    
    trise_flat = trise.flatten()
    events_flat = events.flatten()
    Epol_flat = Epol.reshape(nt, nx * nz)
    vr_flat = vr.reshape(nt, nx * nz)
    I_flat = I.reshape(nt, nx * nz)

    vr_offset = np.zeros((150, nx * nz))
    I_offset = np.zeros((150, nx * nz))
    event_indices = []
    for count in np.arange(0, nx * nz):
        for t in np.arange(trange[0], trange[-1]):
            if (t == np.int(trise_flat[count])):
                event_indices.append(count)
                vr_offset[:, count] = vr_flat[np.int(trise_flat[count]) - 50:np.int(trise_flat[count] + 100), count]
                I_offset[:, count] = I_flat[np.int(trise_flat[count]) - 50:np.int(trise_flat[count] + 100), count]

    vr_CA = np.mean(vr_offset[:, event_indices], axis=-1)
    I_CA = np.mean(I_offset[:, event_indices], axis=-1)
    twindow = np.linspace(-50 * dt, 100 * dt, 150)
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
