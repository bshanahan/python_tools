import numpy as np

def cross_field_flux(n, phi, dx, dz, bf=1.0, norm=True, return_v=False):
    assert n.shape == phi.shape
    if len(n.shape) > 3:
        nt = n.shape[0]
    nx = phi.shape[-3]
    ny = phi.shape[-2]
    nz = phi.shape[-1]
    dphidz = np.zeros((nx,ny,nz))
    dphidx = np.zeros((nx,ny,nz))
#    for t in np.arange(0,nt):
# print(100.*t/nt, "%")
        # for y in np.arange(0,ny):
    for x in np.arange(0,nx):
        dphidz[x,0,:] = np.gradient(phi[x,0,:])/dz[x,:]
    for z in np.arange(0,nz):
        dphidx[:,0,z] = np.gradient(phi[:,0,z])/dx[:,z]
    vperp = bf*(dphidz - dphidx)

    nvx = bf*(n*dphidz)
    vx = bf*(dphidz)
    nvz = bf*(-n*dphidx)
    vz = bf*(-dphidx)
    Gamma_perp = np.abs(n*vperp)
    if norm:
        Gamma_perp /= np.max(Gamma_perp)
        nvx /= np.max(np.abs(nvx))
        nvz /= np.max(np.abs(nvz))
        vx /= np.max(np.abs(vx))
        vz /= np.max(np.abs(vz))
    if return_v:
        return Gamma_perp, nvx, nvz, vx, vz
    else:
        return Gamma_perp, nvx, nvz

def c_correlate(s_1, s_2, lags):
    """
    Numpy implementation of c_correlate.pro IDL routine
    """
    # ensure signals are of equal length
    assert s_1.shape == s_2.shape
    n_s = s_1.shape[0]
    # center both signals
    s_1_center = s_1 - s_1.mean()
    s_2_center = s_2 - s_2.mean()
    # allocate space for correlation
    correlation = np.zeros(lags.shape)
    # iterate over lags
    for i,l in enumerate(lags):
        if l >= 0:
            tmp = s_1_center[:(n_s - l)] * s_2_center[l:]
        else:
            tmp = s_1_center[-l:] * s_2_center[:(n_s + l)]
        correlation[i] = tmp.sum()
    # Divide by standard deviation of both
    correlation /= np.sqrt((s_1_center**2).sum() * (s_2_center**2).sum())
    
    return correlation

def cross_correlation(a, b, twindow=[20,-1], xindex=None, yindex=0, zindex=None, zeromean=True, resolution=40):
    if xindex is None:
        nx = a.shape[1]
        xindex = int(nx/2)
    if zindex is None:
        nz = a.shape[-1]
        zindex = int(nz/2)
        
    t0, t1 = twindow
    a_std = np.std(a[t0:t1,xindex,yindex,zindex],axis=0)
    b_std = np.std(b[t0:t1,xindex,yindex,zindex],axis=0)

    a_norm = a[t0:t1]/a_std
    b_norm = b[t0:t1]/b_std

    if zeromean:
        a_norm -= np.mean(a_norm)
        b_norm -= np.mean(b_norm)

    low = -20
    high = 20

    d = (high-low)/resolution

    result = np.zeros((resolution,resolution))
    for i in np.arange(0,resolution):
        amin = low + i*d
        amax = amin + d
        for j in np.arange(0,resolution):
            bmin = low+j*d
            bmax = bmin+d

            result[i,j] = np.count_nonzero((a_norm >= amin) & (a_norm < amax) & (b_norm >= bmin) & (b_norm < bmax))

    result /= np.sum(result)

    return result 

def phase_shift(a,b, rhok1=1, truncate=0.2, nphase=42):

    nt = a.shape[0]
    nx = a.shape[1]
    ny = a.shape[2]
    nz = a.shape[-1]

    if (nx%2 == 1):
        nx -= 1

    nf = int(nx/2 * 1.-truncate)

    fa = np.fft.fft(a, axis=0)
    fb = np.fft.fft(b, axis=0)

    phase = np.imag(np.log(fa/fb)) / np.pi
    
    dp = 2./nphase

    result = np.zeros((nphase,nf))
    tmpresult = np.zeros((nt,nphase,nf))

    for i in np.arange(0,nphase):
        pmin = -1. + i * dp
        pmax = pmin + dp
        for f in np.arange(0,nf):
            for t in np.arange(0,nt):
                tmpresult[t,i,f] = np.count_nonzero((phase[t,:,0,f] >= pmin) & (phase[t,:,0,f] < pmax))

            result[i,f] = np.mean(tmpresult[:,i,f])

    return result

def rms(a,tmin=0):
    nx = a.shape[1]
    ny = a.shape[2]
    nz = a.shape[-1]

    a_rms = np.zeros((nx,ny,nz)) 
    for i in np.arange(nx):
        for j in np.arange(ny):
            for k in np.arange(nz):
                a_rms[i,j,k]  = np.sqrt(np.mean(np.square(a[tmin:,i,j,k])))
    
    return a_rms


def fourier_spectrum(signal, dt, cutoff=0.2):
    import numpy.fft as fft
    spectrum = fft.fft(signal)
    freq = fft.fftfreq(len(spectrum))/dt
    threshold = cutoff * max(abs(spectrum))
    mask = abs(spectrum) > threshold
    
    return freq, abs(spectrum)
