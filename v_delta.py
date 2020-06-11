from boututils.datafile import DataFile
from boutdata.collect import collect
from boututils.showdata import showdata
import numpy as np
import matplotlib.pyplot as plt

def calc_com_velocity(path='data', plot=False, density_name='n', omega_name='omega_ci'):
    #collect density, time and normalization constants
    n=collect(str(density_name),path=path)
    omega=collect(str(omega_name), path=path)
    t_array=collect("t_array", path=path)/omega
    r_l=collect("rho_s", path=path)
    dx = collect('dx', path=path)[0,0] * r_l
    dz = collect('dz', path=path) * r_l
    nx = n.shape[1]
    nz = n.shape[-1]
    Lx = dx*nx
    Lz = dz*nz


    #subtracting background, normalizing time
    n=n-np.min(n[0,...])
    dt=(t_array[1]-t_array[0])
    v=[]

    x=[]
    z=[]
    
    x_max=nx
    z_max=nz

    #calculating the overall number of particles 
    def N():
        N=[]
        for t in range(0,len(t_array)):
            n_t=0
            for i in range(0, x_max):
                for j in range(0,z_max):
                    n_t+=n[t,i,0,j]
                    N.append(n_t)

        return(N)

    #calculating the centrer of mass
    def getCOM():
        R=[]
        for t in range(0,len(t_array)):
            r=np.array([0.0,0.0])
            for i in range(0,x_max):
                for j in range(0,z_max):
                    r[0] = r[0] + (n[t,i,0,j])*i
                    r[1] = r[1] + (n[t,i,0,j])*j
            r_n=r*r_l*Lx/N[t] #normalisation
            R.append(r_n)
        R = np.array(R)
        print R
        return(R)

    #calculating the radial velocity between two timesteps
    def getv():
        v=[]
        for i in range(0, len(t_array)-1):
            v.append((R[i+1,0]-R[i,0])/dt)
        v.append(v[len(v)-1])
        return(v)


    N=N()
    R=getCOM()
    for i in range(0,len(R)):
        x.append(R[i,0])
        z.append(R[i,1])

    v=np.array(getv()).tolist() #formating reasons
    #peak radial velocity
    v_max=np.amax(v)

    if(plot):
        plt.plot(t_array*1000,v)
        plt.xlabel('t [ms]')

        #plt.plot(x,v,'ro')
        #plt.xlabel('x[m]')
        #plt.ylabel('v [m/s]')

        #plt.plot(t_array/omega*1000, z)
        #plt.xlabel('t[ms]')
        plt.ylabel('x[m]')
        plt.show()

    return v, R
