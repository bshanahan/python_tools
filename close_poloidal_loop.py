from boutdata.collect import collect
from boututils.datafile import DataFile
import numpy as np
import matplotlib.pyplot as plt

def close_poloidal_loop(var, fname):
    nt,nx,ny,nz = var.shape
    f = DataFile(fname)
    r = f.read("R")
    z = f.read("Z")
    var_new = np.empty((nt,nx,ny,nz+1))
    r_new = np.empty((nx,ny,nz+1))
    z_new = np.empty((nx,ny,nz+1))
    for k in np.arange(0,nz+1):
        if (k<nz):
            r_new[...,k] = r[...,k]
            z_new[...,k] = z[...,k]
        else:
            r_new[...,k] = r[...,0]
            z_new[...,k] = z[...,0]
        for t in np.arange(0,nt):
            if (k<nz):
                var_new[t,...,k] = var[t,...,k]
            else:
                var_new[t,...,k] = var[t,...,0]

    return r_new,z_new,var_new


def contourf_poloidal_plane(var,fname,var_contour=None,t=-1,y=0,colorbar=False,save=False,save_fname="poloidal_cross_section.png", cbar_label='var', plot_boundaries=False):
    plt.rc('font', family='Serif')
    # plt.grid(alpha=0.5)
    if var_contour is not None:
        r_new, z_new, var_new  = close_poloidal_loop(var,fname)
        r_new, z_new, var_contour_new  = close_poloidal_loop(var_contour,fname)

        plt.contourf(r_new[:,y,:], z_new[:,y,:], var_new[t,:,y,:],100)
        if colorbar:
            cbar = plt.colorbar()
            cbar.set_label(str(cbar_label), fontsize=14)
        plt.contour(r_new[:,y,:], z_new[:,y,:], var_contour_new[t,:,y,:],10)
        plt.xlabel("R [m]", fontsize=14)
        plt.ylabel("Z [m]", fontsize=14)
        plt.tick_params('both',labelsize=14)
        plt.axis("equal")
        plt.tight_layout()
        if save:
            plt.savefig(save_fname, dpi=300)
        plt.show()

    else:
        
        r_new, z_new, var_new  = close_poloidal_loop(var,fname)
        if plot_boundaries:
            plt.plot(r_new[-1,y,:], z_new[-1,y,:], 'k')
            plt.plot(r_new[0,y,:], z_new[0,y,:], 'k')
        
        # for i in [0,8,16,24,-1]:
            # print i
        plt.contourf(r_new[:,y,:], z_new[:,y,:], var_new[t,:,y,:],100)
        if colorbar:
            cbar = plt.colorbar()
            cbar.set_label(str(cbar_label), fontsize=14)
        plt.xlabel("R [m]", fontsize=14)
        plt.ylabel("Z [m]", fontsize=14)
        plt.tick_params('both',labelsize=14)
        plt.gca().axis("equal")
        
        # ax = plt.axes()
        # ax.set_ylim((np.min(z_new),np.max(z_new)))
        # ax.set_xlim((np.min(r_new),np.max(r_new)))
        # ax.set_aspect('equal')
        plt.tight_layout()

        if save:
            plt.savefig(save_fname, dpi=300)

        plt.show()
        
        # plt.savefig('w7x-vacuum-y'+str(i)+'.png',dpi=500)
        plt.clf()
