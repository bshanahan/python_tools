import numpy as np
import matplotlib.pyplot as plt

def line_plot(independent, dependent, hline=None, vline=None, labels=None, fname=None, xlabel=' ', ylabel=' ', vline_label=None, hline_label=None, AR=(8,4.5)):
    plt.rc('font', family='Serif')
    plt.figure(figsize=AR)

    linewidth = 2.5

    n_dep = len(dependent)
    if labels==None:
        labels = np.full(n_dep,None)
    
    for k in np.arange(0,n_dep):
        plt.plot(independent, dependent[k], linewidth=linewidth, label=labels[k])

    ## plot any straight lines    
    if vline is not None:
        for i in np.arange(0,len(vline)):
            plt.axvline(vline[i], color='black', linestyle='dashed', linewidth=linewidth-0.5, label=vline_label[i])

    if hline is not None:
        for i in np.arange(0,len(hline)):
            plt.axhline(hline[i], color='black', linestyle='dashed', linewidth=linewidth-0.5, label=hline_label[i])


    plt.xlabel(str(xlabel), fontsize=18)
    plt.ylabel(str(ylabel), fontsize=18)
    if labels is not None:
        plt.legend(loc='best')
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.grid(alpha=0.5)
    if fname is not None:
        plt.savefig(str(fname)+'.png', dpi=300)
    plt.show()


def threed_surface_plot(R,Z,phi,contours, xslice=None, cmap=None, fname=None):

    
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import numpy as np
    
    if xslice is None:
        xslice = int(R.shape[0]/2.)

    if cmap is None:
        cmap = cm.viridis
    elif cmap == 'divergent':
        cmap = cm.seismic
    elif cmap == 'plasma':
        cmap = cm.plasma
    elif cmap == 'viridis':
        cmap = cm.viridis
    elif cmap == 'jet':
        cmap = cm.jet

    Phi = np.zeros(R.shape)
    for x in np.arange(R.shape[0]):
        for z in np.arange(R.shape[-1]):
            Phi[x,:,z] = phi

    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)
    
    fig = plt.figure(figsize=(6, 4)) #
    ls = LightSource(270, 45)

    ax = fig.add_subplot(111, projection='3d')
    contour_colors = contours[xslice,...]#/np.max(contours[xslice,...])
    rgb = ls.shade(contours[xslice,:,:], cmap=cmap, blend_mode='soft')
    surf = ax.plot_surface(X[xslice,...], Y[xslice,...], Z[xslice,...], rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=True, shade=False)#, vmin=-1, vmax=1)#, vmin=np.min(contours[xslice,...]), vmax=np.max(contours[xslice,...]))

    print(np.min(rgb), np.min(contours))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(contour_colors)#contour_colors*contours[xslice,...])
    fig.colorbar(m, shrink=0.5)
    set_axes_equal(ax)
    ax.set_axis_off()

    if fname is not None:
        plt.savefig(fname, dpi=500)
        
    plt.show()

def threed_surface_plot_xr(ds, key, xslice=None, cmap=None, fname="3d_plot.png", style="align", interpolate="quadratic"):
    """
    style : str
        "default" : align to the coordinat system
        "align" : align locally to the magnetic flux
    """
    
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    import numpy as np
    
    import xarray as xr
    
    if xslice is None:
        xslice = ds.R.shape[0]//2
    
    ds = ds.isel(x=xslice)

    if cmap is None:
        if np.min(ds[key]) * np.max(ds[key]) < 0:
            cmap = cm.seismic
        else:
            cmap = cm.viridis
    elif cmap == "divergent":
        cmap = cm.seismic
    elif cmap == "plasma":
        cmap = cm.plasma
    elif cmap == "viridis":
        cmap = cm.viridis

    fig = plt.figure(figsize=(6, 4))

    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(270, 45)
    contours = ds[key]
    assert contours.dims == ds.R.dims
    rgb = ls.shade(contours.values, cmap=cmap, blend_mode='soft')
    if style == "align":
        dy = ds.y[1] - ds.y[0]
        ym = ds.y - dy
        yp = ds.y + dy
        dsc = xr.Dataset()
        dsc["X"] = ds.R * np.cos(ds.y)
        dsc["Y"] = ds.R * np.sin(ds.y)
        dsc["Zv"] = ds.Z
        dsb = xr.Dataset()
        dsb["X"] = ds["backward_R"] * np.cos(ym)
        dsb["Y"] = ds["backward_R"] * np.sin(ym)
        dsb["Zv"] = ds["backward_Z"]
        dsf = xr.Dataset()
        dsf["X"] = ds["forward_R"] * np.cos(yp)
        dsf["Y"] = ds["forward_R"] * np.sin(yp)
        dsf["Zv"] = ds["forward_Z"]
        if interpolate == "quadratic":
            dsp = 0.75 * dsc + 3/8 * dsf - 1/8 * dsb
            dsm = 0.75 * dsc + 3/8 * dsb - 1/8 * dsf
        else:
            dsp = 0.5 * dsc + .5 * dsf
            dsm = 0.5 * dsc + .5 * dsb
        dsn = xr.combine_nested([dsm, dsp], ["y_bounds"])
        print(dsn)
        #print(dsn, dsf)
        dsn[key] = ds[key]
        assert ds.R.dims[0] == 'y', f"Expected element 1 of `{ds.R.dims}` to be `y`"
        for yi in range(len(dsn.y)):
            dsi = dsn.isel(y=yi)
            rgbi = rgb[yi][None, :, :]
            rgbi = np.vstack([rgbi, rgbi])
            #print(dsi.X, rgbi.shape)
            assert dsi.X.shape == rgbi.shape[:2]
            ax.plot_surface(dsi.X, dsi.Y, dsi.Zv, rstride=1, cstride=1, facecolors=rgbi,
                            linewidth=0, antialiased=True, shade=False)
        
    else:
        assert style == "default", f"Unexpexted plotting style `{style}`"
        X = ds.R * np.cos(ds.y)
        Y = ds.R * np.sin(ds.y)
        ax.plot_surface(X, Y, ds.Z, rstride=1, cstride=1, facecolors=rgb,linewidth=0, antialiased=False, shade=False)
    

    ax.set_axis_off()
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(contours)
    fig.colorbar(m, shrink=0.5)#contours[xslice,...], shrink=0.5)
    set_axes_equal(ax)
    if fname is not None:
        plt.savefig(fname, dpi=1000)
    plt.show()

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    This routine was written by karlo and Trenton McKinney on stackoverflow: 
    https://stackoverflow.com/a/31364297
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def threed_surface_plot(R,Z,phi,contours, cmap=None, fname=None, cbarlabel=None):

    
    from matplotlib import cbook
    from matplotlib import cm,font_manager
    from matplotlib.colors import LightSource
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import numpy as np

    if cmap is None:
        cmap = cm.viridis
    elif cmap == 'divergent':
        cmap = cm.seismic
    elif cmap == 'plasma':
        cmap = cm.plasma
    elif cmap == 'viridis':
        cmap = cm.viridis
    elif cmap == 'jet':
        cmap = cm.jet

    Phi = np.zeros(R.shape)
    for z in np.arange(R.shape[1]):
        Phi[:,z] = phi

    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)
    
    fig = plt.figure(figsize=(6, 4)) #
    ls = LightSource(270, 45)

    ax = fig.add_subplot(111, projection='3d')
    contour_colors = contours[...]
    rgb = ls.shade(contours[:,:], cmap=cmap, blend_mode='soft')
    surf = ax.plot_surface(X[...], Y[...], Z[...], rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=True, shade=False)#, vmin=-1, vmax=1)#, vmin=np.min(contours[xslice,...]), vmax=np.max(contours[xslice,...]))

    print(np.min(rgb), np.min(contours))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(contour_colors)#contour_colors*contours[xslice,...])
    cbar = fig.colorbar(m, shrink=0.5)
    cbar.set_label(label=r'$\sigma_n \quad \mathrm{[m^{-3}]}$')
    cbax = cbar.ax
    text = cbax.yaxis.label
    font = font_manager.FontProperties(family='Serif', size=16)
    text.set_font_properties(font)
    cbax.tick_params(labelsize=14) 
    set_axes_equal(ax)
    ax.set_axis_off()
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=500)

        
    plt.show()

def subcontours():
    from matplotlib import ticker
    fig1, axs = plt.subplots(nrows=1,ncols=2, sharey='row',figsize=(8, 4.5))
    (ax1,ax2) = axs
    levels = np.linspace(-200,200, 100)
    cs1 = ax1.contourf(RR,ZZ,jpol_pi[15,:,0,:], norm=colors.CenteredNorm(), levels=levels, extend='both')
    ax1.contour(RR,ZZ,n_pi[15,:,0,:], 2, colors='black', alpha=0.5)
    ax1.set_ylabel('Z [cm]', fontsize=18)
    ax1.set_xlabel('R [cm]', fontsize=18)
    ax1.axis('equal')
    ax1.tick_params('both', labelsize=14)
    ax1.set_title('Equation (1.8)', font='Serif', fontsize=14)
    cs2 = ax2.contourf(RR,ZZ,jpol[15,:,0,:], norm=colors.CenteredNorm(), levels=levels, extend='both')
    ax2.contour(RR,ZZ,n[15,:,0,:], 2, colors='black', alpha=0.5)
    ax2.set_xlabel('R [cm]', fontsize=18)
    ax2.axis('equal')
    ax2.tick_params('both', labelsize=14)
    ax2.set_title('Equation (1.9)', font='Serif', fontsize=14)

    plt.tight_layout();
    cbar = plt.colorbar(cs1,ax=axs, extend='both')
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    # cbar.ax.locator_params(nbins=8)
    cbar.set_label(r'$\mathrm{\nabla \cdot J_{pol} \quad [A/m]}$', fontsize=14)
    # ax = cbar.ax
    # text = ax.yaxis.label
    # font = matplotlib.font_manager.FontProperties(family='serif', size=16)
    # text.set_font_properties(font)
    plt.savefig('jpol_comp.png',dpi=500)
    plt.show()
def make_suplot():
    ###subplot
    fig,axs = plt.subplots(1,2, figsize=(8,4.5),sharey=True)
    
    y=0
    axs[0].plot(linesx[5].R, linesx[5].Z, '-', color='#00a09c')
    axs[0].plot(linesx[7].R, linesx[7].Z, '-', color='#00a09c')

    axs[0].plot(Rc[2,y,:], Zc[2,y,:], '#005555');
    axs[0].plot(Rc[-2,y,:], Zc[-2,y,:], '#005555');

    axs[0].plot(Rc[2::64,y,::32], Zc[2::64,y,::32], 'k-', alpha=0.2)
    axs[0].plot(Rc[54::52,y,::32].T, Zc[54::54,y,::32].T, 'k-', alpha=0.2)

    #    axs[0].plot(r[:8,y,:], z[:8,y,:], 'k.', markersize=0.5)
    axs[0].axis('equal')
    #    axs[0].grid(alpha = 0.5)
    axs[0].set_xlabel('R [m]', fontsize=18)
    axs[0].set_ylabel('Z [m]', fontsize=18)
    axs[0].tick_params('both', labelsize=14)


    y = 18

    # axs[1].plot(Rc[2,y,:], Zc[2,y,:], 'b-');
    # axs[1].plot(Rc[-2,y,:], Zc[-2,y,:], 'r-');
    # axs[1].plot(r[:8,y,:], z[:8,y,:], 'k.', markersize=0.5)
    axs[1].plot(linesxb[5].R, linesxb[5].Z, '-', color='#00a09c')
    axs[1].plot(linesxb[7].R, linesxb[7].Z, '-', color='#00a09c')

    axs[1].plot(Rc[2,y,:], Zc[2,y,:], '#005555');
    axs[1].plot(Rc[-2,y,:], Zc[-2,y,:], '#005555');

    axs[1].plot(Rc[2::64,y,::32], Zc[2::64,y,::32], 'k-', alpha=0.2)
    axs[1].plot(Rc[54::52,y,::32].T, Zc[54::54,y,::32].T, 'k-', alpha=0.2)
    axs[1].axis('equal')
    #    axs[1].grid(alpha = 0.5)
    axs[1].set_xlabel('R [m]', fontsize=18)
    axs[1].tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.savefig('W7-X_grid.png',dpi=500)
    plt.show()


def mlab_surf(r,phi,z,surf, rind, show_ghosts=False, ng=4, cmap="viridis", fname=None):
    from mayavi import mlab
    
    x=np.zeros(r.shape)
    y=np.zeros(r.shape)
    dphi = phi[1]-phi[0]
    for j in np.arange(0,phi.shape[0]):
        x[:,j,:] = r[:,j,:]#*np.cos(phi[j])
        y[:,j,:] = np.ones(r[:,j,:].shape)*phi[j]#*np.sin(phi[j])
    
    mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0))
    mlab.mesh(x[rind,...],y[rind,...], z[rind,...], scalars= surf[rind,:,:], colormap=cmap)
    phiend = phi[-1] + dphi
    # mlab.scalarbar(nb_labels =7, orientation='vertical', title='$\sigma_n \quad \mathrm{[m^{-3}]}$')
    k=0
    mlab.plot3d(x[rind,:,k],y[rind,:,k], z[rind,:,k], -y[rind,:,k], colormap='Purples', tube_radius=0.01)
    if (show_ghosts):
        for ig in np.arange(0,ng):
            phi += phiend
            xg = np.copy(x)
            yg = np.copy(y)
            zg = np.copy(z)
            for jg in np.arange(0,phi.shape[0]):
                xg[:,jg,:] = r[:,jg,:]*np.cos(phi[jg])
                yg[:,jg,:] = r[:,jg,:]*np.sin(phi[jg])

            mlab.mesh(xg[rind,...],yg[rind,...], zg[rind,...], scalars= surf[rind,:,:], colormap=cmap)
    if fname is not None:
        mlab.savefig(fname, size=(512,512))
    mlab.show()
    


def mlab_plot3d(r,phi,z,s, rind=5,zind=0, show_ghosts=False, ng=4, cmap="viridis"):
    x=np.zeros(r.shape)
    y=np.zeros(r.shape)
    dphi = phi[1]-phi[0]
#    vmaxabs = np.max(np.abs(s[rind,:,zind]))
    for j in np.arange(0,phi.shape[0]):
        x[:,j,:] = r[:,j,:]*np.cos(phi[j])
        y[:,j,:] = r[:,j,:]*np.sin(phi[j])

    mlab.figure(bgcolor=(0,0,0))
    for k in zind:
        mlab.plot3d(x[rind,:,k],y[rind,:,k], z[rind,:,k], s[rind,:,k], colormap=cmap, vmax=vmaxabs, vmin=-vmaxabs)

    phiend = phi[-1] + dphi
    mlab.scalarbar(nb_labels =7, orientation='vertical', title='$\sigma_n \quad \mathrm{[m^{-3}]}$')

    if (show_ghosts):
        for ig in np.arange(0,ng):
            phi += phiend
            xg = np.copy(x)
            yg = np.copy(y)
            zg = np.copy(z)
            for jg in np.arange(0,phi.shape[0]):
                xg[:,jg,:] = r[:,jg,:]*np.cos(phi[jg])
                yg[:,jg,:] = r[:,jg,:]*np.sin(phi[jg])

            for k in zind:
                mlab.plot3d(xg[rind,:,k],yg[rind,:,k], zg[rind,:,k], s[rind,:,k], colormap=cmap, vmax=vmaxabs, vmin=-vmaxabs)

    mlab.show()


import numpy
from mayavi.mlab import *

def test_plot3d():
    """Generates a pretty set of lines."""
    n_mer, n_long = 2, 1
    dphi = np.pi / 1000.0
    phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    R = 2
    r = 0.48
    
    x = np.cos(mu) * (R + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (R + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    R = 2
    r = 0.48

    
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, 2 * np.pi, 20)
    torus = np.zeros((3,20,20))

    for i in range(0,20):
        for j in range(0,20):
            torus[0][i][j] = (R + r * np.cos(phi[j])) * np.cos(theta[i])
            torus[1][i][j] = (R + r * np.cos(phi[j])) * np.sin(theta[i])
            torus[2][i][j] = r * np.sin(phi[j])


    mlab.mesh(torus[0], torus[1], torus[2], representation='wireframe')
    mlab.plot3d(x, y, z, np.ones(z.shape), tube_radius=0.025, colormap='Greys')
    mlab.show()


def plot_flows():
    import make_colorbar
    #plt.contourf(R[4:-4,y,zind[0]:zind[-1]], Z[4:-4,y,zind[0]:zind[-1]],(vz)[4:-4,y,zind[0]:zind[-1]], 100, norm=colors.CenteredNorm());
    plt.rc('font', family='Serif')    
    plt.quiver(R[6:-6:16,y,zind[0]:zind[-1]:4],Z[6:-6:16,y,zind[0]:zind[-1]:4], vx[6:-6:16,0,zind[0]:zind[-1]:4],vz[6:-6:16,0,zind[0]:zind[-1]:4], np.sqrt(vx**2+vz**2)[6:-6:16,0,zind[0]:zind[-1]:4], width=5e-3, linewidth=2, angles='uv', scale=2.5e3);
#    plt.contourf(R[6:-6,y,:], Z[6:-6,y,:],vz[6:-6,0,:],100, norm=colors.CenteredNorm())
    plt.axis('equal');
    make_colorbar(label=r'$\mathregular{v \quad [m/s]}$')

    plt.plot(R[-2,0,:], Z[-2,0,:], 'k-', linewidth=2)
    ## Make a colorbar
    # import matplotlib.pylab as pyl
    # from numpy import arange
    # import matplotlib
    
    # cb = pyl.colorbar(label=r'$\mathrm{v_{pol} \quad [km/s]}$')
    # ax = cb.ax
    # text = ax.yaxis.label
    # font = matplotlib.font_manager.FontProperties(family='serif', weight='bold', size=16)
    # text.set_font_properties(font)

    # cb.ax.tick_params(labelsize=12)


    ###
    # cb = plt.colorbar(label='$\mathregular{v_{pol}} \quad \mathregular{[km/s]}$');#$\mathrm{v_{pol} \quad [km/s]}$');
    # label = cb.ax.yaxis.label
    # font = matplotlib.font_manager.FontProperties(family='serif', size=14)
    # label.set_font_properties(font)
    for i in np.arange(33,44,3):
        plt.plot(lines[i].R, lines[i].Z, '-', color='#005555', alpha=0.5)
    for i in np.arange(3,14,3):
        plt.plot(lines[i].R, lines[i].Z, '-', color='#005555', alpha=0.5)
    for i in np.arange(18,29,3):
        plt.plot(lines[i].R, lines[i].Z, '-', color='#005555', alpha=0.5)
    for i in np.arange(48,59,3):
        plt.plot(lines[i].R, lines[i].Z, '-', color='#005555', alpha=0.5)
    for i in np.arange(63,74,3):
        plt.plot(lines[i].R, lines[i].Z, '-', color='#005555', alpha=0.5)

    plt.plot(linesx[5].R, linesx[5].Z, '-', color='#005555', alpha=0.5, linewidth=2)
    plt.plot(linesx[7].R, linesx[7].Z, '-', color='#005555', alpha=0.5, linewidth=2)

    
    #plt.plot(r[::2,y,:], z[::2,y,:], '.', color='grey', alpha=0.5,markersize=1)
    #plt.contour(R[14:-4,y,zind[0]:zind[-1]], Z[14:-4,y,zind[0]:zind[-1]], cl[14:-4,zind[0]:zind[-1]], 5, colors='k');
    
    plt.xlabel('R [m]', fontsize=16); plt.ylabel('Z [m]', fontsize=16)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.xlim(5.3,5.7)
    plt.ylim(0.7,1.05)
#    plt.savefig('flows_cl_flux_surfaces_lowi.png', dpi=400, transparent=True)


def plot2contours(y=[0,18]):
    plt.rc('font', family='Serif')
    fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 2.32]},figsize=(8,4.5))
    #fig.suptitle('A tale of 2 subplots')
    
    cf1 = ax1.contourf(R[2:-2,y[0],:], Z[2:-2,y[0],:], cl[:,y[0],:], 100)
    ax1.set_ylabel('Z [m]', fontsize=16)
    ax1.set_xlabel('R [m]', fontsize=16)
    ax1.axis('equal')
    ax1.tick_params('both', labelsize=14)
    ax1.set_ylim([-1,1])

    if plot_islands:
        for i in np.arange(33,44,3):
            ax1.plot(linesb[i].R, linesb[i].Z, '-', color='grey')
        for i in np.arange(3,14,3):
            ax1.plot(linesb[i].R, linesb[i].Z, '-', color='grey')
        for i in np.arange(18,29,3):
            ax1.plot(linesb[i].R, linesb[i].Z, '-', color='grey')
        for i in np.arange(48,59,3):
            ax1.plot(linesb[i].R, linesb[i].Z, '-', color='grey')
        for i in np.arange(63,74,3):
            ax1.plot(linesb[i].R, linesb[i].Z, '-', color='grey')

        
    cf2 = ax2.contourf(R[2:-2,y[1],:], Z[2:-2,y[1],:], cl[:,y[1],:], 100)
    ax2.set_xlabel('R [m]', fontsize=16)
    ax2.axis('equal')
    ax2.tick_params('both', labelsize=14)
    ax2.set_ylim([-1,1])
    
    if plot_islands:
        for i in np.arange(33,44,3):
            ax1.plot(linest[i].R, linest[i].Z, '-', color='grey')
        for i in np.arange(3,14,3):
            ax1.plot(linest[i].R, linest[i].Z, '-', color='grey')
        for i in np.arange(18,29,3):
            ax1.plot(linest[i].R, linest[i].Z, '-', color='grey')
        for i in np.arange(48,59,3):
            ax1.plot(linest[i].R, linest[i].Z, '-', color='grey')
        for i in np.arange(63,74,3):
            ax1.plot(linest[i].R, linest[i].Z, '-', color='grey')
        
#    fig.colorbar(cf1, label=r'$L_\parallel$', fontsize=16)
    cb = fig.colorbar(cf1);
    cb.set_label(r'$\mathrm{L_\parallel \quad [m]}$',family='Times New Roman', size=14)

    cb.set_label(r'$L_\parallel \quad \mathrm{[m]}$',family='DejaVu Serif', size=14)
    plt.tight_layout()
    plt.savefig('cl_bean_tri.png', dpi=400, transparent=True)
    plt.show()


def plot2axes():
    linewidth = 2.5

    fig, ax1 = plt.subplots(figsize=(8,4.5))
    plt.rc('font', family='DejaVu Serif')

    ax1.set_xlabel('R [m]', fontsize=18, fontname='DejaVu Serif')
    ax1.set_ylabel(r'$\left<n\right> \quad [\mathrm{m^{-3}}]$', color='k', alpha = 0.7, fontsize=18,math_fontfamily='dejavuserif' )
    ax1.plot(R[8:-2,29,int(np.mean(zind))], n[-1,8:-2,29,int(np.mean(zind))], color='black', linewidth=linewidth, alpha=0.4)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel(r'$\left<\phi\right> \quad \mathrm{[V]}$', color='#006c66', fontsize=18 , math_fontfamily='dejavuserif')  # we already handled the x-label with ax1
    #ax2.plot(R[8:-2,29,int(np.mean(zind))], phimean[8:-2,29,int(np.mean(zind))], color='#006c66', linewidth=linewidth )
    ax2.plot(Rhi[32:-2,29,int(np.mean(zhiind))], phi_i_mean[32:-2,29,int(np.mean(zhiind))], color='#006c66', linewidth=linewidth );
    ax2.tick_params('both', labelsize=14)
    ax1.tick_params('both', labelsize=14)
    #plt.plot(R[8:-2,29,int(np.mean(zind))], 1e4*cl[8:-2,int(np.mean(zind))], color='k', linewidth=linewidth)



    ax2.axvspan(R[8,29,int(np.mean(zind))], R[37,29,int(np.mean(zind))], alpha=0.2, color='#006c66')
#    ax1.grid(alpha=0.3)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('nvsphi_profile.png', dpi=400, transparent=True)
    plt.show()


def plot_contourf():
    cf = plt.contourf(R[4:-4,y,zind[0]:zind[-1]], Z[4:-4,y,zind[0]:zind[-1]],(cl)[2:-2,y,zind[0]:zind[-1]], 100);
    plt.xlabel('R [m]', fontsize=18, fontname='DejaVu Serif')
    plt.ylabel('Z [m]', fontsize=18, fontname='DejaVu Serif')
    plt.tick_params('both', labelsize=14)
    plt.axis('equal');

    cb = plt.colorbar(cf);
    cb.set_label(r'$\mathrm{L_\parallel \quad [m]}$',family='DejaVu Serif', size=14)

    for i in np.arange(33,44,4):
        plt.plot(lines[i].R, lines[i].Z, 'w-', alpha=0.7, linewidth=2.5)

    plt.plot(linesx[5].R, linesx[5].Z, 'w-', alpha=0.7, linewidth=2.5)
    plt.plot(linesx[7].R, linesx[7].Z, 'w-', alpha=0.7, linewidth=2.5)
    plt.xlim(5.98,6.125)
    plt.ylim(-0.35,-0.1)

    plt.tight_layout()
    plt.savefig('cl_closeup_diagnostics.png', dpi=400, transparent=True)
    plt.show()

    
    plt.rc('font', family='Serif')
    plt.figure(figsize=(5,4.5))

    #fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 2.32]},figsize=(8,4.5))
    #fig.suptitle('A tale of 2 subplots')
    
    cf1 = plt.contourf(R[6:-2,y[0],:], Z[6:-2,y[0],:], cl[4:,y[0],:], 100)
    plt.ylabel('Z [m]', fontsize=16)
    plt.xlabel('R [m]', fontsize=16)
    plt.axis('equal')
    plt.tick_params('both', labelsize=14)
    cb = plt.colorbar(cf1);
    cb.set_label(r'$\mathrm{L_\parallel \quad [m]}$',family='Times New Roman', size=14)

    if plot_islands:
        for i in np.arange(33,44,4):
            plt.plot(lines[i].R, lines[i].Z, 'w-', alpha=0.7)
        for i in np.arange(3,14,4):
            plt.plot(lines[i].R, lines[i].Z, 'w-', alpha=0.7)
        for i in np.arange(18,29,4):
            plt.plot(lines[i].R, lines[i].Z, 'w-', alpha=0.7)
        for i in np.arange(48,59,4):
            plt.plot(lines[i].R, lines[i].Z, 'w-', alpha=0.7)
        for i in np.arange(63,74,4):
            plt.plot(lines[i].R, lines[i].Z, 'w-', alpha=0.7)
        plt.plot(linesx[5].R, linesx[5].Z, 'w-', alpha=0.7)
        plt.plot(linesx[7].R, linesx[7].Z, 'w-', alpha=0.7)

    plt.tight_layout()
    plt.savefig('cl_GPI_plane_islands.png', dpi=400, transparent=True)
    plt.show()


def make_colorbar(label='label', fontsize=16, labelsize=12):
    import matplotlib.pylab as pyl
    from numpy import arange
    import matplotlib
    
    cb = plt.colorbar(label=label)
    ax = cb.ax
    text = ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=fontsize)
    text.set_font_properties(font)

    cb.ax.tick_params(labelsize=labelsize)
    
def plotly_example():
    import numpy as np
    import plotly.graph_objects as go
    
    # Generate nicely looking random 3D-field
    np.random.seed(0)
    l = 30
    X, Y, Z = np.mgrid[:l, :l, :l]
    vol = np.zeros((l, l, l))
    pts = (l * np.random.rand(3, 15)).astype(np.int)
    vol[tuple(indices for indices in pts)] = 1
    from scipy import ndimage
    vol = ndimage.gaussian_filter(vol, 4)
    vol /= vol.max()

    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol.flatten(),
        isomin=0.2,
        isomax=0.7,
        opacity=0.1,
        surface_count=25,
    ))
    fig.update_layout(scene_xaxis_showticklabels=False,
                      scene_yaxis_showticklabels=False,
                      scene_zaxis_showticklabels=False)
    fig.show()
