
for t in trange:
    plt.contourf(RR,ZZ,n0*n[int(t),:,0,:],100, vmin=np.min(n0*n[0,:,0,:]), vmax=np.max(n0*n[0,:,0,:]))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel('R [m]',fontsize=21)
    plt.ylabel('Z [m]',fontsize=21)
    plt.tick_params('both',labelsize=14)
    cbar.set_label(r'Density (m$^{-3}$)', fontsize=21)
    plt.tight_layout()
    plt.savefig('n_'+str(int(t)).zfill(3)+'.png',dpi=300)
    plt.close()
