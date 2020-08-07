import sys
sys.path.insert(0, "/home/gregor-pechstein/BOUT-dev/tools/pylib/") 
sys.path.insert(0, '../../tools/pylib/python_tools') 
from boutdata.collect import collect 
from boututils.datafile import DataFile 
import numpy as np 
from boututils import calculus as calc 
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt                                         
 

def Plot_I_sat(path,delta_measured,blob_size_error,twindow,I_CA,t_array,t_range):

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( delta_measured,blob_size_error,'v', label=r'Size error')

    plt.grid(alpha=0.5)
    plt.xlabel(r'Blob size measured [mm]', fontsize=18)
    plt.ylabel(r'Size measurement error [$\%$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/size_error.png', dpi=300)
    plt.show()


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( twindow[:,0]*1e6,I_CA[:,0],'v', label=r'$J_{sat}$ in '+str(np.around(t_array[t_range[0,0]]*1e6))  + ' to '+str(np.around(t_array[t_range[0,1]]*1e6)) +'ms' )
    plt.plot( twindow[:,4]*1e6,I_CA[:,4],'s', label=r'$J_{sat}$  in '+str(np.around(t_array[t_range[4,0]]*1e6))  + ' to '+str(np.around(t_array[t_range[4,1]]*1e6)) +'ms' )
    plt.plot( twindow[:,8]*1e6,I_CA[:,8],'o', label=r'$J_{sat}$  in '+str(np.around(t_array[t_range[8,0]]*1e6))  + ' to '+str(np.around(t_array[t_range[8,1]]*1e6)) +'ms' )
      

    plt.grid(alpha=0.5)
    plt.xlim(-100, 100)
    plt.xlabel(r't [$\mu s$]', fontsize=18)
    plt.ylabel(r'$J_{sat}$ [$mA/mm^2$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/I_sat.png', dpi=300)
    plt.show()

    return


def Plot_pin_distance(path,distance,velocity_error_1Probe,twindow,vr_CA,v_in_t_range):


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( 2*distance*1e3,velocity_error_1Probe,'v', label=r'Velocity error')

    plt.grid(alpha=0.5)
    plt.xlabel(r'distance between pins [mm]', fontsize=18)
    plt.ylabel(r'Velocity error [$\%$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/pin_distance_error.png', dpi=300)
    plt.show()


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( twindow,vr_CA[:,4],'v', label=r'$\Delta s$ = '+ str(np.around(distance[4],decimals=1))+'mm')
    plt.plot( twindow,vr_CA[:,9],'s', label=r'$\Delta s$  = '+ str(np.around(distance[9],decimals=1))+'mm')
    plt.plot( twindow,vr_CA[:,1],'o', label=r'$\Delta s$  = '+ str(np.around(distance[1],decimals=1))+'mm')
    plt.plot( twindow,np.full(twindow.shape,np.max(v_in_t_range)) ,'k--', label=r' Max Real velocity')    

    plt.grid(alpha=0.5)
    plt.xlim(-100, 100)
    plt.xlabel(r't [$\mu s$]', fontsize=18)
    plt.ylabel(r'$\mathrm{v}$ [m/s]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/v_pin_distance.png', dpi=300)
    plt.show()

    return


def Plot_secondProbe(path, distSecondProbe_z, velocity_error_1Probe, velocity_error_2Probe, velocity_error_direct, v_pol_error):


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( distSecondProbe_z*1e3,np.full(distSecondProbe_z.shape[0],velocity_error_1Probe),'v', label=r'Velocity error 1. Probe')
    plt.plot( distSecondProbe_z*1e3,velocity_error_2Probe,'o', label=r'Velocity error 2. Probe')
    plt.plot( distSecondProbe_z*1e3,velocity_error_direct,'s', label=r'Velocity Error time difference')

    plt.grid(alpha=0.5)
    plt.xlabel(r'vertical offset [mm]', fontsize=18)
    plt.ylabel(r'Velocity error [$\%$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper right', framealpha=0, fontsize=12)
    plt.savefig(path+'/2Probs_error_vr.png', dpi=300)
    plt.show()

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( distSecondProbe_z*1e3,v_pol_error,'s', label=r'Poloidal velocity error')

    plt.grid(alpha=0.5)
    plt.xlabel(r'vertical offset [mm]', fontsize=18)
    plt.ylabel(r'Velocity error [$\%$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper right', framealpha=0, fontsize=12)
    plt.savefig(path+'/2Probs_error_v_pol.png', dpi=300)
    plt.show()

    return


def Plot_Inclination(path,inclinationAngle, twindow,vr_CA,v_in_t_range, I_CA,velocity_error_1Probe):

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( twindow,vr_CA[:,6],'v', label=r'Inclination = '+ str(np.around(inclinationAngle[6], decimals=2))+'rad')
    plt.plot( twindow,vr_CA[:,8],'s', label=r'Inclination = '+ str(np.around(inclinationAngle[8], decimals=2))+'rad')
    plt.plot( twindow,vr_CA[:,10],'o', label=r'Inclination = '+ str(np.around(inclinationAngle[10], decimals=2))+'rad')
    plt.plot( twindow,np.full(twindow.shape,np.max(v_in_t_range)) ,'k--', label=r' Max Real velocity')

    plt.grid(alpha=0.5)
    plt.xlim(-100, 100)
    plt.xlabel(r't [$\mu s$]', fontsize=18)
    plt.ylabel(r'$\mathrm{v}$ [m/s]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/v_inclination2.png', dpi=300)
    plt.show()


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( twindow,I_CA,'v', label=r'$J_{sat}$')

    plt.grid(alpha=0.5)
    plt.xlim(-100, 100)
    plt.xlabel(r't [$\mu s$]', fontsize=18)
    plt.ylabel(r'$J_{sat}$ [$mA/mm^2$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/v_inclination_I_CA.png', dpi=300)
    plt.show()



    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(inclinationAngle,velocity_error_1Probe[:],'v', label=r'Velocity Error')
    x_tick = np.arange(-np.pi/8, np.pi/8+0.01,np.pi/16) 
    x_label = [r"$-\frac{\pi}{8}$", r"$-\frac{\pi}{16}$", r"$0$", r"$+\frac{\pi}{16}$",   r"$+\frac{\pi}{8}$"]
    ax.set_xticks(x_tick)
    ax.set_xticklabels(x_label, fontsize=20)
    plt.grid(alpha=0.5)     
    plt.xlabel(r'Inclination Angle ', fontsize=18)
    plt.ylabel(r'Velocity error [$\%$]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/inclination_error2.png', dpi=300)
    plt.show()
