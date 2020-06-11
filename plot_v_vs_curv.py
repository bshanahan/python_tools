from boutdata.collect import collect
import numpy as np
from calc_velocity import calc_com_velocity
import matplotlib.pyplot as plt


const_dirs = ['slightly_neg/const_curv', 'ballooning/const_curv', 'wobbly/non_bous/const_curv']
weak_dirs = [ 'slightly_neg', 'ballooning/less_curv/non-bous', 'wobbly/non_bous']
strong_dirs = ['slightly_neg/2.5curv','ballooning/2.5curv', 'wobbly/2.5curv' ]

curvatures = [0,0.33333,0.666667, 1.0]
velocities = np.zeros((4,3))
indices = [0,1,2,3]
for ind, c_dir, w_dir, s_dir in zip(indices,const_dirs,weak_dirs,strong_dirs):
    
    v_c, pos_fit_c, pos_c, R_c, Z_c = calc_com_velocity(path=c_dir)
    v_w, pos_fit_w, pos_w, R_w, Z_w = calc_com_velocity(path=w_dir)
    v_s, pos_fit_s, pos_s, R_s, Z_s = calc_com_velocity(path=s_dir)


    # curvatures[ind+1] = np.mean(collect('bxcvz', path=c_dir),axis=1)
    velocities[ind+1,:] = [np.max(v_c),np.max(v_w),np.max(v_s)]

plt.plot(curvatures,velocities[:,0],'ko', label='a=0')
plt.plot(curvatures,velocities[:,1],'bs', label='a=0.67')
plt.plot(curvatures,velocities[:,2],'r^', label='a=1.67')

plt.legend()
plt.show()
    
    
    
