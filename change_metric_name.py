import numpy as np
from boututils.datafile import DataFile 
import os

def new_to_old(filename):
    f = DataFile(filename)
    
    newfile = DataFile(os.path.splitext(filename)[0]+str(".BOUT_metrics.nc"),create=True)

    name_changes = {"g_yy":"g_22",
                    "gyy":"g22",
                    "gxx":"g11",
                    "gxz":"g13",
                    "gzz":"g33",
                    "g_xx":"g_11",
                    "g_xz":"g_13",
                    "g_zz":"g_33"}

    for key in f.keys():
        name = key
        if name in name_changes:
            name = name_changes[name]
        newfile.write(name, np.asarray(f.read(key)))

        
    f.close()
    newfile.close()

    newfile.list()
        
def change_variable(filename, variable, new_value):
    f = DataFile(filename)
    
    newfile = DataFile(os.path.splitext(filename)[0]+str(variable)+"."+str(new_value),create=True)

    var_changes = {str(variable)}
    

    for key in f.keys():
        name = key
        if name in var_changes:
            name = name_changes[name]
        newfile.write(name, np.asarray(f.read(key)))

        
    f.close()
    newfile.close()

    newfile.list()
