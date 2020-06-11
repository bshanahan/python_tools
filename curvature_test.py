    def curvature(self):
        """
        Calculate and return curvature
        """
        
        dx = 1.0 / float(self.nx-1)      # x from 0 to 1 
        dz = 2.*np.pi / float(self.nz)   # z from 0 to 2pi
        
        # Get arrays of indices
        xind, zind = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing='ij')
        
        dRdx, dZdx = self.getCoordinate(xind, zind, dx=1)
        dRdx /= dx
        dZdx /= dx        
        dRdz, dZdz = self.getCoordinate(xind, zind, dz=1)
        dRdz /= dz
        dZdz /= dz

        dl = np.sqrt(dx**2 + dz**2)
        
        Kdx = dRdx + dZdx
        Kdz = dRdz + dZdz
        K = Kdx+Kdz



        
    # Calculate curvature
    R = maps["R"]
    Z = maps["Z"]
    forward_R = maps["forward_R"]
    backward_R = maps["backward_R"]
    forward_Z = maps["forward_Z"]
    backward_Z = maps["backward_Z"]

    gyy = metric["gyy"]
    g_yy = metric["g_yy"]
    g_yy_yup = forward_R**2
    g_yy_ydown = backward_R**2
    gyy_yup = 1/g_yy_yup
    gyy_ydown = 1/g_yy_ydown

    K = np.zeros(R.shape)
    for j in range(grid.numberOfPoloidalGrids()):
        pol_forward, y_forward = grid.getPoloidalGrid(j+1)
        pol_back, y_back = grid.getPoloidalGrid(j-1)
        pol_grid,ypos = grid.getPoloidalGrid(yindex)

        y_spacing = (y_forward-y_back)/2.
        y_spacing_forward = abs(y_forward-ypos)
        y_spacing_backward = abs(ypos-y_back)

        dRdx = pol_grid.metric()["dRdx"] # R-hat
        dRdz = pol_grid.metric()["dRdz"] # R-hat
        dZdx = pol_grid.metric()["dZdx"] # Z-hat
        dZdz = pol_grid.metric()["dZdz"] # Z-hat
        # dr = np.sqrt(dRdx**2 + dRdz**2)
        # dRdx /= dr
        # dRdz /= dr
        
        dRdy = (forward_R[:,(j+1)% grid.numberOfPoloidalGrids(),:] - backward_R[:,j-1,:] )/ y_spacing # R-hat
        dZdy = (forward_Z[:,(j+1)% grid.numberOfPoloidalGrids(),:] - backward_Z[:,j-1,:] )/ y_spacing # Z-hat

        
        dRdy_forward = (forward_R[:,(j+1)% grid.numberOfPoloidalGrids(),:] - R[:,j,:])/y_spacing_forward
        dRdy_back = (- backward_R[:,j-1,:] + R[:,j,:])/y_spacing_backward
        dZdy_forward = (forward_Z[:,(j+1)% grid.numberOfPoloidalGrids(),:] - Z[:,j,:])/y_spacing_forward
        dZdy_back = (- backward_Z[:,j-1,:] + Z[:,j,:])/y_spacing_backward

        dydy = R[:,j,:]

        dl_forward = np.sqrt(dRdy_forward**2 + dZdy_forward**2 + dydy**2)
        dl_back = np.sqrt(dRdy_back**2 + dZdy_back**2 + dydy**2)

        # dZ = dZdy_forward/dl_forward - dZdy_back/dl_back
        dR = dRdy_forward/dl_forward - dRdy_back/dl_back
        # dY = dydy/dl_forward - dydy/dl_back
        dZ = np.sqrt(dZdx**2 + dZdz**2)
        dR_length = np.sqrt(dRdx**2 + dRdz**2)
        dY = dydy/dl_forward - dydy/dl_back
        
        unit_length = np.sqrt(dZ**2 + dR**2 + dY**2)

        dl = np.sqrt(dRdy**2 + dZdy**2)
        # dRdy /= dl
        # dZdy /= dl
        
        dR2dy2 = forward_R[:,(j+1)% grid.numberOfPoloidalGrids(),:] - 2*R[:,j,:] + backward_R[:,j-1,:] / ((y_spacing)**2)
        dZ2dy2 = forward_Z[:,(j+1)% grid.numberOfPoloidalGrids(),:] - 2*Z[:,j,:] + backward_Z[:,j-1,:] / ((y_spacing)**2)
        dl2 = np.sqrt(dR2dy2**2 + dZ2dy2**2)
        # dR2dy2 /= dl2**2
        # dZ2dy2 /= dl2**2

        dg_yydy = g_yy_yup[:,(j+1)% grid.numberOfPoloidalGrids(),:] - g_yy_ydown[:,j-1,:] / (y_spacing)
        dgyydy = gyy_yup[:,(j+1)% grid.numberOfPoloidalGrids(),:] - gyy_ydown[:,j-1,:] / (y_spacing)

        K_R = gyy[:,j,:] * (gyy[:,j,:] * dR2dy2 + dRdy * dgyydy) - (dRdy/np.sqrt(g_yy[:,j,:]))*(dRdx+dRdy+dRdz)
        K_Z = gyy[:,j,:] * (gyy[:,j,:] * dZ2dy2 + dZdy * dgyydy) - (dZdy/np.sqrt(g_yy[:,j,:]))*(dRdx+dRdy+dRdz)
        
        K[:,j,:] = dRdx/dR_length + dRdz/dR_length + dZdx/dZ + dZdz/dZ + dR#/unit_length# + dZ/unit_length
        # K[:,j,:] = (1/(g_yy[:,j,:]**2))*(dR2dy2 + dZ2dy2) - (1/(g_yy[:,j,:]))*(dRdy+dZdy)*dg_yydy - ((dRdy+dZdy)/ np.sqrt(g_yy[:,j,:]))*( dRdx + dRdz ) 

