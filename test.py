from osa import Client
import numpy as np
import matplotlib.pyplot as plt

srv2 = Client("http://esb.ipp-hgw.mpg.de:8280/services/MeshSrv?wsdl")

#describe geometry
mset = srv2.types.SurfaceMeshSet() # a set of surface meshes
mset.references = []
for i in  [17, 16, 15, 9, 7, 6, 5, 2]: # assemblies ids from database
    ref = srv2.types.DataReference() # for each assembly a data reference
    ref.dataId = "%d" %i
    mset.references.append(ref)
# add references to single components, in this case ports
w1 = srv2.types.SurfaceMeshWrap()
ref = srv2.types.DataReference()
ref.dataId = "387" # component id
w1.reference = ref
w2 = srv2.types.SurfaceMeshWrap()
ref = srv2.types.DataReference()
ref.dataId = "387" # component id
w2.reference = ref
mset.meshes = [w1, w2]

# intersection call for phi = 0
result = srv2.service.intersectMeshPhiPlane(0.0, mset)
    
# plotting
for s in result: #loop over non-empty triangle intersections
    xyz = np.array((s.vertices.x1, s.vertices.x2, s.vertices.x3)).T
    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2) # major radius
    z = xyz[:,2]
    plt.plot(R, z, "-k", lw = .5, zorder=-20)

plt.show()




for y, index in zip(np.linspace(0.01,2*np.pi/5.,16),np.arange(0,16)) :
    S = field.Sf(RR,y,ZZ)
    plt.contour(RR, ZZ, S, np.linspace(0.0,0.01,14), colors='black')
    plt.axis("equal")
    plt.xlabel('R [m]', fontsize=18)
    plt.ylabel('R [m]', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.savefig('Dommaschk_S_'+str(index)+'.png')
    plt.clf()
