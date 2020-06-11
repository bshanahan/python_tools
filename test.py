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
