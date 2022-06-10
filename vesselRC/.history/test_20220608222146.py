from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
from utils import point_utils as pu
# mesh = trimesh.load('ArteryObjAN1.obj')
# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
# np.savez('mesh.npz',points=points,sdf=sdf)

mesh = np.load('mesh.npz', allow_pickle=True)
points = mesh['points']
pu.