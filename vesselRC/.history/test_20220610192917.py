from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
from utils import point_utils as pu
# mesh = trimesh.load('ArteryObjAN1.obj')
# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
# np.savez('mesh.npz',points=points,sdf=sdf)

mesh = np.load('pc.npz', allow_pickle=True)
#points = np.concatenate([mesh['points'],mesh['sdf'].reshape(-1,1)], axis=1)
points = mesh['']

grid_points, occ_idx, grid_shape = pu.np_get_occupied_idx(
        #point_samples[:100000, :3],
            points,
            xmin = (-1.,-1.,-1.),
            crop_size=0.125,
            ntarget=1024,  # we do not require `point_crops` (i.e. `_` in returns), so we set it to 1
            overlap=True,
            normalize_crops=False,
            return_shape=True)
grid_points.shape