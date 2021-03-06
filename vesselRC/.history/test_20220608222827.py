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
print(points.max(axis=0))
 _, occ_idx, grid_shape = pu.np_get_occupied_idx(
            #point_samples[:100000, :3],
            points[:, :3],
            xmin=xmin - 0.5 * self.part_size,
            xmax=xmax + 0.5 * self.part_size,
            crop_size=self.part_size,
            ntarget=1,  # we do not require `point_crops` (i.e. `_` in returns), so we set it to 1
            overlap=self.overlap,
            normalize_crops=False,
            return_shape=True)
#pu.np_get_occupied_idx()