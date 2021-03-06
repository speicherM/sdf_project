from mesh_to_sdf import sample_sdf_near_surface
import mesh_to_sdf
import trimesh
import pyrender
import numpy as np
from utils import point_utils as pu
# mesh = trimesh.load('ArteryObjAN1.obj')
# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
# np.savez('mesh.npz',points=points,sdf=sdf)

# mesh = np.load('pc.npz', allow_pickle=True)
# #points = np.concatenate([mesh['points'],mesh['sdf'].reshape(-1,1)], axis=1)
# points = mesh['points']
mesh = trimesh.load('ArteryObjAN1.obj')
mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
t = mesh_to_sdf.get_surface_point_cloud(mesh, sample_point_count=100000, calculate_normals=True)
#print(t.points)
grid_points, occ_idx, grid_shape = pu.np_get_occupied_idx(
        #point_samples[:100000, :3],
            tpoints,
            xmin = (-1.,-1.,-1.),
            crop_size=0.125,
            ntarget=1024,  # we do not require `point_crops` (i.e. `_` in returns), so we set it to 1
            overlap=True,
            normalize_crops=False,
            return_shape=True)
grid_points.shape