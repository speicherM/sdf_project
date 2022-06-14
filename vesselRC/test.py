from mesh_to_sdf import sample_sdf_near_surface
import mesh_to_sdf
import trimesh
import pyrender
import numpy as np
from utils import point_utils as pu
from tqdm import tqdm
from tqdm._tqdm import trange
from time import sleep
# mesh = trimesh.load('ArteryObjAN1.obj')
# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
# np.savez('mesh.npz',points=points,sdf=sdf)

# mesh = np.load('pc.npz', allow_pickle=True)
# #points = np.concatenate([mesh['points'],mesh['sdf'].reshape(-1,1)], axis=1)
# points = mesh['points']
# mesh = trimesh.load('ArteryObjAN1.obj')
# mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
# t = mesh_to_sdf.get_surface_point_cloud(mesh, sample_point_count=100000, calculate_normals=True)
# #print(t.points)
# grid_points, occ_idx, grid_shape = pu.np_get_occupied_idx(
#         #point_samples[:100000, :3],
#             t.points,
#             xmin = (-1.,-1.,-1.),
#             crop_size=0.125,
#             ntarget=1024,  # we do not require `point_crops` (i.e. `_` in returns), so we set it to 1
#             overlap=True,
#             normalize_crops=False,
#             return_shape=True)
# grid_points.shape



# def process_mesh(mesh_filepath, target_filepath):
#     print(mesh_filepath + " --> " + target_filepath)
#     mesh = trimesh.load(mesh_filepath)
#     points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
#     sdf_points= np.concatenate([points,sdf.reshape(-1,1)], axis=1)
#     mesh = trimesh.load(mesh_filepath)
#     mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
#     t = mesh_to_sdf.get_surface_point_cloud(mesh, sample_point_count=100000, calculate_normals=True)
#     np.savez(target_filepath, sdf_points = sdf_points,point_cloud = t.points)

# pbar = tqdm(["a", "b", "c", "d"],colour='black')
# for char in pbar:
#     pbar.set_description("Processing %s" % char)
#     sleep(1)
 