from mesh_to_sdf import sample_sdf_near_surface
import mesh_to_sdf
import trimesh
import pyrender
import numpy as np

import os
from plyfile import PlyData, PlyElement
import pandas as pd

# mesh = trimesh.load('ArteryObjAN1.obj')
# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
# np.savez('mesh.npz',points=points,sdf=sdf)

mesh = np.load('mesh.npz', allow_pickle=True)
points = mesh['points']
sdf = mesh['sdf']
points = points[sdf<0]

colors = np.zeros(points.shape)
colors[:, 2] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
# todo show the pc of the object

mesh = trimesh.load('ArteryObjAN1.obj')
mesh = mesh_to_sdf.
t = mesh_to_sdf.get_surface_point_cloud(mesh, sample_point_count=100000, calculate_normals=True)
print(t.points)

#print(data_np)
colors2 = np.zeros(t.points.shape)
# min = data_np[:,:3].min(axis=0)
# max = data_np[:,:3].max(axis=0)
# data_np[:,:3] = (data_np[:,:3]-min)/max
colors2[:, 0] = 1
cloud2 = pyrender.Mesh.from_points(t.points, colors=colors2)
scene = pyrender.Scene()
scene.add(cloud)
scene.add(cloud2)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)