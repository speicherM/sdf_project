from mesh_to_sdf import sample_sdf_near_surface

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
# todo load the pc of the object


scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)