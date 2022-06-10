from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np

import os
from plyfile import PlyData, PlyElement
import pandas as pd

mesh = trimesh.load('ArteryObjAN1.obj')
mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=10000000, calculate_normals=True)
np.savez('mesh.npz',points=points,sdf=sdf)