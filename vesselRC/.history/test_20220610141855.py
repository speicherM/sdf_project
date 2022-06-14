from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np

import os
from plyfile import PlyData, PlyElement
import pandas as pd

mesh = trimesh.load('ArteryObjAN1.obj')
points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
np.savez('mesh.npz',points=points,sdf=sdf)