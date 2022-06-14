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
min = points.min(axis=0)
max = points.max(axis=0)
points = 
sdf = mesh['sdf']
points = points[sdf<0]

colors = np.zeros(points.shape)
colors[:, 2] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
# todo load the pc of the object


file_dir = 'pc.ply'  #文件的路径
plydata = PlyData.read(file_dir)  # 读取文件

data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    data_np[:, i] = data_pd[name]

#print(data_np)
colors2 = np.zeros(data_np[:,:3].shape)
colors2[:, 0] = 1
cloud2 = pyrender.Mesh.from_points(data_np[:,:3], colors=colors2)
scene = pyrender.Scene()
scene.add(cloud)
#scene.add(cloud2)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)