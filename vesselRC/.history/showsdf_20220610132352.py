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


file_dir = '文件的路径'  #文件的路径
plydata = PlyData.read(file_dir)  # 读取文件
data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    data_np[:, i] = data_pd[name]
print(data_np)
————————————————
版权声明：本文为CSDN博主「PROMINEM」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_42445581/article/details/105069655
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)