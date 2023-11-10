"""
TO BE DELETED ON RELEASE
Temp script to test my snippets
"""
import numpy as np
from scipy.spatial.transform import Rotation
A = [np.array([[-0.16658794,-0.98439172,0.05675115,0.03130435],
[-0.40065415,0.01498771,-0.91610716,0.08209651],
[0.90095712,-0.17534935,-0.39689811,0.83262567],
[0,0,0,1]]),
np.array([[-0.13629521,-0.98743576,0.07995924,0.02984001],
[-0.36697875,-0.02464675,-0.92990315,0.0772117],
[0.92018982,-0.15608406,-0.35900939,0.8243955],
[0,0,0,1]]),
np.array([[-0.18768764,-0.9805611,0.0572063,0.03134696],
[-0.41664948,0.02673757,-0.90867428,0.06645181],
[0.88948053,-0.19438128,-0.41356934,0.83351511],
[0,0,0,1]]),
np.array([[-0.14390719,-0.98944202,0.01716066,0.03144084],
[-0.4231933,0.04585607,-0.90487861,0.0765878],
[0.89453745,-0.13748021,-0.42532501,0.82922846],
[0,0,0,1]]),
np.array([[-0.19042757,-0.97876509,0.07586512,0.03383361],
[-0.39542726,0.00574141,-0.91847973,0.06561362],
[0.89853972,-0.20490235,-0.38812443,0.82512993],
[0,0,0,1]]),
np.array([[-0.16798417,-0.9838861,0.06122703,0.03271796],
[-0.42791527,0.01682529,-0.90366259,0.07794089],
[0.8880703,-0.17800037,-0.42384702,0.82707524],
[0,0,0,1]]),
np.array([[-0.18238554,-0.97913675,0.08958837,0.02429443],
[-0.4209898,-0.00457468,-0.90705418,0.07917912],
[0.88853936,-0.20314873,-0.41137298,0.83418525],
[0,0,0,1]]),
np.array([[-0.15299591,-0.98784097,0.0275988,0.02915702],
[-0.37881177,0.03282965,-0.92489165,0.07116388],
[0.91273924,-0.15195877,-0.37922928,0.82146414],
[0,0,0,1]]),
np.array([[-0.18111851,-0.98064845,0.07432384,0.02864799],
[-0.39392401,0.0030927,-0.91913819,0.07145297],
[0.901121,-0.19575025,-0.38686182,0.84190764],
[0,0,0,1]]),
np.array([[-0.14461576,-0.98550864,0.08864671,0.03598223],
[-0.4206022,-0.01986744,-0.907028,0.06535205],
[0.89564454,-0.16845492,-0.4116347,0.83276494],
[0,0,0,1]])]

trans = None
for i in A:
    if trans is None:
        trans = i[:3,-1].reshape(1,-1)
    else:
        trans = np.row_stack((trans,i[:3,-1].reshape(1,-1)))
print(np.std(trans, axis=0))



trans = None
for i in A:
    i = Rotation.from_matrix(i[:3,:3]).as_rotvec()
    if trans is None:
        trans = i.reshape(1,-1)
    else:
        trans = np.row_stack((trans,i.reshape(1,-1)))
print(np.std(trans*180/3.14, axis=0))


a = np.load(r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\validation_data_blender_DR\0000004meta.npz")
for file in a.files:
    print(file)
    print(a[file])