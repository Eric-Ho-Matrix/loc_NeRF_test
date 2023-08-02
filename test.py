import numpy as np
import torch
from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nerf.network import NeRFNetwork

import cv2

# sensor_image = cv2.imread("0.png")

# img = np.copy(sensor_image)

# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# sift = cv2.SIFT_create()
# keypoints = sift.detect(img, None)

# # Initiate ORB detector
# # orb = cv2.ORB_create()
# # find the keypoints with ORB
# # keypoints2 = orb.detect(img_gray,None)

# feat_img = cv2.drawKeypoints(img, keypoints, img)

# #keypoints = keypoints + keypoints2
# #keypoints = keypoints2

# xy = [keypoint.pt for keypoint in keypoints]
# xy = np.array(xy).astype(int)

# # Remove duplicate points
# xy_set = set(tuple(point) for point in xy)
# xy = np.array([list(point) for point in xy_set]).astype(int)

# obs_img_noised = sensor_image
# W_obs = sensor_image.shape[0]
# H_obs = sensor_image.shape[1]

# # find points of interest of the observed image
# POI = xy

# obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)
# obs_img_noised = torch.tensor(obs_img_noised).cuda()

# # create meshgrid from the observed image
# coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, H_obs - 1, H_obs), np.linspace(0, W_obs - 1, W_obs)), -1), dtype=int)

# print(coords.shape)

# # create sampling mask for interest region sampling strategy
# interest_regions = np.zeros((H_obs, W_obs, ), dtype=np.uint8)
# interest_regions[POI[:,0], POI[:,1]] = 1
# I = 3
# interest_regions = cv2.dilate(interest_regions, np.ones((5, 5), np.uint8), iterations=I)

# interest_regions = np.array(interest_regions, dtype=bool)
# print(interest_regions)

# interest_regions = coords[interest_regions]

# print(interest_regions)




### testbench for 2 different methods to calculate orientation given a bezier curve

## method 1 : use exp mapping directly

# def skew_matrix(vec):
#     batch_dims = vec.shape[:-1]
#     S = torch.zeros(*batch_dims, 3, 3)
#     S[..., 0, 1] = -vec[..., 2]
#     S[..., 0, 2] =  vec[..., 1]
#     S[..., 1, 0] =  vec[..., 2]
#     S[..., 1, 2] = -vec[..., 0]
#     S[..., 2, 0] = -vec[..., 1]
#     S[..., 2, 1] =  vec[..., 0]
#     return S

# def next_rotation(R, omega, dt):
#     # Propagate rotation matrix using exponential map of the angle displacements
#     angle = omega*dt
#     theta = torch.norm(angle, p=2)
#     if theta == 0:
#         exp_i = torch.eye(3)
#     else:
#         exp_i = torch.eye(3)
#         angle_norm = angle / theta
#         K = skew_matrix(angle_norm)
#         exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

#     next_R = R @ exp_i
#     return next_R

# start_R = torch.tensor([[0,-1.0,0],[1.0,0,0],[0,0,1.0]])

# omega = torch.tensor([0.2,0.3,0.1])

# dt = 0.1

# next_R = next_rotation(start_R, omega, dt)

# print(next_R)

## method 2: Find heading direction
# start_R = torch.tensor([[0,-1.0,0],[1.0,0,0],[0,0,1.0]])

# initial_accel = torch.tensor([10.0])

# start_accel = start_R @ torch.tensor([0,0,1.0]) * initial_accel

# accel_mag = torch.norm(start_accel, dim=-1, keepdim=True)

# # needs to be pointing in direction of acceleration
# z_axis_body = start_accel/accel_mag

# # remove states with rotations already constrained
# x,y,_ = start_R @ torch.tensor( [1.0, 0, 0 ] )

# print(start_R @ torch.tensor( [1.0, 0, 0 ] ))

# z_angle = torch.atan2(y, x)

# in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)    # what is this for ??  z_angle in the first iteration seems randomly generated 

# x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
# x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
# y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

# # S, 3, 3 # assembled manually from basis vectors
# next_R = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

# print(next_R)

### End testbench for get orientation


# coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, 20 - 1, 20), np.linspace(0, 20 - 1, 20)), -1), dtype=int).reshape(20 * 20, 2)

# print(coords)

# coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, 20 - 1, 20), np.linspace(0, 20 - 1, 20)), -1), dtype=int)

# coords = coords.reshape(20 * 20, 2)

# rand_inds = np.random.choice(coords.shape[0], 10, replace=False)

# print(rand_inds)

# print(coords[rand_inds])

a = np.random.normal(size = (10))

print(a)