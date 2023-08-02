import os, sys
sys.path.append("/home/qin/Desktop/test_for_loc_NeRF")
import numpy as np
import torch
import shutil
import pathlib
import subprocess
from tqdm import trange
import argparse
from nerf.utils import *
from nerf.provider import NeRFDataset
import json
from nav.math_utils import vec_to_rot_matrix, mahalanobis, rot_x, nerf_matrix_to_ngp_torch, nearestPD, calcSE3Err

# Import Helper Classes
from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)
from nav.math_utils import vec_to_rot_matrix, rot_matrix_to_vec, rot_x, skew_matrix_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### End of original import header

### start loc-NeRF import header
from turtle import pos

import numpy as np
import gtsam
import cv2
import torch
import time
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import locnerf
from full_filter import NeRF
from particle_filter import ParticleFilter
from utils import get_pose
### End of loc-NeRF import header


class Navigator():
    def __init__(self, initial_pose, obs_img_pose, sensor_image, get_rays, render_fn):
        self.num_particles = 200
        self.run_inerf_compare = 1
        self.use_weighted_avg = 1
        self.all_pose_est = []
        # self.min_bounds = {'px':1.0, 'py':1.0, 'pz':1.0, 'rx':1.0, 'ry':1.0, 'rz':1.0}    # noise bound
        # self.max_bounds = {'px':1.0, 'py':1.0, 'pz':1.0, 'rx':1.0, 'ry':1.0, 'rz':1.0}
        self.gt_pose = obs_img_pose.cpu().numpy()
        self.initial_pose = initial_pose.cpu().numpy()
        self.center_about_pred_pose = False
        self.forward_passes_limit = 78643200
        self.rgb_input_count = 0
        self.use_convergence_protection = True
        self.course_samples = 64   # number course samples per ray
        self.fine_samples = 64     # number fine samples per ray
        self.number_convergence_particles = 10
        self.log_results = True
        self.sampling_strategy = 'random'
        self.batch_size = 64
        self.convergence_noise = 0.05
        self.photometric_loss = "rgb"
        self.sensor_image = sensor_image
        self.num_updates = 0
        self.use_refining = True
        self.alpha_super_refine = 0.05
        self.alpha_refine = 0.09
        self.use_particle_reduction = True
        self.min_number_particles = 100
        self.run_predicts = False

        self.W = sensor_image.shape[0]
        self.H = sensor_image.shape[1]

        self.get_rays  =  get_rays
        self.render_fn =  render_fn

        # create meshgrid from the observed image
        self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.W - 1, self.W), np.linspace(0, self.H - 1, self.H)), -1),
                            dtype=int)

        self.coords = self.coords.reshape(self.H * self.W, 2)

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)


        # Set initial distribution of particles.
        self.get_initial_distribution(self.initial_pose)

        # Add initial pose estimate before first update step is run.
        if self.use_weighted_avg:
            position_est = self.filter.compute_weighted_position_average()
        else:
            position_est = self.filter.compute_simple_position_average()

        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = gtsam.Pose3(rot_est, position_est).matrix()
        self.all_pose_est.append(pose_est)

    def get_initial_distribution(self, initial_pose):
        # for non-global loc mode, get random pose based on iNeRF evaluation method from their paper
        # sample random axis from unit sphere and then rotate by a random amount between [-40, 40] degrees
        # translate along each axis by a random amount between [-10, 10] cm
        rot_rand = 20.0

        trans_rand = 0.1
        
        # get random axis and angle for rotation
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        axis = np.array([x,y,z])
        axis = axis / np.linalg.norm(axis)
        angle = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
        euler = (gtsam.Rot3.AxisAngle(axis, angle)).ypr()     

        # euler[0], euler[1], euler[2] = gtsam.Rot3.Rot(vec_to_rot_matrix(initial_pose[6:9])).rpy()
        # initial_pose = initial_pose.detach().cpu().numpy()
        euler[0] = initial_pose[6]
        euler[1] = initial_pose[7]
        euler[2] = initial_pose[8]
        t_x = initial_pose[0]
        t_y = initial_pose[1]
        t_z = initial_pose[2]

        self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, 0, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 0, 0]), size = (self.num_particles, 6))

        # center translation at randomly sampled position
        self.initial_particles_noise[:, 0] += t_x
        self.initial_particles_noise[:, 1] += t_y
        self.initial_particles_noise[:, 2] += t_z

        for i in range(self.initial_particles_noise.shape[0]):
            # rotate random 3 DOF rotation about initial random rotation for each particle
            n1 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
            n2 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
            n3 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
            euler_particle = gtsam.Rot3.AxisAngle(axis, angle).retract(np.array([n1, n2, n3])).ypr()

            # add rotation noise for initial particle distribution
            self.initial_particles_noise[i,3] = euler_particle[0] * 180.0 / np.pi
            self.initial_particles_noise[i,4] = euler_particle[1] * 180.0 / np.pi 
            self.initial_particles_noise[i,5] = euler_particle[2] * 180.0 / np.pi  

        self.initial_particles = self.set_initial_particles()
        self.filter = ParticleFilter(self.initial_particles)

    def set_initial_particles(self):
        ## get translation matrix
        reshape_initial_pose = np.eye(4)
        reshape_initial_pose[:3,:3] = gtsam.Rot3.Ypr(self.initial_pose[6],self.initial_pose[7],self.initial_pose[8]).matrix()
        reshape_initial_pose[:3,3] = self.initial_pose[:3]

        self.initial_pose = reshape_initial_pose

        initial_positions = np.zeros((self.num_particles, 3))
        rots = []
        for index, particle in enumerate(self.initial_particles_noise):
            x = particle[0]
            y = particle[1]
            z = particle[2]
            phi = particle[3]
            theta = particle[4]
            psi = particle[5]

            particle_pose = get_pose(phi, theta, psi, x, y, z, self.initial_pose, self.center_about_pred_pose)
            
            # set positions
            initial_positions[index,:] = [particle_pose[0,3], particle_pose[1,3], particle_pose[2,3]]
            # set orientations
            rots.append(gtsam.Rot3(particle_pose[0:3,0:3]))
            # print(initial_particles)


        return {'position':initial_positions, 'rotation':np.array(rots)}
    
    def check_if_position_error_good(self, return_error = False):
        """
        check if position error is less than 5cm, or return the error if return_error is True
        """
        acceptable_error = 0.05
        if self.use_weighted_avg:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average())
            if return_error:
                return error
            return error < acceptable_error
        else:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average())
            if return_error:
                return error
            return error < acceptable_error
        
    def check_if_rotation_error_good(self, return_error = False):
        """
        check if rotation error is less than 5 degrees, or return the error if return_error is True
        """
        acceptable_error = 5.0
        average_rot_t = (self.filter.compute_simple_rotation_average()).transpose()
        # check rot in bounds by getting angle using https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices

        r_ab = average_rot_t @ (self.gt_pose[0:3,0:3])
        rot_error = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
        print("rotation error: ", rot_error)
        if return_error:
            return rot_error
        return abs(rot_error) < acceptable_error

    def render_from_pose(self, pose):
        rot = rot_x(torch.tensor(np.pi/2)) @ pose[:3, :3]
        trans = pose[:3, 3]
        pose, trans = nerf_matrix_to_ngp_torch(rot, trans)

        new_pose = torch.eye(4)
        new_pose[:3, :3] = pose
        new_pose[:3, 3] = trans

        rays = self.get_rays(new_pose.reshape((1, 4, 4)))

        output = self.render_fn(rays["rays_o"], rays["rays_d"])
        #output also contains a depth channel for use with depth data if one chooses

        rgb = torch.squeeze(output['image'])

        return rgb

    def set_noise(self, scale):
        self.px_noise = 0.01 / scale
        self.py_noise = 0.01 / scale
        self.pz_noise = 0.01 / scale
        self.rot_x_noise = 0.02 / scale
        self.rot_y_noise = 0.02 / scale
        self.rot_z_noise = 0.02 / scale

    def check_refine_gate(self):
    
        # get standard deviation of particle position
        sd_xyz = np.std(self.filter.particles['position'], axis=0)
        norm_std = np.linalg.norm(sd_xyz)
        refining_used = False
        print("sd_xyz:", sd_xyz)
        print("norm sd_xyz:", np.linalg.norm(sd_xyz))

        if norm_std < self.alpha_super_refine:
            print("SUPER REFINE MODE ON")
            # reduce original noise by a factor of 4
            self.set_noise(scale = 4.0)
            refining_used = True
        elif norm_std < self.alpha_refine:
            print("REFINE MODE ON")
            # reduce original noise by a factor of 2
            self.set_noise(scale = 2.0)
            refining_used = True
        else:
            # reset noise to original value
            self.set_noise(scale = 1.0)
        
        if refining_used and self.use_particle_reduction:
            self.filter.reduce_num_particles(self.min_number_particles)
            self.num_particles = self.min_number_particles

    
    def get_loss(self, partical_pose, batch, photometric_loss='rgb'):

        H, W, _ = self.sensor_image.shape

        target_s = self.sensor_image[batch[:, 0], batch[:, 1]] # TODO check ordering here 
        target_s = (np.array(target_s) / 255.).astype(np.float32)
        target_s = torch.Tensor(target_s).to(device)

        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(self.sensor_image)
        # plt.show()

        start_time = time.time()

        RGB = []

        for i, state in enumerate(partical_pose):

            R = state[:3,:3]
            rot = rot_x(torch.tensor(np.pi/2)) @ R[:3, :3]

            pose, trans = nerf_matrix_to_ngp_torch(rot, state[:3,3])    # transfer to torch-ngp coods

            new_pose = torch.eye(4)
            new_pose[:3, :3] = pose
            new_pose[:3, 3] = trans

            # with torch.no_grad():
            #     render = self.render_from_pose(new_pose)
            #     # rgb = self.render_from_pose(pose)
            #     render = torch.squeeze(render).cpu().detach().numpy()
                
            #     #Add keypoint visualization
            #     render = render.reshape((800, 800, -1))

            #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            #     ax.imshow(render)
            #     plt.show()

            with torch.no_grad():
                
                rays = self.get_rays(new_pose.reshape((1, 4, 4)))

                rays_o = rays["rays_o"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]
                rays_d = rays["rays_d"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]

                output = self.render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))

            RGB.append(output['image'].reshape((-1, 3)))

        nerf_time = time.time() - start_time
        
        # RGB = output['image'].reshape((len(partical_pose),-1, 3))

        losses = []
        for i in range(len(partical_pose)):
            rgb = RGB[i]

            if photometric_loss == 'rgb':
                # loss = torch.nn.functional.mse_loss(rgb, target_s)
                loss = self.img2mse(rgb.reshape((-1, 3)), target_s.reshape(-1,3))

            else:
                # TODO throw an error
                print("DID NOT ENTER A VALID LOSS METRIC")
            losses.append(loss.item())
        return losses, nerf_time

    
    def rgb_run(self, render_full_image=False):
        print("processing image")
        start_time = time.time()
        self.rgb_input_count += 1

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = [gtsam.Rot3(i.matrix()) for i in self.filter.particles['rotation']]

        if self.use_convergence_protection:
            for i in range(self.number_convergence_particles):
                t_x = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_y = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_z = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                # TODO this is not thread safe. have two lines because we need to both update
                # particles to check the loss and the actual locations of the particles
                self.filter.particles["position"][i] = self.filter.particles["position"][i] + np.array([t_x, t_y, t_z])
                particles_position_before_update[i] = particles_position_before_update[i] + np.array([t_x, t_y, t_z])

        total_nerf_time = 0

        if self.sampling_strategy == 'random':
            rand_inds = np.random.choice(self.coords.shape[0], size=self.batch_size, replace=False)
            batch = self.coords[rand_inds]

        loss_poses = []
        for index, particle in enumerate(particles_position_before_update):
            loss_pose = np.zeros((4,4))
            rot = particles_rotation_before_update[index]
            loss_pose[0:3, 0:3] = rot.matrix()
            loss_pose[0:3,3] = particle[0:3]
            loss_pose[3,3] = 1.0
            loss_poses.append(torch.tensor(loss_pose, device = device, dtype=torch.float32))
        losses, nerf_time = self.get_loss(loss_poses, batch, self.photometric_loss)
   
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/losses[index]
        total_nerf_time += nerf_time

        self.filter.update()
        self.num_updates += 1
        print("UPDATE STEP NUMBER", self.num_updates, "RAN")
        print("number particles:", self.num_particles)

        if self.use_refining: # TODO make it where you can reduce number of particles without using refining
            self.check_refine_gate()

        if self.use_weighted_avg:
            avg_pose = self.filter.compute_weighted_position_average()
        else:
            avg_pose = self.filter.compute_simple_position_average()

        avg_rot = self.filter.compute_simple_rotation_average()
        self.nerf_pose = gtsam.Pose3(avg_rot, gtsam.Point3(avg_pose[0], avg_pose[1], avg_pose[2])).matrix()

        if self.use_weighted_avg:
            position_est = self.filter.compute_weighted_position_average()
            print("average position of all particles: ", position_est)
            print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average()))
        else:
            position_est = self.filter.compute_simple_position_average()
            print("average position of all particles: ", position_est)
            print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average()))

        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = gtsam.Pose3(rot_est, position_est).matrix()

        if self.log_results:
            self.all_pose_est.append(pose_est)
    
        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        if not self.run_predicts:
            self.filter.predict_no_motion(self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise) #  used if you want to localize a static image
        
        # return is just for logging
        return pose_est




### ------ TORCH-NGP SPECIFIC ----- ###
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
parser.add_argument('--test', action='store_true', help="test mode")
parser.add_argument('--workspace', type=str, default='workspace')
parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=30000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='latest')
parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

### network backbone options
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

### dataset options
parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
# (the default value is for the fox dataset)
parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=1920, help="GUI width")
parser.add_argument('--H', type=int, default=1080, help="GUI height")
parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

### experimental
parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

opt = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.empty_cache()

if opt.O:
    opt.fp16 = True
    opt.cuda_ray = False
    opt.preload = False

if opt.ff:
    opt.fp16 = False
    assert opt.bg_radius <= 0, "background model is not implemented for --ff"
    from nerf.network_ff import NeRFNetwork
elif opt.tcnn:
    opt.fp16 = False
    assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
    from nerf.network_tcnn import NeRFNetwork
else:
    from nerf.network import NeRFNetwork

seed_everything(opt.seed)

model = NeRFNetwork(
    encoding="hashgrid",
    bound=opt.bound,
    cuda_ray=opt.cuda_ray,
    density_scale=1,
    min_near=opt.min_near,
    density_thresh=opt.density_thresh,
    bg_radius=opt.bg_radius,
)

# # model.eval()
metrics = [PSNRMeter(),]
criterion = torch.nn.MSELoss(reduction='none')
trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
dataset = NeRFDataset(opt, device=device, type='test')        #Importing dataset in order to get the same camera intrinsics as training
### -----  END OF TORCH-NGP SPECIFIC ----- #

def add_noise_to_state(state, noise):
    return state + noise

class Camara:
    def __init__(self, camera_cfg, blender_cfg):

    #Initialize camera params
        self.path = camera_cfg['path']
        self.half_res = camera_cfg['half_res']
        self.white_bg = camera_cfg['white_bg']

        self.data = {
        'pose': None,
        'res_x': camera_cfg['res_x'],           # x resolution
        'res_y': camera_cfg['res_y'],           # y resolution
        'trans': camera_cfg['trans'],     # Boolean
        'mode': camera_cfg['mode']             # Must be either 'RGB' or 'RGBA'
        }   

        self.blend = blender_cfg['blend_path']
        self.blend_script = blender_cfg['script_path']

        self.iter = 0


    # capture image
    def get_img(self, data):
        pose_path = self.path + f'/{self.iter}.json'
        img_path = self.path + f'/{self.iter}.png'

        try: 
            with open(pose_path,"w+") as f:
                json.dump(data, f, indent=4)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        # Run the capture image script in headless blender
        subprocess.run(['blender', '-b', self.blend, '-P', self.blend_script, '--', pose_path, img_path])

        try: 
            img = imageio.imread(img_path)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        img = (np.array(img) / 255.0).astype(np.float32)
        if self.half_res is True:
            width = int(img.shape[1]//2)
            height = int(img.shape[0]//2)
            dim = (width, height)

            # resize image
            img = cv2.resize(img, dim)

        if self.white_bg is True:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])

        img = (np.array(img) * 255.).astype(np.uint8)
        print('Received updated image')
        return img

path = 'sim_img_cache/'     # Directory where pose and images are exchanged
blend_file = 'stonehenge.blend'     # Blend file of your scene

camera_cfg = {
'half_res': False,      # Half resolution
'white_bg': True,       # White background
'path': path,           # Directory where pose and images are stored
'res_x': 800,           # x resolution (BEFORE HALF RES IS APPLIED!)
'res_y': 800,           # y resolution
'trans': True,          # Boolean    (Transparency)
'mode': 'RGBA'          # Can be RGB-Alpha, or just RGB
}

blender_cfg = {
'blend_path': blend_file,
'script_path': 'viz_func.py'        # Path to Blender script
}

# Creates a workspace to hold all the trajectory data
basefolder = "paths" / pathlib.Path(opt.workspace)
if basefolder.exists():
    print(basefolder, "already exists!")
    if input("Clear it before continuing? [y/N]:").lower() == "y":
        shutil.rmtree(basefolder)
basefolder.mkdir()
(basefolder / "init_poses").mkdir()
(basefolder / "init_costs").mkdir()
(basefolder / "replan_poses").mkdir()
(basefolder / "replan_costs").mkdir()
(basefolder / "estimator_data").mkdir()
print("created", basefolder)

## instantiate Camara
camera = Camara(camera_cfg, blender_cfg)

### Initial pose for robot in world coods
init_rates = torch.tensor([0,0,0])

start_R = torch.tensor([[1.0,0,0],[0,1.0,0],[0,0,1.0]])

rot_vec = rot_matrix_to_vec(start_R)

start_pos = torch.tensor([0.39, -0.67, 0.2]).float()

start_state = torch.cat( [start_pos, init_rates, rot_vec, init_rates], dim=0)
### END Initial pose def


### Add noise to initial_state to represent GT_state
mpc_noise_mean = torch.tensor([0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0, 0])    # Mean of process noise [positions, lin. vel, angles, ang. rates]
mpc_noise_std = torch.tensor([2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2])    # standard dev. of noise

noise = torch.normal(mpc_noise_mean, mpc_noise_std)

newstate_noise = add_noise_to_state(start_state, noise)

new_state = newstate_noise.clone()  ## state with noise
new_state = new_state.to(device) 
### END of GT_state


## get translation matrix
new_pose = torch.eye(4, device=device)
new_pose[:3, :3] = rot_x(torch.tensor(np.pi/2)) @ vec_to_rot_matrix(new_state[6:9])   ### transfer Euler angle to rot matrix and rot to camera pose (Accord to NeRF_nav)
new_pose[:3, 3] = new_state[:3]  
## end get trans. matrix



camera.data['pose'] = new_pose.tolist()

# Capture image from blender
img = camera.get_img(camera.data)
img = torch.from_numpy(img)    ## GT img return from NeRF model
camera.iter += 1

# Revert camera pose to be in body frame
new_pose[:3, :3] = rot_x(torch.tensor(-np.pi/2)) @ new_pose[:3, :3]    ## return back to robot body     Basically GT_pose from observation


## define Rendering from the NeRF functions
render_fn = lambda rays_o, rays_d: model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
get_rays_fn = lambda pose: get_rays(pose, dataset.intrinsics, dataset.H, dataset.W)
## end def


#### start loc_NeRF

mcl_local = Navigator(start_state, new_pose, img, get_rays_fn, render_fn)    
num_forward_passes_per_iteration = [0]
position_error_good = []
rotation_error_good = []
ii = 0


while num_forward_passes_per_iteration[-1] < mcl_local.forward_passes_limit:
    print()
    print("forward pass limit, current number forward passes:", mcl_local.forward_passes_limit, num_forward_passes_per_iteration[-1])

    position_error_good.append(int(mcl_local.check_if_position_error_good()))
    rotation_error_good.append(int(mcl_local.check_if_rotation_error_good()))
    if ii != 0:
        pos_est = mcl_local.rgb_run()
        num_forward_passes_per_iteration.append(num_forward_passes_per_iteration[ii-1] + mcl_local.num_particles * (mcl_local.course_samples + mcl_local.fine_samples) * mcl_local.batch_size)
    ii += 1
    

