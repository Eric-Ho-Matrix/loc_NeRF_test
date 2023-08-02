import os, sys
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

model.eval()
metrics = [PSNRMeter(),]
criterion = torch.nn.MSELoss(reduction='none')
trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
dataset = NeRFDataset(opt, device=device, type='test')        #Importing dataset in order to get the same camera intrinsics as training
### -----  END OF TORCH-NGP SPECIFIC ----- #

def add_noise_to_state(state, noise):
    return state + noise

class iNeRF:
    def __init__(self, get_rays_fn, render_fn):

        #NERF SPECIFIC CONFIGS
        self.get_rays = get_rays_fn
        self.render_fn = render_fn

        self.iter = 1000
        self.error_print_rate, self.render_rate = [20, 100]
        self.is_filter = True
        self.render_viz = True
        self.dil_iter = 3
        self.batch_size = 1024

        self.iteration = 0

        self.kernel_size = 5

        self.lrate = 1e-3
        
        if self.render_viz:
            self.f, self.axarr = plt.subplots(1, 3, figsize=(15, 50))

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

    def find_POI(self, img_rgb, render=False): # img - RGB image in range 0...255
        img = np.copy(img_rgb)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints = sift.detect(img, None)

        # Initiate ORB detector
        # orb = cv2.ORB_create()
        # find the keypoints with ORB
        # keypoints2 = orb.detect(img_gray,None)

        if render:
            feat_img = cv2.drawKeypoints(img_gray, keypoints, img)
        else:
            feat_img = None

        #keypoints = keypoints + keypoints2
        #keypoints = keypoints2

        xy = [keypoint.pt for keypoint in keypoints]
        xy = np.array(xy).astype(int)

        # Remove duplicate points
        xy_set = set(tuple(point) for point in xy)
        xy = np.array([list(point) for point in xy_set]).astype(int)

        return xy, feat_img # pixel coordinates
    

    def measurement_fn(self, state, target, batch):
      
        H, W, _ = target.shape

        # with torch.no_grad():

        #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #     ax.imshow(target.cpu().numpy())
        #     plt.show()

        #Assuming the camera frustrum is oriented in the body y-axis. The camera frustrum is in the -z axis
        # in its own frame, so we need a 90 degree rotation about the x-axis to transform 
        #TODO: Check this, doesn't look right. Should be camera to world
        R = vec_to_rot_matrix(state[6:9])
        rot = rot_x(torch.tensor(np.pi/2)) @ R[:3, :3]

        pose, trans = nerf_matrix_to_ngp_torch(rot, state[:3])    # what is this for ???

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

        rays = self.get_rays(new_pose.reshape((1, 4, 4)))

        rays_o = rays["rays_o"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]
        rays_d = rays["rays_d"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]

        output = self.render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))
        #output also contains a depth channel for use with depth data if one chooses

        rgb = output['image'].reshape((-1, 3))

        target = target[batch[:, 0], batch[:, 1]]      #TODO: Make sure target size is [H, W, 3]

        loss_rgb = torch.nn.functional.mse_loss(rgb, target)

        loss = loss_rgb

        return loss

    def estimate_relative_pose(self, sensor_image, start_state, obs_img_pose=None):
        #start-state is 12-vector

        obs_img_noised = sensor_image
        W_obs = sensor_image.shape[0]
        H_obs = sensor_image.shape[1]

        # find points of interest of the observed image
        POI, features = self.find_POI(obs_img_noised, render=self.render_viz)  # xy pixel coordinates of points of interest (N x 2)

        print(f'Found {POI.shape[0]} features')
        ### IF FEATURE DETECTION CANT FIND POINTS, RETURN INITIAL
        if len(POI.shape) == 1:
            self.losses = []
            self.states = []
            error_text = 'Feature Detection Failed.'
            print(f'{error_text:.^20}')
            return start_state.clone().detach(), False

        obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)
        obs_img_noised = torch.tensor(obs_img_noised).cuda()

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, H_obs - 1, H_obs), np.linspace(0, W_obs - 1, W_obs)), -1), dtype=int)

        # create sampling mask for interest region sampling strategy
        interest_regions = np.zeros((H_obs, W_obs, ), dtype=np.uint8)
        interest_regions[POI[:,0], POI[:,1]] = 1
        I = self.dil_iter
        interest_regions = cv2.dilate(interest_regions, np.ones((self.kernel_size, self.kernel_size), np.uint8), iterations=I)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]    ## get the interested region (after dilation)

        #Optimzied state is 12 vector initialized as the starting state to be optimized. Add small epsilon to avoid singularities
        optimized_state = start_state.clone().detach() + 1e-6
        optimized_state.requires_grad_(True)

        # Add velocities, omegas, and pose object to optimizer
        if self.is_filter is True:
            optimizer = torch.optim.Adam(params=[optimized_state], lr=self.lrate, betas=(0.9, 0.999), capturable=True)
        else:
            raise('Not implemented')

        # calculate initial angles and translation error from observed image's pose
        if obs_img_pose is not None:
            pose = torch.eye(4)
            pose[:3, :3] = vec_to_rot_matrix(optimized_state[6:9])
            pose[:3, 3] = optimized_state[:3]
            print('initial error', calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose))

        #Store data
        losses = []
        states = []
        POSS_ERR = []
        TIME = []

        POSS_ERR.append(calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose)) ## add initial pose err
        TIME.append(0)
        
        start_t = time.time()
        for k in range(self.iter):
            optimizer.zero_grad()
            rand_inds = np.random.choice(interest_regions.shape[0], size=self.batch_size, replace=False)
            batch = interest_regions[rand_inds]

            #pix_losses.append(loss.clone().cpu().detach().numpy().tolist())
            #Add dynamics loss

            loss = self.measurement_fn(optimized_state, obs_img_noised, batch)

            losses.append(loss.item())
            states.append(optimized_state.clone().cpu().detach().numpy().tolist())

            loss.backward()
            optimizer.step()

            # NOT IMPLEMENTED: EXPONENTIAL DECAY OF LEARNING RATE
            #new_lrate = self.lrate * (0.8 ** ((k + 1) / 100))
            #new_lrate = extra_arg_dict['lrate'] * np.exp(-(k)/1000)
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = new_lrate

            # print results periodically
            if obs_img_pose is not None and ((k + 1) % self.error_print_rate == 0 or k == 0):
                print('Step: ', k)
                print('Loss: ', loss)
                print('State', optimized_state)

                with torch.no_grad():
                    pose = torch.eye(4)
                    pose[:3, :3] = vec_to_rot_matrix(optimized_state[6:9])
                    pose[:3, 3] = optimized_state[:3]
                    pose_error = calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose)
                    
                    print('error', pose_error)
                    print('-----------------------------------')
                    
                    if (k+1) % self.render_rate == 0 :
                        # record pose_err and time comsuption (every 100 steps)
                        TIME.append((time.time() - start_t))
                        POSS_ERR.append(pose_error)
                        
                        if self.render_viz:
                            rgb = self.render_from_pose(pose)
                            rgb = torch.squeeze(rgb).cpu().detach().numpy()
                            
                            #Add keypoint visualization
                            render = rgb.reshape((obs_img_noised.shape[0], obs_img_noised.shape[1], -1))
                            gt_img = obs_img_noised.cpu().numpy()
                            render[batch[:, 0], batch[:, 1]] = np.array([0., 1., 0.])
                            gt_img[batch[:, 0], batch[:, 1]] = np.array([0., 1., 0.])

                            self.f.suptitle(f'Time step: {self.iteration}. Grad step: {k+1}. Trans. error: {pose_error[0]} m. Rotate. error: {pose_error[1]} deg.')
                            self.axarr[0].imshow(gt_img)
                            self.axarr[0].set_title('Ground Truth')

                            self.axarr[1].imshow(features)
                            self.axarr[1].set_title('Features')

                            self.axarr[2].imshow(render)
                            self.axarr[2].set_title('NeRF Render')

                            plt.pause(1)

        print("Done with main relative_pose_estimation loop")
        self.target = obs_img_noised
        self.batch = batch

        self.losses = losses
        self.states = states
        self.pose_error = POSS_ERR
        self.time = TIME
        return optimized_state.clone().detach(), True


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

torch.set_default_tensor_type('torch.cuda.FloatTensor')

camera = Camara(camera_cfg, blender_cfg)
# rot = torch.tensor([1.0,0,0], [0,1.0,0], [0,0,1.0])

init_rates = torch.tensor([0,0,0], device=device)

start_R = torch.tensor([[1.0,0,0],[0,1.0,0],[0,0,1.0]], device=device)

rot_vec = rot_matrix_to_vec(start_R)

start_pos = torch.tensor([0.39, -0.67, 0.2], device=device).float()

newstate = torch.cat( [start_pos, init_rates, rot_vec, init_rates], dim=0)

## noise

mpc_noise_mean = torch.tensor([0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)    # Mean of process noise [positions, lin. vel, angles, ang. rates]
mpc_noise_std = torch.tensor([2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2], device=device)    # standard dev. of noise

noise = torch.normal(mpc_noise_mean, mpc_noise_std)

newstate_noise = add_noise_to_state(newstate, noise)

new_state = newstate_noise.clone().detach()    ## state with noise

new_pose = torch.eye(4)
new_pose[:3, :3] = rot_x(torch.tensor(np.pi/2)) @ vec_to_rot_matrix(new_state[6:9])   
new_pose[:3, 3] = new_state[:3]

camera.data['pose'] = new_pose.tolist()

# Capture image
img = camera.get_img(camera.data)
img = torch.from_numpy(img)    ## GT img return from NeRF model
camera.iter += 1

# Revert camera pose to be in body frame
new_pose[:3, :3] = rot_x(torch.tensor(-np.pi/2)) @ new_pose[:3, :3]    ## return back to robot body

# # visualize obs_img
# f, ax = plt.subplots(1, 2, figsize=(15, 20))
# ax[0].set_title('Ground Truth')
# ax[0].imshow(img)

# plt.pause(20)

# Rendering from the NeRF functions
render_fn = lambda rays_o, rays_d: model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
get_rays_fn = lambda pose: get_rays(pose, dataset.intrinsics, dataset.H, dataset.W)

Estimator = iNeRF(get_rays_fn, render_fn)
# batch, features = Estimator.find_POI(img, render=True)

# ax[1].set_title('Features')
# ax[1].imshow(features)

# define initial optimization state

init_rates = torch.tensor([0.01,0.005,0], device=device)

start_R = torch.tensor([[0.99,0,0],[0,0.95,0],[0,0,0.92]], device=device)

rot_vec = rot_matrix_to_vec(start_R)

start_pos = torch.tensor([0.35, -0.75, 0.3], device=device).float()  # 0.39, -0.67, 0.2

start_state = torch.cat( [start_pos, init_rates, rot_vec, init_rates], dim=0)

final_state = Estimator.estimate_relative_pose(img, start_state, obs_img_pose=new_pose.cpu().numpy())

# plot all error and time comsuption
fig, ax = plt.subplots(1, 4, figsize=(30, 4))

xx = [i for i in range(1000) if i == 0 or (i+1) % 100 == 0]

ax[0].plot(xx, [Estimator.losses[i] for i in xx], label = 'Loss') 
ax[0].set_title("Loss every {} trials".format(Estimator.render_rate))

ax[1].plot(xx, Estimator.time, label = 'Time')
ax[1].set_title("Time Consuption every {} trials".format(Estimator.render_rate))

ax[2].plot(xx, [Err_[0] for Err_ in Estimator.pose_error], label = 'Trans. err')
ax[2].set_title("Trans. Error every {} trials".format(Estimator.render_rate))

ax[3].plot(xx, [Err_[1] for Err_ in Estimator.pose_error], label = 'Rot. err')
ax[3].set_title("Rotate Error every {} trials".format(Estimator.render_rate))

plt.show()
