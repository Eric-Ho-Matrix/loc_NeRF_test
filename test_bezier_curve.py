import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def DrawAxis(pos, rot, ax=Axes3D.axes):
    ax.quiver(pos[0], pos[1], pos[2], rot[0,0], rot[1,0], rot[2,0], color='b', label='X')
    ax.quiver(pos[0], pos[1], pos[2], rot[0,1], rot[1,1], rot[2,1], color='g', label='Y')
    ax.quiver(pos[0], pos[1], pos[2], rot[0,2], rot[1,2], rot[2,2], color='r', label='Z')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

P0, P1, P2, P3 = torch.tensor(((0,0,0),(20,0,0),(0,20,0),(30,30,0)), dtype=torch.float32, device=device)

t = torch.arange(0,1,0.01, device=device, requires_grad=True)
# t = torch.tensor([0.0], requires_grad=True, device=device)

a0 = (1-t)**3
a1 = 3*(1-t)**2 * t
a2 = 3 * t**2 * (1-t)
a3 = t ** 3

x = a0*P0[0] + a1*P1[0] + a2*P2[0] + a3*P3[0]
y = a0*P0[1] + a1*P1[1] + a2*P2[1] + a3*P3[1]
z = a0*P0[2] + a1*P1[2] + a2*P2[2] + a3*P3[2]

vx = torch.autograd.grad(x,t,grad_outputs=torch.ones_like(x), create_graph=True)[0]
vy = torch.autograd.grad(y,t,grad_outputs=torch.ones_like(y), create_graph=True)[0]
vz = torch.autograd.grad(z,t,grad_outputs=torch.ones_like(z), create_graph=True)[0]

ax = torch.autograd.grad(vx,t,grad_outputs=torch.ones_like(x), retain_graph=True)[0]
ay = torch.autograd.grad(vy,t,grad_outputs=torch.ones_like(y), retain_graph=True)[0]
az = torch.autograd.grad(vz,t,grad_outputs=torch.ones_like(z), retain_graph=True)[0]

V_total = torch.stack([vx,vy,vz], dim=-1)
X_body = V_total / torch.norm(V_total, dim=-1, keepdim=True)

A_total = torch.stack([ax,ay,az], dim=-1)
A_total_body = A_total + torch.tensor([0, 0, 10.0], device=device)
Z_body = A_total_body / torch.norm(A_total_body, dim=-1, keepdim=True)

Y_body = torch.cross(Z_body,X_body,dim=-1)
Y_body /= torch.norm(Y_body, dim=-1, keepdim=True)

## create rot matrix
X_body = X_body.unsqueeze(1)
Y_body = Y_body.unsqueeze(1)
Z_body = Z_body.unsqueeze(1)

Rot = torch.cat((X_body, Y_body, Z_body), dim=1).swapdims(-1,-2).detach().cpu().numpy()

POS = torch.stack([x,y,z], dim=-1).detach().cpu().numpy()



# plt.figure()
# plt.plot(t.detach().cpu().numpy(), x.detach().cpu().numpy(), label='pos')
# plt.plot(t.detach().cpu().numpy(), vx.detach().cpu().numpy(), label='vel')
# plt.plot(t.detach().cpu().numpy(), ax.detach().cpu().numpy(), label='accel')
# plt.show()


ax = plt.axes(projection='3d')
ax.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), z.detach().cpu().numpy(), lw=2, c='r')
# ax.scatter(x.item(), y.item(), z.item(), c='r', s=50) 
for i in range(Rot.shape[0]):
    if i%20 == 0:
        DrawAxis(POS[i], Rot[i], ax=ax)

ax.set_xlabel('X', labelpad=5)
ax.set_ylabel('Y', labelpad=5)
ax.set_zlabel('Z', labelpad=5)
ax.set_title('Bezier curve')
plt.tight_layout()
plt.show()
