import time
import torch
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")
# 设置训练参数
maxtrained_number = 5000   # 设置PINN最大训练次数
N_x = 20   # 空间维度配点数
N_y = 20   # 空间维度配点数
N_t = 10   # 时间维度配点数
eL = 1e-5  # 解空间训练终止训练条件
X_min = -np.pi   # 空间维度最小值
X_max = np.pi   # 空间维度最大值
Y_min = -np.pi   # 空间维度最小值
Y_max = np.pi   # 空间维度最大值
T_min = 0.0   # 时间维度最小值
T_max = 1.0   # 时间维度最大值
miu = 0.1   # 方程参数
net_size = [3, 50, 50, 3]   # 网络结构
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type(torch.DoubleTensor)
setup_seed(1)

class NormalizeLayer(nn.Module):
    def __init__(self, a, b):
        super(NormalizeLayer, self).__init__()
        # 确保a和b匹配输入x的维度
        self.register_parameter('a', nn.Parameter(torch.tensor(a, dtype=torch.double, requires_grad=False)))
        self.register_parameter('b', nn.Parameter(torch.tensor(b, dtype=torch.double, requires_grad=False)))
    def forward(self, x):
        # 对每个维度归一化变换
        return self.a * x + self.b
class Net(nn.Module):
    def __init__(self, size=net_size, a=[2 / (X_max - X_min), 2 / (Y_max - Y_min), 2 / (T_max - T_min)], b=[-1 - 2 * X_min / (X_max - X_min), -1 - 2 * Y_min / (Y_max - Y_min), -1 - 2 * T_min / (T_max - T_min)]):
        super(Net, self).__init__()
        self.normalize = NormalizeLayer(a, b)
        layers = []
        for i in range(len(size) - 1):
            layers.append(nn.Linear(size[i], size[i + 1]).double())
            if i < len(size) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x, return_hidden_layer=False):
        x_normalized = self.normalize(x)
        # x_normalized = x # 跳过归一化

        for layer in self.net:
            x_normalized = layer(x_normalized)

        return x_normalized

    def get_hidden_layer_output(self, x):
        return self.forward(x, return_hidden_layer=True)

loss = nn.MSELoss()

def gradients(u, x, order=1):
    if order == 1:
        grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True,
                                   only_inputs=True, allow_unused=True)[0]
        if grad is None:
            raise RuntimeError(f"Variable with requires_grad not connected in graph: {x}")
        return grad
    else:
        return gradients(gradients(u, x, order=1), x, order=order - 1)

def prepare_training_points():
    # interior
    x_line = torch.linspace(X_min, X_max, N_x)
    y_line = torch.linspace(Y_min, Y_max, N_y)
    t_line = torch.linspace(T_min, T_max, N_t)
    x_grid, y_grid, t_grid = torch.meshgrid(x_line, y_line, t_line, indexing='ij')
    x_in = x_grid.reshape(-1, 1)
    y_in = y_grid.reshape(-1, 1)
    t_in = t_grid.reshape(-1, 1)
    X_in = torch.cat([x_in, y_in, t_in], dim=1)
    X_in.requires_grad_(True)

    # initial
    x0 = torch.linspace(X_min, X_max, N_x)
    y0 = torch.linspace(Y_min, Y_max, N_y)
    X0, Y0 = torch.meshgrid(x0, y0, indexing='ij')
    T0 = torch.zeros_like(X0)
    x0f = X0.reshape(-1, 1)
    y0f = Y0.reshape(-1, 1)
    t0f = T0.reshape(-1, 1)
    X_init = torch.cat([x0f, y0f, t0f], dim=1)
    X_init.requires_grad_(True)
    u_exact, v_exact, p_exact = exact(x0f, y0f, t0f)

    # boundary: x-bound
    y_b = torch.linspace(Y_min, Y_max, N_y).reshape(-1, 1)
    t_b = torch.linspace(T_min, T_max, N_t).reshape(-1, 1)
    Yb, Tb = torch.meshgrid(y_b.squeeze(), t_b.squeeze(), indexing='ij')
    Yb, Tb = Yb.reshape(-1, 1), Tb.reshape(-1, 1)
    Xb_min = X_min * torch.ones_like(Yb)
    Xb_max = X_max * torch.ones_like(Yb)

    # boundary: y-bound
    x_b = torch.linspace(X_min, X_max, N_x).reshape(-1, 1)
    Xb, Tb2 = torch.meshgrid(x_b.squeeze(), t_b.squeeze(), indexing='ij')
    Xb, Tb2 = Xb.reshape(-1, 1), Tb2.reshape(-1, 1)
    Yb_min = Y_min * torch.ones_like(Xb)
    Yb_max = Y_max * torch.ones_like(Xb)

    return X_in, X_init, u_exact, v_exact, p_exact, \
           Xb_min, Yb, Tb, Xb_max, Yb, Tb, \
           Xb, Yb_min, Tb2, Xb, Yb_max, Tb2
def exact(x, y, t):
    u = -torch.cos(x) * torch.sin(y) * torch.exp(-2 * miu * t)
    v = torch.sin(x) * torch.cos(y) * torch.exp(-2 * miu * t)
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * torch.exp(-4 * miu * t)
    return u, v, p

def loss_total(net, X_in, X_init, u_exact, v_exact, p_exact,
               Xb_min, Yb1, Tb1, Xb_max, Yb2, Tb2,
               Xb3, Yb_min, Tb3, Xb4, Yb_max, Tb4):
    loss_fn = nn.MSELoss()

    # PDE residual
    x = X_in[:, [0]].clone().detach().requires_grad_(True)
    y = X_in[:, [1]].clone().detach().requires_grad_(True)
    t = X_in[:, [2]].clone().detach().requires_grad_(True)
    X_in_temp = torch.cat([x, y, t], dim=1)
    X_in_temp.requires_grad_(True)
    out = net(X_in_temp)
    u = out[:, [0]]
    v = out[:, [1]]
    p = out[:, [2]]

    u_t = gradients(u, t, 1)
    u_x = gradients(u, x, 1)
    u_y = gradients(u, y, 1)
    u_xx = gradients(u, x, 2)
    u_yy = gradients(u, y, 2)

    v_t = gradients(v, t, 1)
    v_x = gradients(v, x, 1)
    v_y = gradients(v, y, 1)
    v_xx = gradients(v, x, 2)
    v_yy = gradients(v, y, 2)

    p_x = gradients(p, x, 1)
    p_y = gradients(p, y, 1)

    res_u = u_t + u * u_x + v * u_y + p_x - miu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y + p_y - miu * (v_xx + v_yy)
    continuity = u_x + v_y

    loss_pde = loss_fn(res_u, torch.zeros_like(res_u)) + \
               loss_fn(res_v, torch.zeros_like(res_v)) + \
               loss_fn(continuity, torch.zeros_like(continuity))

    # Initial condition
    out_init = net(X_init)
    loss_init = loss_fn(out_init[:, 0:1], u_exact) + \
                loss_fn(out_init[:, 1:2], v_exact) + \
                loss_fn(out_init[:, 2:3], p_exact)

    # Boundary condition
    def boundary_loss(xb, yb, tb):
        Xb = torch.cat([xb, yb, tb], dim=1)
        out = net(Xb)
        u_b, v_b, p_b = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        u_true, v_true, p_true = exact(xb, yb, tb)
        return loss_fn(u_b, u_true) + loss_fn(v_b, v_true) + loss_fn(p_b, p_true)

    loss_bc = boundary_loss(Xb_min, Yb1, Tb1) + boundary_loss(Xb_max, Yb2, Tb2) + \
              boundary_loss(Xb3, Yb_min, Tb3) + boundary_loss(Xb4, Yb_max, Tb4)

    return loss_pde + loss_init + loss_bc

def compute_error(net, x):
    out = net(x).detach().numpy()
    u_pred, v_pred, p_pred = out[:, 0], out[:, 1], out[:, 2]
    u_exact, v_exact, p_exact = exact(x[:, 0], x[:, 1], x[:, 2])
    u_exact = u_exact.detach().numpy()
    v_exact = v_exact.detach().numpy()
    p_exact = p_exact.detach().numpy()

    l2_error_u = np.sqrt(np.sum((u_pred - u_exact) ** 2) / np.sum(u_exact ** 2))
    l2_error_v = np.sqrt(np.sum((v_pred - v_exact) ** 2) / np.sum(v_exact ** 2))
    l2_error_p = np.sqrt(np.sum((p_pred - p_exact) ** 2) / np.sum(p_exact ** 2))

    max_error_u = np.max(np.abs(u_pred - u_exact))
    max_error_v = np.max(np.abs(v_pred - v_exact))
    max_error_p = np.max(np.abs(p_pred - p_exact))

    return max_error_u, max_error_v, max_error_p, l2_error_u, l2_error_v, l2_error_p
def subplot_3(ax, x1, x2, field, title):
    c = ax.tricontourf(x1, x2, field, levels=50, cmap='viridis')
    plt.colorbar(c, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def subplot_2(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_solutions_2d(net, x):
    x1 = x[:, 0].detach().numpy()
    x2 = x[:, 1].detach().numpy()
    out = net(x).detach().numpy()
    u_pred, v_pred, p_pred = out[:, 0], out[:, 1], out[:, 2]
    u_exact, v_exact, p_exact = exact(x[:, 0], x[:, 1], x[:, 2])
    u_exact = u_exact.detach().numpy()
    v_exact = v_exact.detach().numpy()
    p_exact = p_exact.detach().numpy()

    error_u = np.abs(u_pred - u_exact)
    error_v = np.abs(v_pred - v_exact)
    error_p = np.abs(p_pred - p_exact)

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    subplot_3(axs[0, 0], x1, x2, u_pred, "u_pred")
    subplot_3(axs[0, 1], x1, x2, u_exact, "u_exact")
    subplot_3(axs[0, 2], x1, x2, error_u, "|u_error|")

    subplot_3(axs[1, 0], x1, x2, v_pred, "v_pred")
    subplot_3(axs[1, 1], x1, x2, v_exact, "v_exact")
    subplot_3(axs[1, 2], x1, x2, error_v, "|v_error|")

    subplot_3(axs[2, 0], x1, x2, p_pred, "p_pred")
    subplot_3(axs[2, 1], x1, x2, p_exact, "p_exact")
    subplot_3(axs[2, 2], x1, x2, error_p, "|p_error|")

    plt.tight_layout()
    plt.show()

X_in, X_init, u_exact, v_exact, p_exact, \
Xb_min, Yb1, Tb1, Xb_max, Yb2, Tb2, \
Xb3, Yb_min, Tb3, Xb4, Yb_max, Tb4 = prepare_training_points()
net = Net().double()
num, losss, max_errors_u, l2_errors_u, max_errors_v, l2_errors_v, max_errors_p, l2_errors_p = [], [], [], [], [], [], [], []
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
start_time = time.time()
for i in range(maxtrained_number):
    opt.zero_grad()
    l = loss_total(net, X_in, X_init, u_exact, v_exact, p_exact,
                   Xb_min, Yb1, Tb1, Xb_max, Yb2, Tb2,
                   Xb3, Yb_min, Tb3, Xb4, Yb_max, Tb4)
    l.backward()
    if l <= eL:
        print('解空间的训练停止')
        max_error_u, max_error_v, max_error_p, l2_error_u, l2_error_v, l2_error_p = compute_error(net, X_in)
        print('Epoch ={},Loss ={:.3e}'.format(i + 1, l.item()))
        break
    if (i + 1) % 10 == 0:
        max_error_u, max_error_v, max_error_p, l2_error_u, l2_error_v, l2_error_p = compute_error(net, X_in)
        num.append(i)
        losss.append(l.item())
        max_errors_u.append(max_error_u.item())
        l2_errors_u.append(l2_error_u.item())
        max_errors_v.append(max_error_v.item())
        l2_errors_v.append(l2_error_v.item())
        max_errors_p.append(max_error_p.item())
        l2_errors_p.append(l2_error_p.item())
        if (i + 1) % 100 == 0:
            print('Epoch ={},Loss ={:.2e},max_u ={:.2e}, max_v ={:.2e}, max_p ={:.2e}, l2_u ={:.2e}, l2_v ={:.2e}, l2_p ={:.2e}'.format(i + 1, l.item(), max_error_u, max_error_v, max_error_p, l2_error_u, l2_error_v, l2_error_p))
    opt.step()
end_time = time.time()
print('time={:.2f}s'.format(end_time - start_time))
plot_solutions_2d(net, X_in)
plt.plot(num, np.log10(losss))
plt.xlabel('training frequency')
plt.ylabel('log10(loss)')
plt.show()
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
subplot_2(axs[0, 0], num, np.log10(max_errors_u), "log10(max_errors_u)")
subplot_2(axs[0, 1], num, np.log10(max_errors_v), "log10(max_errors_v)")
subplot_2(axs[0, 2], num, np.log10(max_errors_p), "log10(max_errors_p)")

subplot_2(axs[1, 0], num, np.log10(l2_errors_u), "log10(l2_errors_u)")
subplot_2(axs[1, 1], num, np.log10(l2_errors_v), "log10(l2_errors_v)")
subplot_2(axs[1, 2], num, np.log10(l2_errors_p), "log10(l2_errors_p)")

plt.tight_layout()
plt.show()