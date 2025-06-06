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
maxtrained_number = 5000
N_x = 20
N_y = 20
N_bc = 40
X_min = 0.0
X_max = 2.0
Y_min = 0.0
Y_max = 2.0
element_number = 250
net_size = [2, 50, element_number, 1]

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
    def __init__(self, size=net_size, a=[2 / (X_max - X_min),2 / (X_max - X_min)], b=[-1 - 2 * X_min / (X_max - X_min),-1 - 2 * X_min / (X_max - X_min)]):
        super(Net, self).__init__()
        self.normalize = NormalizeLayer(a, b)
        layers = []
        for i in range(len(size) - 1):
            if i < len(size) - 2:
                layers.append(nn.Linear(size[i], size[i + 1]).double())
                layers.append(nn.Tanh())
            else:
                # 对于最后一个线性层,关闭偏置
                layers.append(nn.Linear(size[i], size[i + 1], bias=False).double())
        self.net = nn.Sequential(*layers)


    def forward(self, x, return_hidden_layer=False):
        x_normalized = self.normalize(x)
        last_activation_output = None  # 用于保存最后一个激活层的输出

        for layer in self.net:
            x_normalized = layer(x_normalized)

            # 如果当前层是激活层，则更新last_activation_output
            if isinstance(layer, nn.Tanh):
                last_activation_output = x_normalized

        # 如果需要返回最后一个隐藏层的输出（即最后一个激活层的输出）
        if return_hidden_layer:
            return last_activation_output

        return x_normalized

    def get_hidden_layer_output(self, x):
        return self.forward(x, return_hidden_layer=True)

loss = nn.MSELoss()
def interior():
    # 生成 x 和 t 的空间
    x_line = torch.linspace(X_min, X_max, N_x)
    y_line = torch.linspace(Y_min, Y_max, N_y)
    # 生成网格
    x_grid, y_grid = torch.meshgrid(x_line, y_line, indexing='ij')
    x_flat = x_grid.reshape(-1, 1).double()
    y_flat = y_grid.reshape(-1, 1).double()
    x_flat.requires_grad_(True)
    y_flat.requires_grad_(True)
    return x_flat, y_flat
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
def loss_interior(u):
    x_interior, y_interior = interior()
    u_in = u(torch.cat([x_interior, y_interior], dim=1))
    term_interior = F(x_interior, y_interior)
    return loss(- gradients(u_in, x_interior, 2) - gradients(u_in, y_interior, 2), term_interior)
def loss_left(u):
    x_left = X_min * torch.ones(N_y, 1).double().requires_grad_(True)
    y_left = torch.linspace(Y_min, Y_max, N_y).unsqueeze(1).double().requires_grad_(True)
    term_left = exact(x_left, y_left)
    u_bc = u(torch.cat([x_left, y_left], dim=1))
    return loss(u_bc, term_left)
def loss_right(u):
    x_right = X_max * torch.ones(N_y, 1).double().requires_grad_(True)
    y_right = torch.linspace(Y_min, Y_max, N_y).unsqueeze(1).double().requires_grad_(True)
    term_right = exact(x_right, y_right)
    u_bc = u(torch.cat([x_right, y_right], dim=1))
    return loss(u_bc, term_right)
def loss_down(u):
    x_down = torch.linspace(X_min, X_max, N_bc).unsqueeze(1).double().requires_grad_(True)
    y_down = Y_min * torch.ones(N_bc, 1).double().requires_grad_(True)
    term_down = exact(x_down, y_down)
    u_bc = u(torch.cat([x_down, y_down], dim=1))
    return loss(u_bc, term_down)
def loss_up(u):
    x_up = torch.linspace(X_min, X_max, N_bc).unsqueeze(1).double().requires_grad_(True)
    y_up = Y_max * torch.ones(N_bc, 1).double().requires_grad_(True)
    term_up = exact(x_up, y_up)
    u_bc = u(torch.cat([x_up, y_up], dim=1))
    return loss(u_bc, term_up)
def exact(x, y):
    exact = torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
    return exact
def F(x, y):
    F = 2 * torch.pi * torch.pi * torch.sin(torch.pi*x) * torch.sin(torch.pi*y)
    return F

def assemble_matrix(model, point):
    # 前向传播和梯度计算
    out = model.get_hidden_layer_output(point)
    du2_dx2 = []
    du2_dy2 = []
    for i in range(element_number):
        g1 = torch.autograd.grad(outputs=out[:, i], inputs=point,
                                 grad_outputs=torch.ones_like(out[:, i]),
                                 create_graph=True, retain_graph=True)[0]
        gxx = torch.autograd.grad(outputs=g1[:, 0], inputs=point,
                                 grad_outputs=torch.ones_like(out[:, i]),
                                 create_graph=True, retain_graph=True)[0]
        gyy = torch.autograd.grad(outputs=g1[:, 1], inputs=point,
                                  grad_outputs=torch.ones_like(out[:, i]),
                                  create_graph=True, retain_graph=True)[0]
        du2_dx2.append(gxx[:, 0].squeeze().detach().numpy())
        du2_dy2.append(gyy[:, 1].squeeze().detach().numpy())
    # 组装 PDE 条件的矩阵 A_I
    A_I = -np.array(du2_dx2).T - np.array(du2_dy2).T
    F_I = np.zeros([N_x * N_y, 1])
    # 组装 PDE 条件的向量 f_I
    for i in range(N_x * N_y):
        F_I[i] = F(point[i, 0], point[i, 1]).detach().numpy()

    points_down_x = torch.linspace(X_min, X_max, N_x).requires_grad_(True)
    points_down_y = Y_min * torch.ones_like(points_down_x)
    points_down = torch.cat([points_down_x.reshape(-1, 1), points_down_y.reshape(-1, 1)], dim=1)
    A_down = model.get_hidden_layer_output(points_down).detach().numpy()
    F_down = exact(points_down_x, points_down_y).detach().numpy().reshape(-1, 1)

    points_up_x = points_down_x
    points_up_y = Y_max * torch.ones_like(points_up_x)
    points_up = torch.cat([points_up_x.reshape(-1, 1), points_up_y.reshape(-1, 1)], dim=1)
    A_up = model.get_hidden_layer_output(points_up).detach().numpy()
    F_up = exact(points_up_x, points_up_y).detach().numpy().reshape(-1, 1)

    points_left_y = torch.linspace(Y_min, Y_max, N_y).unsqueeze(1).double().requires_grad_(True)
    points_left_x = X_min * torch.ones_like(points_left_y)
    points_left = torch.cat([points_left_x, points_left_y], dim=1)
    A_left = model.get_hidden_layer_output(points_left).detach().numpy()
    F_left = exact(points_left_x, points_left_y).detach().numpy().reshape(-1, 1)

    points_right_y = points_left_y
    points_right_x = X_max * torch.ones_like(points_right_y)
    points_right = torch.cat([points_right_x, points_right_y], dim=1)
    A_right = model.get_hidden_layer_output(points_right).detach().numpy()
    F_right = exact(points_right_x, points_right_y).detach().numpy().reshape(-1, 1)

    A = np.vstack((A_I, A_down, A_up, A_left, A_right))
    f = np.vstack((F_I, F_down, F_up, F_left, F_right))
    w = lstsq(A, f)[0]
    return w
def compute_error(net, x):
    # 网络预测
    u_pred = net(x).detach().numpy().flatten()
    # 精确解
    u_exact = exact(x[:, 0],x[:, 1]).detach().numpy().flatten()
    # 计算误差
    max_error = np.max(np.abs(u_pred - u_exact))
    relative_l2_error = np.sqrt(np.sum((u_pred - u_exact) ** 2)/np.sum(u_exact ** 2))
    return max_error, relative_l2_error
def plot_solutions_2d(net, x):
    # 计算预测解和精确解
    u_pred = net(x).detach().numpy().flatten()
    u_exact = exact(x[:, 0], x[:, 1]).detach().numpy().flatten()
    error = np.abs(u_pred - u_exact)

    # 获取x的两个分量
    x1 = x[:, 0].detach().numpy()
    x2 = x[:, 1].detach().numpy()

    # 创建一个 1x3 的图像布局
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # 绘制预测解
    contour = axs[0].tricontourf(x1, x2, u_pred, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axs[0])
    axs[0].set_title("Predictive Solution")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")

    # 绘制精确解
    contour = axs[1].tricontourf(x1, x2, u_exact, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axs[1])
    axs[1].set_title("Exact Solution")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")

    # 绘制误差图
    contour = axs[2].tricontourf(x1, x2, error, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axs[2])
    axs[2].set_title("Absolute Error")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("t")
    plt.show()

points_interior_x, points_interior_y = interior()
points_interior = torch.cat([points_interior_x, points_interior_y], dim=1)
u = Net().double()
opt = torch.optim.Adam(params=[param for name, param in u.named_parameters()])
num, losss, max_errors, l2_errors = [], [], [], []
start_time1 = time.time()
for i in range(maxtrained_number):
    opt.zero_grad()
    l = loss_interior(u) + (loss_left(u) + loss_right(u) + loss_down(u) + loss_up(u))/4
    l.backward()
    if (i + 1) % 100 == 0:
        max_error, l2_error = compute_error(u, points_interior)
        num.append(i)
        losss.append(l.item())
        max_errors.append(max_error.item())
        l2_errors.append(l2_error.item())
        print('trained_number ={},loss ={:.3e},L_infty error ={:.3e},relative_L_2 error ={:.3e}'.format(i + 1, l.item(), max_error.item(),l2_error.item()))
    opt.step()
end_time1 = time.time()
plot_solutions_2d(u, points_interior)
print('PINN_time={:.2f}s'.format(end_time1 - start_time1))
print('ELM训练开始')
start_time2 = time.time()
w = assemble_matrix(u, points_interior)
end_time2 = time.time()
#u.net[2*(len(net_size) - 2)].weight = nn.Parameter(torch.from_numpy(w).double().view(1, -1))
u.net[-1].weight.data = torch.from_numpy(w.reshape(1, -1)).double()
# 计算loss
l = loss_interior(u) + (loss_left(u) + loss_right(u) + loss_down(u) + loss_up(u))/4
max_error , l2_error = compute_error(u, points_interior)
print('ELM训练结束')
print('loss ={:.3e},L_infty error ={:.3e},L_2 error ={:.3e}'.format(l.item(), max_error.item(),l2_error.item()))
print('ELMtime={:.2f}s'.format(end_time2 - start_time2))
print('totaltime={:.2f}s'.format(end_time2 - start_time1))
# 画图
plot_solutions_2d(u, points_interior)
num.append(maxtrained_number+10)
losss.append(l.item())
max_errors.append(max_error.item())
l2_errors.append(l2_error.item())
plt.plot(num, np.log10(losss))
plt.xlabel('training frequency')
plt.ylabel('log10(loss)')
plt.show()
plt.plot(num, np.log10(max_errors))
plt.xlabel('training frequency')
plt.ylabel('log10(L_infty_error)')
plt.show()
plt.plot(num, np.log10(l2_errors))
plt.xlabel('training frequency')
plt.ylabel('log10(l2_errors)')
plt.show()