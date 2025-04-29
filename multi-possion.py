import numpy as onp
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, elu
from scipy.interpolate import griddata

from jax.flatten_util import ravel_pytree
from jax import lax
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import os
import pickle

from scipy.interpolate import griddata
from scipy.fftpack import fft, ifft

from model import u_exact, f_exact, generate_training_res_data, generate_training_bc_data, generate_test_res_data, DataGenerator, MLP, MLP1, MLP_ada0, MLP_ada1, MLP_ada2

sys.path.append('..')
os.makedirs('./data', exist_ok=True)
os.makedirs('./result', exist_ok=True)
print("自适应频率求解Poisson问题")
print("第零次自适应开始：")
'''
solve possion equation
'''
Q_res=10000
Q_bcs=1000
Q_test=1000
net1=[1, 100, 100, 100, 1]
batch_size_bcs=2000
batch_size_res=5000
epoch=100001
lr=1e-02
decay_st=500
decay_ra=0.9


key = random.PRNGKey(1)
key2 = random.PRNGKey(2)
x_train, f_train = generate_training_res_data(key, Q_res)
x_bc_train, f_bc_train = generate_training_bc_data(Q_bcs)
x_test, u_true = generate_test_res_data(key2, Q_test)

# params1 = pickle.load(open(config.save_model_path, 'rb'))
# Initialize model

net_init1, net_apply1 = MLP(net1, activation=np.sin)
params1 = net_init1(rng_key=random.PRNGKey(1432))
net_init2, net_apply2 = MLP(net1, activation=np.sin)
params2 = net_init2(rng_key=random.PRNGKey(1234))
net_init3, net_apply3 = MLP(net1, activation=np.sin)
params3 = net_init3(rng_key=random.PRNGKey(1233))
net_init4, net_apply4 = MLP(net1, activation=np.sin)
params4 = net_init4(rng_key=random.PRNGKey(1334))
net_init5, net_apply5 = MLP(net1, activation=np.sin)
params5 = net_init5(rng_key=random.PRNGKey(1872))
net_init6, net_apply6 = MLP(net1, activation=np.sin)
params6 = net_init6(rng_key=random.PRNGKey(1904))

params_init_ada0 = (params1, params2, params3, params4, params5, params6)
get_params_outside = optimizers.adam(
    optimizers.exponential_decay(lr, decay_steps=decay_st, decay_rate=decay_ra)
)

model_ada0 = MLP_ada0( params_init_ada0, get_params_outside)
physics_dataset = DataGenerator(x_train, f_train, batch_size_res)
boundary_dataset = DataGenerator(x_bc_train, f_bc_train, batch_size_bcs)
model_ada0.train(physics_dataset, boundary_dataset, x_test, u_true, nIter=epoch)

params = model_ada0.get_params(model_ada0.opt_state)
x_plt = np.linspace(0.0, 1.0, 1000)
u_true1 = u_exact(x_plt)

u_pred = model_ada0.predict_u(params, x_plt)
error_ada0 = np.sqrt(np.mean((u_pred - u_true1) ** 2) / np.mean((u_true1 ** 2)))

formatted_number = f"{error_ada0:.2e}"
# 输出带有前缀的结果
print(f"第零次自适应结束，当前相对L2误差为: {formatted_number}")
print("开始提取频率")
y = u_pred
y = onp.array(y)
fft_y = fft(y)
N = 400
x = onp.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = onp.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = onp.angle(fft_y)  # 取复数的角度
# normalization_y = abs_y / N  # 归一化处理（双边频谱）
sum = onp.sum(abs_y)
normalization_y = abs_y/1000   # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）


normalization_half_y = normalization_half_y[1:]
half_x = half_x[1:]
need_fre = half_x[onp.argwhere(normalization_half_y > 0.001)]
d = normalization_half_y[onp.argwhere(normalization_half_y > 0.001)]
d = d * 2

d = onp.squeeze(d)
d_index = onp.argsort(d)
max_fre = d[d_index[-1]]
fre_last = []
norm1 = []
for i in range(d.shape[0]):
    if d[i] > 0.01 * max_fre:
        fre_last.append(need_fre[i])
        norm1.append(d[i])

fre_last = onp.array(fre_last)
norm1 = onp.array(norm1)
w_num = 1
d = norm1
need_fre = fre_last
d_index = np.argsort(d)
fre_ar = need_fre[d_index]
d_ar = d[d_index]

fre_norm = onp.zeros(6,)
fre_norm[0] = onp.sum(d_ar[- w_num:])#/np.sum(d_ar[- 6 * w_num:])
for i in range(4):
    fre_norm[i + 1] = onp.sum(d_ar[-(i + 2) * w_num:-(i + 1) * w_num])#/np.sum(d_ar[- 6 * w_num:])
fre_norm[5] = onp.sum(d_ar[0:-5 * w_num])

need_fre_last = np.array(fre_ar)
fre_norm = np.array(fre_norm)
#### need_fre_last          fre_norm
print("提取频率为", need_fre_last)
print("提取完频率后进行第一次自适应")
print("第一次自适应开始：")
lr=1e-03

fre = need_fre_last
fre_norm = fre_norm
net2 = [3, 100, 100, 100, 1]

net_init1, net_apply1 = MLP1(net2, activation=np.sin)
params1 = net_init1(rng_key=random.PRNGKey(1432))
net_init2, net_apply2 = MLP1(net2, activation=np.sin)
params2 = net_init2(rng_key=random.PRNGKey(1234))
net_init3, net_apply3 = MLP1(net2, activation=np.sin)
params3 = net_init3(rng_key=random.PRNGKey(1233))
net_init4, net_apply4 = MLP1(net2, activation=np.sin)
params4 = net_init4(rng_key=random.PRNGKey(1334))

params_init_ada1 = (params1, params2, params3, params4)
get_params_outside = optimizers.adam(
    optimizers.exponential_decay(lr, decay_steps=decay_st, decay_rate=decay_ra)
)

model_ada1 = MLP_ada1(params_init_ada1, get_params_outside, fre, fre_norm)
model_ada1.train(physics_dataset, boundary_dataset, x_test, u_true, nIter=epoch)

# # Compute test error
# Predict
params = model_ada1.get_params(model_ada1.opt_state)

u_pred = model_ada1.predict_u(params, x_plt)
error_ada1 = np.sqrt(np.mean((u_pred - u_true1) ** 2) / np.mean((u_true1 ** 2)))

formatted_number = f"{error_ada1:.2e}"
# 输出带有前缀的结果
print(f"第一次自适应结束，当前相对L2误差为: {formatted_number}")
print("开始提取频率")
y = u_pred
y = onp.array(y)
fft_y = fft(y)
N = 400
x = onp.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = onp.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = onp.angle(fft_y)  # 取复数的角度
# normalization_y = abs_y / N  # 归一化处理（双边频谱）
sum = onp.sum(abs_y)
normalization_y = abs_y/1000   # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）


normalization_half_y = normalization_half_y[1:]
half_x = half_x[1:]
need_fre = half_x[onp.argwhere(normalization_half_y > 0.001)]
d = normalization_half_y[onp.argwhere(normalization_half_y > 0.001)]
d = d * 2

d = onp.squeeze(d)
d_index = onp.argsort(d)
max_fre = d[d_index[-1]]
fre_last = []
norm1 = []
for i in range(d.shape[0]):
    if d[i] > 0.01 * max_fre:
        fre_last.append(need_fre[i])
        norm1.append(d[i])

fre_last = onp.array(fre_last)
norm1 = onp.array(norm1)
w_num = 1
d = norm1
need_fre = fre_last
d_index = np.argsort(d)
fre_ar = need_fre[d_index]
d_ar = d[d_index]

fre_norm = onp.zeros(6,)
fre_norm[0] = onp.sum(d_ar[- w_num:])#/np.sum(d_ar[- 6 * w_num:])
for i in range(4):
    fre_norm[i + 1] = onp.sum(d_ar[-(i + 2) * w_num:-(i + 1) * w_num])#/np.sum(d_ar[- 6 * w_num:])
fre_norm[5] = onp.sum(d_ar[0:-5 * w_num])

need_fre_last = np.array(fre_ar)
fre_norm = np.array(fre_norm)
#### need_fre_last          fre_norm
print("提取频率为", need_fre_last)
print("提取完频率后进行第二次自适应")
print("第二次自适应开始：")


net_init1, net_apply1 = MLP(net2, activation=np.sin)
params1 = net_init1(rng_key=random.PRNGKey(1432))
net_init2, net_apply2 = MLP(net2, activation=np.sin)
params2 = net_init2(rng_key=random.PRNGKey(1234))
net_init3, net_apply3 = MLP(net2, activation=np.sin)
params3 = net_init3(rng_key=random.PRNGKey(1233))

params_init_ada2 = (params1, params2, params3)
get_params_outside = optimizers.adam(
    optimizers.exponential_decay(lr, decay_steps=decay_st, decay_rate=decay_ra)
)

model_ada2 = MLP_ada2(params_init_ada2, get_params_outside)
model_ada2.train(physics_dataset, boundary_dataset, x_test, u_true, nIter=epoch)

params = model_ada2.get_params(model_ada2.opt_state)
u_pred = model_ada2.predict_u(params, x_plt)

error_ada1 = np.sqrt(np.mean((u_pred - u_true1) ** 2) / np.mean((u_true1 ** 2)))

formatted_number = f"{error_ada1:.2e}"
# 输出带有前缀的结果
print(f"第二次自适应结束，当前相对L2误差为: {formatted_number}")
print("开始提取频率")
y = u_pred
y = onp.array(y)
fft_y = fft(y)
N = 400
x = onp.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = onp.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = onp.angle(fft_y)  # 取复数的角度
# normalization_y = abs_y / N  # 归一化处理（双边频谱）
sum = onp.sum(abs_y)
normalization_y = abs_y/1000   # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）


normalization_half_y = normalization_half_y[1:]
half_x = half_x[1:]
need_fre = half_x[onp.argwhere(normalization_half_y > 0.001)]
d = normalization_half_y[onp.argwhere(normalization_half_y > 0.001)]
d = d * 2

d = onp.squeeze(d)
d_index = onp.argsort(d)
max_fre = d[d_index[-1]]
fre_last = []
norm1 = []
for i in range(d.shape[0]):
    if d[i] > 0.01 * max_fre:
        fre_last.append(need_fre[i])
        norm1.append(d[i])

fre_last = onp.array(fre_last)
norm1 = onp.array(norm1)
w_num = 1
d = norm1
need_fre = fre_last
d_index = np.argsort(d)
fre_ar = need_fre[d_index]
d_ar = d[d_index]

fre_norm = onp.zeros(6,)
fre_norm[0] = onp.sum(d_ar[- w_num:])#/np.sum(d_ar[- 6 * w_num:])
for i in range(4):
    fre_norm[i + 1] = onp.sum(d_ar[-(i + 2) * w_num:-(i + 1) * w_num])#/np.sum(d_ar[- 6 * w_num:])
fre_norm[5] = onp.sum(d_ar[0:-5 * w_num])

need_fre_last = np.array(fre_ar)
fre_norm = np.array(fre_norm)
#### need_fre_last          fre_norm
print("提取频率为", need_fre_last)
print("提取频率与上一次频率提取结果相同，频率自适应停止")
print(f"共自适应频率两次，最终误差为：{formatted_number}" )

