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


save_model_path='./data/ex2.pkl'
save_result_path='./result/'

net1=[1, 100, 100, 100, 1]
net2=[1, 100, 100, 100, 1]

pi = np.pi
coe_100 = 0.1
def u_exact(x):
    return np.sin(2 * np.pi * x) + coe_100 * np.sin(2 * np.pi * 100 * x)

def f_exact(x):
    return - (2 * np.pi) ** 2 * np.sin(2 * np.pi * x) - coe_100 * (100 * 2 * np.pi) ** 2 * np.sin(100 * 2 * np.pi * x)

@jax.jit
def scos(x):
    return 0.5 * np.sin(x) + 0.5 * np.cos(x)

def generate_training_res_data(key, P=1):
    all_sample = random.uniform(key, (P, 1))
    x = all_sample[:, [0]]
    f = f_exact(x)
    return x, f

def generate_training_bc_data(P=1):
    x1 = np.zeros((P, 1))
    x2 = np.ones((P, 1))
    x = np.vstack((x1, x2))
    f = np.zeros_like(x)
    return x, f

def generate_test_res_data(key, P=1):
    all_sample = random.uniform(key, (P, 1))
    x = all_sample[:, [0]]
    f = u_exact(x)
    return x, f

class DataGenerator(data.Dataset):
    def __init__(self, x, f
                 , batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.x = x  # input sample
        self.f = f

        self.N = x.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        x = self.x[idx, :]
        f = self.f[idx, :]
        # Construct batch
        inputs = x
        outputs = f
        return inputs, outputs

# Define the neural net
def MLP(layers, activation=relu):
    ''' Vanilla MLP'''

    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        W, b = params[0]
        outputs = np.dot(inputs, W) + b
        inputs = activation(outputs)
        for W, b in params[1:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply


class MLP_ada0:
    def __init__(self, params_outside, get_params_outside):

        # Network initialization and evaluation functions
        self.layers_init1, self.layers_apply1 = MLP(net1, activation=scos)
        self.layers_init2, self.layers_apply2 = MLP(net1, activation=scos)
        self.layers_init3, self.layers_apply3 = MLP(net1, activation=scos)
        self.layers_init4, self.layers_apply4 = MLP(net1, activation=scos)
        self.layers_init5, self.layers_apply5 = MLP(net1, activation=scos)
        self.layers_init6, self.layers_apply6 = MLP(net1, activation=scos)

        params = params_outside

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = get_params_outside
        self.opt_state = self.opt_init(params)
        _, self.unravel_params = ravel_pytree(params)
        self.itercount = itertools.count()
        self.current_count = 0

        # Loggers
        self.loss_log = []
        self.loss_physics_log = []
        self.loss_bcs_log = []
        self.error = []

    # Define Net architecture
    def MLP_net(self, params, x):
        params1, params2, params3, params4, params5, params6 = params
        outputs = (1 * self.layers_apply1(params1, x * 1) + 2 * self.layers_apply2(params2, x * 2) \
                   + 4 * self.layers_apply3(params3, x * 4) + 8 * self.layers_apply4(params4, x * 8) \
                   + 16 * self.layers_apply5(params5, x * 16) + 32 * self.layers_apply6(params6, x * 32))
        return outputs.reshape(())

    # Define ODE/PDE residual
    def residual_net(self, params, x, y):
        u_xx = grad(grad(self.MLP_net, argnums=1), argnums=1)(params, x)
        return u_xx - y

    def compute_K(self, params, x, y):
        K = grad(self.residual_net, argnums=0)(params, x, y)
        K, _ = ravel_pytree(K)
        K = np.dot(K, K)
        return K

    def compute_error(self, params, x_test, u_true):
        u_test = vmap(self.MLP_net, (None, 0))(params, x_test[:, 0])
        u_test = u_test.reshape(-1, 1)
        error = np.sqrt(
            np.mean((u_test - u_true) ** 2) / np.mean((u_true ** 2)))
        return error

    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        x = inputs
        # Compute forward pass
        f_pred = vmap(self.MLP_net, (None, 0))(params, x[:, 0])
        # Compute loss
        loss = 100000 * np.mean((f_pred.flatten()) ** 2)
        return loss

    # Define physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        x = inputs
        f = outputs
        # Compute forward pass
        u = vmap(self.residual_net, (None, 0, 0))(params, x[:, 0], f[:, 0])
        # Compute loss
        loss = np.mean((u.flatten()) ** 2)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss(self, params, physics_batch, boundary_batch):
        res_inputs, res_outputs = physics_batch
        bcs_inputs, bcs_outputs = boundary_batch
        x_res = res_inputs
        f_res = res_outputs

        x_bcs = bcs_inputs
        # Compute forward

        u_res = vmap(self.residual_net, (None, 0, 0))(params, x_res[:, 0], f_res[:, 0])
        # Compute loss
        u_bcs = vmap(self.MLP_net, (None, 0))(params, x_bcs[:, 0])
        loss_physics = np.mean((u_res.flatten()) ** 2)
        loss_bc = np.mean((u_bcs.flatten()) ** 2)
        loss = loss_physics + 100000 * loss_bc
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, physics_batch, boundary_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, physics_batch, boundary_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, physics_dataset, boundary_dataset, x_test, u_true, nIter=10000):
        # Define the data iterator
        physics_data = iter(physics_dataset)
        boundary_data = iter(boundary_dataset)
        # physics_batch = next(physics_data)

        pbar = trange(nIter)
        physics_batch = next(physics_data)

        # Main training loop
        for it in pbar:
            boundary_batch = next(boundary_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, physics_batch, boundary_batch)
            if it % 1000 == 0:
                key1 = random.PRNGKey(it)
                all_sample = random.uniform(key1, (5000, 1))
                x = all_sample[:, [0]]
                f = f_exact(x)
                physics_batch = (x, f)

            if it % 2000 == 0:
                params = self.get_params(self.opt_state)
                # Compute losses
                loss_value = self.loss(params, physics_batch, boundary_batch)
                loss_phy = self.loss_physics(params, physics_batch)
                loss_bc = self.loss_bcs(params, boundary_batch)

                err = self.compute_error(params, x_test, u_true)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_physics_log.append(loss_phy)
                self.loss_bcs_log.append(loss_bc)
                self.error.append(err)
                # Print losses during training
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_phy': loss_phy,
                                  'loss_bc': loss_bc,
                                  'error': err})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, x):
        u = vmap(self.MLP_net, (None, 0))(params, x)
        return u


# Define the resnet
def MLP1(layers, activation=relu):
    ''' Vanilla MLP'''

    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def input_encoding(x, fre_x):
        out = np.hstack([2 * np.pi * fre_x * x,
                        np.sin(2 * np.pi * fre_x * x), np.cos(2 * np.pi * fre_x * x)])

        return out


    def apply(params, inputs, fre_x):
        #x = inputs[0]
        inputs = input_encoding(inputs, fre_x)

        W, b = params[0]
        outputs = np.dot(inputs, W) + b
        inputs = activation(outputs)
        for W, b in params[1:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply

class MLP_ada1:
    def __init__(self, params_outside, get_params_outside, fre, fre_norm):

        # Network initialization and evaluation functions
        self.fre_norm = fre_norm
        self.fre = fre

        self.layers_init1, self.layers_apply1 = MLP1(net2, activation=scos)
        self.layers_init2, self.layers_apply2 = MLP1(net2, activation=scos)
        self.layers_init3, self.layers_apply3 = MLP1(net2, activation=scos)
        self.layers_init4, self.layers_apply4 = MLP1(net2, activation=scos)

        params = params_outside

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = get_params_outside
        self.opt_state = self.opt_init(params)
        _, self.unravel_params = ravel_pytree(params)
        self.itercount = itertools.count()
        self.current_count = 0

        # Loggers
        self.loss_log = []
        self.loss_physics_log = []
        self.loss_bcs_log = []
        self.error = []
    # Define Net architecture
    def MLP_net(self, params, x):
        params1, params2, params3, params4 = params
        inputs = np.stack([x])
        outputs = (self.fre_norm[0] * self.layers_apply1(params1, inputs, self.fre[- 1:, 0]) + self.fre_norm[1] * self.layers_apply2(params2, inputs, self.fre[- 1 * 2:- 1* 1, 0])\
                   + self.fre_norm[2] * self.layers_apply3(params3, inputs, self.fre[- 1 * 3:- 1*2, 0]) + self.fre_norm[3] * self.layers_apply4(params4, inputs, self.fre[- 1 * 4:- 1* 3, 0]))
        return outputs.reshape(())

    # Define ODE/PDE residual
    def residual_net(self, params, x, y):
        # u_x = grad(self.MLP_net, argnums=1)(params, x)
        u_xx = grad(grad(self.MLP_net, argnums=1), argnums=1)(params, x)
        return u_xx - y

    def compute_K(self, params, x, y):
        K = grad(self.residual_net, argnums=0)(params, x, y)
        K, _ = ravel_pytree(K)
        K = np.dot(K, K)
        return K

    def compute_error(self, params, x_test, u_true):
        u_test = vmap(self.MLP_net, (None, 0))(params, x_test[:, 0])
        u_test = u_test.reshape(-1, 1)
        error = np.sqrt(
            np.mean((u_test - u_true) ** 2) / np.mean((u_true ** 2)))
        return error

    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        x = inputs
        # Compute forward pass
        f_pred = vmap(self.MLP_net, (None, 0))(params, x[:, 0])
        # Compute loss
        loss = 100000 * np.mean((f_pred.flatten()) ** 2)
        return loss

    # Define physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        x = inputs
        f = outputs
        # Compute forward pass
        u = vmap(self.residual_net, (None, 0, 0))(params, x[:, 0], f[:, 0])
        # Compute loss
        loss = np.mean((u.flatten()) ** 2)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss(self, params, physics_batch, boundary_batch):
        res_inputs, res_outputs = physics_batch
        bcs_inputs, bcs_outputs = boundary_batch
        x_res = res_inputs
        f_res = res_outputs

        x_bcs = bcs_inputs
        # Compute forward

        u_res = vmap(self.residual_net, (None, 0, 0))(params, x_res[:, 0], f_res[:, 0])
        # Compute loss
        u_bcs = vmap(self.MLP_net, (None, 0))(params, x_bcs[:, 0])
        loss_physics = np.mean((u_res.flatten()) ** 2)
        loss_bc = np.mean((u_bcs.flatten()) ** 2)
        loss = loss_physics + 100000 * loss_bc
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, physics_batch, boundary_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, physics_batch, boundary_batch)
        return self.opt_update(i, g, opt_state)

        # Optimize parameters in a loop
    def train(self, physics_dataset, boundary_dataset, x_test, u_true, nIter=10000):
        # Define the data iterator
        physics_data = iter(physics_dataset)
        boundary_data = iter(boundary_dataset)
        physics_batch = next(physics_data)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            boundary_batch = next(boundary_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, physics_batch, boundary_batch)
            if it % 1000 ==0 :
                key1 = random.PRNGKey(it)
                all_sample = random.uniform(key1, (5000, 1))
                x = all_sample[:, [0]]
                f = f_exact(x)
                physics_batch = (x, f)

            if it % 2000 == 0:
                params = self.get_params(self.opt_state)
                # Compute losses
                loss_value = self.loss(params, physics_batch, boundary_batch)
                loss_phy = self.loss_physics(params, physics_batch)
                loss_bc = self.loss_bcs(params, boundary_batch)

                err = self.compute_error(params, x_test, u_true)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_physics_log.append(loss_phy)
                self.loss_bcs_log.append(loss_bc)
                self.error.append(err)
                # Print losses during training
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_phy': loss_phy,
                                  'loss_bc': loss_bc,
                                  'error': err})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, x):
        u = vmap(self.MLP_net, (None, 0))(params, x)
        return u


class MLP_ada2:
    def __init__(self, params_outside, get_params_outside):

        # Network initialization and evaluation functions

        self.layers_init1, self.layers_apply1 = MLP(net2, activation=scos)
        self.layers_init2, self.layers_apply2 = MLP(net2, activation=scos)
        self.layers_init3, self.layers_apply3 = MLP(net2, activation=scos)

        params = params_outside

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = get_params_outside
        self.opt_state = self.opt_init(params)
        _, self.unravel_params = ravel_pytree(params)
        self.itercount = itertools.count()
        self.current_count = 0

        # Loggers
        self.loss_log = []
        self.loss_physics_log = []
        self.loss_bcs_log = []
        self.error = []
    # Define Net architecture
    def MLP_net(self, params, x):
        params1, params2, params3 = params
        input1 = np.stack([1 * 2 * np.pi * x, np.sin(1 * 2 * np.pi * x), np.cos(1 * 2 * np.pi * x)])
        input2 = np.stack([100 * 2 * np.pi * x, np.sin(100 * 2 * np.pi * x), np.cos(100 * 2 * np.pi * x)])
        input3 = np.stack([101 * 2 * np.pi * x, np.sin(101 * 2 * np.pi * x), np.cos(101 * 2 * np.pi * x)])

        outputs = (0.99889886 * self.layers_apply1(params1, input1) + 0.09828694 * self.layers_apply2(params2, input2) \
                   + 0.0110136 * self.layers_apply3(params3, input3))
        return outputs.reshape(())

    # Define ODE/PDE residual
    def residual_net(self, params, x, y):
        # u_x = grad(self.MLP_net, argnums=1)(params, x)
        u_xx = grad(grad(self.MLP_net, argnums=1), argnums=1)(params, x)
        return u_xx - y

    def compute_K(self, params, x, y):
        K = grad(self.residual_net, argnums=0)(params, x, y)
        K, _ = ravel_pytree(K)
        K = np.dot(K, K)
        return K

    def compute_error(self, params, x_test, u_true):
        u_test = vmap(self.MLP_net, (None, 0))(params, x_test[:, 0])
        u_test = u_test.reshape(-1, 1)
        error = np.sqrt(
            np.mean((u_test - u_true) ** 2) / np.mean((u_true ** 2)))
        return error

    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        x = inputs
        # Compute forward pass
        f_pred = vmap(self.MLP_net, (None, 0))(params, x[:, 0])
        # Compute loss
        loss = 100000 * np.mean((f_pred.flatten()) ** 2)
        return loss

    # Define physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        x = inputs
        f = outputs
        # Compute forward pass
        u = vmap(self.residual_net, (None, 0, 0))(params, x[:, 0], f[:, 0])
        # Compute loss
        loss = np.mean((u.flatten()) ** 2)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss(self, params, physics_batch, boundary_batch):
        res_inputs, res_outputs = physics_batch
        bcs_inputs, bcs_outputs = boundary_batch
        x_res = res_inputs
        f_res = res_outputs

        x_bcs = bcs_inputs
        # Compute forward

        u_res = vmap(self.residual_net, (None, 0, 0))(params, x_res[:, 0], f_res[:, 0])
        # Compute loss
        u_bcs = vmap(self.MLP_net, (None, 0))(params, x_bcs[:, 0])
        loss_physics = np.mean((u_res.flatten()) ** 2)
        loss_bc = np.mean((u_bcs.flatten()) ** 2)
        loss = loss_physics + 100000 * loss_bc
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, physics_batch, boundary_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, physics_batch, boundary_batch)
        return self.opt_update(i, g, opt_state)

        # Optimize parameters in a loop
    def train(self, physics_dataset, boundary_dataset, x_test, u_true, nIter=10000):
        # Define the data iterator
        physics_data = iter(physics_dataset)
        boundary_data = iter(boundary_dataset)
        physics_batch = next(physics_data)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            boundary_batch = next(boundary_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, physics_batch, boundary_batch)
            if it % 1000 ==0 :
                key1 = random.PRNGKey(it)
                all_sample = random.uniform(key1, (5000, 1))
                x = all_sample[:, [0]]
                f = f_exact(x)
                physics_batch = (x, f)

            if it % 2000 == 0:
                params = self.get_params(self.opt_state)
                # Compute losses
                loss_value = self.loss(params, physics_batch, boundary_batch)
                loss_phy = self.loss_physics(params, physics_batch)
                loss_bc = self.loss_bcs(params, boundary_batch)

                err = self.compute_error(params, x_test, u_true)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_physics_log.append(loss_phy)
                self.loss_bcs_log.append(loss_bc)
                self.error.append(err)
                # Print losses during training
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_phy': loss_phy,
                                  'loss_bc': loss_bc,
                                  'error': err})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, x):
        u = vmap(self.MLP_net, (None, 0))(params, x)
        return u
