# this code taken from https://github.com/jettify/pytorch-optimizer and modified

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_optimizer as optim
from hyperopt import fmin, hp, tpe
from madgrad.madgrad import MADGRAD

plt.style.use('seaborn-white')


def rosenbrock(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x ** 2 - A * lib.cos(x * math.pi * 2))
        + (y ** 2 - A * lib.cos(y * math.pi * 2))
    )
    return f


def execute_steps(
    func, initial_state, optimizer_class, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):

        def callback():
            optimizer.zero_grad()
            f = func(x)
            f.backward(create_graph=True, retain_graph=True)
            torch.nn.utils.clip_grad_norm_(x, 1.0)
            return f

        optimizer.step(callback)
        steps[:, i] = x.detach().numpy()
    return steps


def objective_rastrigin(params):
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    initial_state = (-2.0, 3.5)
    minimum = (0, 0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        rastrigin, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_rosenbrok(params):
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    minimum = (1.0, 1.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        rosenbrock, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def plot_rastrigin(grad_iter, optimizer_name, lr):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='x')
    dist = np.linalg.norm(np.array(minimum) -
                          np.array([iter_x[-1], iter_y[-1]]))
    ax.set_title(
        f'Rastrigin func: {optimizer_name} with {len(iter_x)} '
        f'iterations, lr={lr:.6},\nlast point: ({iter_x[-1]:.4}, {iter_y[-1]:.4}), distance to min: {dist:.4}'
    )
    plt.plot(*minimum, 'gD')
    plt.plot(iter_x[-1], iter_y[-1], 'b*')
    plt.savefig('docs/rastrigin_{}.png'.format(optimizer_name))
    return (iter_x[-1], iter_y[-1]), dist


def plot_rosenbrok(grad_iter, optimizer_name, lr):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='x')

    dist = np.linalg.norm(np.array(minimum) -
                          np.array([iter_x[-1], iter_y[-1]]))

    ax.set_title(
        f'Rosenbrock func: {optimizer_name} with {len(iter_x)} '
        f'iterations, lr={lr:.6},\nlast point: ({iter_x[-1]:.4}, {iter_y[-1]:.4}), distance to min: {dist:.4}'
    )
    plt.plot(*minimum, 'gD')
    plt.plot(iter_x[-1], iter_y[-1], 'b*')
    plt.savefig('docs/rosenbrock_{}.png'.format(optimizer_name))
    return (iter_x[-1], iter_y[-1]), dist


def execute_experiments(
    optimizers, objective, func, plot_func, initial_state, seed=1
):
    seed = seed
    last_points = []
    dists = []
    names = []
    lrs = []
    for item in optimizers:
        optimizer_class, lr_low, lr_hi = item
        space = {
            'optimizer_class': hp.choice('optimizer_class', [optimizer_class]),
            'lr': hp.loguniform('lr', lr_low, lr_hi),
        }
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.RandomState(seed),
        )
        print(best['lr'], optimizer_class)

        steps = execute_steps(
            func,
            initial_state,
            optimizer_class,
            {'lr': best['lr']},
            num_iter=500,
        )
        last_point, dist = plot_func(
            steps, optimizer_class.__name__, best['lr'])
        last_points.append(last_point)
        dists.append(dist)
        names.append(optimizer_class.__name__)
        lrs.append(best['lr'])
    np.savez(f"viz_data_{objective.__name__}", names=np.array(names), last_points=np.array(last_points),
             dists=np.array(dists), lrs=np.array(lrs))


def LookaheadYogi(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)


if __name__ == '__main__':
    # python examples/viz_optimizers.py

    # Each optimizer has tweaked search space to produce better plots and
    # help to converge on better lr faster.
    optimizers = [
        (MADGRAD, -8, 0.5),
        # baselines
        (torch.optim.Adam, -8, 0.5),
        (torch.optim.SGD, -8, -1.0),
        (torch.optim.Adadelta, -8, 0.5),
        (torch.optim.AdamW, -8, 0.5),
        (torch.optim.Adamax, -8, 0.5),
        (torch.optim.Adagrad, -8, 0.2),
        (torch.optim.ASGD, -8, 0),
        (torch.optim.LBFGS, -8, 0),
        (torch.optim.RMSprop, -8, -1.6),
        (torch.optim.Rprop, -8, 0.5),
        # Adam based
        (optim.AdaBound, -8, 0.3),
        (optim.Adahessian, -1, 8),
        (optim.AdaMod, -8, 0.2),
        (optim.AdamP, -8, 0.2),
        (optim.DiffGrad, -8, 0.4),
        (optim.Lamb, -8, -2.9),
        (optim.NovoGrad, -8, -1.7),
        (optim.RAdam, -8, 0.5),
        (optim.Yogi, -8, 0.1),
        # SGD/Momentum based
        (optim.AccSGD, -8, -1.4),
        (optim.SGDW, -8, -1.5),
        (optim.SGDP, -8, -1.5),
        (optim.PID, -8, -1.0),
        (optim.QHM, -6, -0.2),
        (optim.QHAdam, -8, 0.1),
        (optim.Ranger, -8, 0.1),
        (optim.RangerQH, -8, 0.1),
        (optim.RangerVA, -8, 0.1),
        (optim.Shampoo, -8, 0.1),
        (LookaheadYogi, -8, 0.1),
        (optim.AggMo, -8, -1.5),
        (optim.SWATS, -8, -1.5),
        (optim.Adafactor, -8, 0.5),
        (optim.A2GradUni, -8, 0.1),
        (optim.A2GradInc, -8, 0.1),
        (optim.A2GradExp, -8, 0.1),
        (optim.AdaBelief, -8, 0.1),
        (optim.Apollo, -8, 0.1),
    ]
    execute_experiments(
        optimizers,
        objective_rastrigin,
        rastrigin,
        plot_rastrigin,
        (-2.0, 3.5),
    )

    execute_experiments(
        optimizers,
        objective_rosenbrok,
        rosenbrock,
        plot_rosenbrok,
        (-2.0, 2.0),
    )
