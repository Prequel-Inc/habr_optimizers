import os
import shutil

import numpy as np
import torch
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter

from common.optimizers import opts_without_momentum


def run_benchmark(bench, train_iter, test_iter, cfg):
    """Run experiments on provided optimizers, schedulers,
    dataloaders (in bench class) with provided train and test iteration
    functions.

    Args:
        bench (OptimizersBenchmark): Class with setup for benchmarking.
        train_iter (function): Function for train iteration.
        test_iter (function): Function for test iteration.
        cfg (Box): Configuration.
    """
    n_sched = cfg.train.n_sched
    ix_sched = cfg.args.ix_sched
    bench.train_iter_func = train_iter
    bench.test_iter_func = test_iter
    skip = True
    for opt_name, opt in bench.optimizers_dict.items():
        if opt_name == bench.start_with:
            skip = False
        if skip:
            continue
        try:
            print(opt_name)
            bench.opt = opt
            bench.opt_name = opt_name
            if not bench.has_schedulers_loop:
                trainloop(bench)
            else:
                items = list(bench.schedulers_dict.items())[
                    ix_sched:ix_sched + n_sched]
                print(items)
                for sched_name, sched_data in items:
                    print(sched_name)
                    bench.sched_data = sched_data
                    bench.sched_name = sched_name
                    trainloop(bench)
        except Exception as e:
            print(f"Exception: {str(e)}")


def trainloop(bench):
    """Trainloop for single optimizer.

    Args:
        bench (OptimizersBenchmark): Class with setup for benchmarking.
    """
    bench.model = bench.model_class().to(bench.device)

    if bench.sched_data is not None:
        sched, sched_kwargs, bench.sched_step_note = bench.sched_data

        if bench.sched_step_note == 'cyclic' and bench.opt_name in opts_without_momentum:
            sched_kwargs.update({'cycle_momentum': False})

    if bench.opt_name == "Lookahead":
        inner_opt = optim.Yogi(
            bench.model.parameters(), lr=bench.lr)
        bench.optimizer = bench.opt(inner_opt)
        if bench.sched_data is not None:
            bench.scheduler = sched(inner_opt, **sched_kwargs)
    else:
        bench.optimizer = bench.opt(bench.model.parameters(), lr=bench.lr)
        if bench.sched_data is not None:
            bench.scheduler = sched(bench.optimizer, **sched_kwargs)

    if bench.sched_data is not None:
        exp_name = f"{bench.opt_name}_lr{bench.lr}_bs{bench.bs}_sched_{bench.sched_name}"
    else:
        exp_name = f"{bench.opt_name}_lr{bench.lr}_bs{bench.bs}"

    bench.logger = SummaryWriter(f"{bench.logdir}/{exp_name}")

    train_losses = []
    test_losses = []
    metric_vals = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    times = []
    lrs = []

    for epoch in range(1, bench.epochs + 1):
        print(f'Epoch: {epoch}')
        bench.curr_epoch = epoch
        train_losses_ep, time_ep = bench.train_iter_func(bench)

        train_losses += train_losses_ep
        times.append(time_ep)
        last_lr = bench.optimizer.param_groups[0]['lr']
        lrs.append(last_lr)

        test_losses_ep, metrics = bench.test_iter_func(bench)
        test_losses += test_losses_ep
        for key, _ in metrics.items():
            metric_vals[key].append(metrics[key])

        if bench.sched_step_note == 'plateau':
            bench.scheduler.step(torch.mean(torch.Tensor(test_losses_ep)))
        if bench.sched_step_note == 'step':
            bench.scheduler.step()

    exp_path = os.path.join(bench.exp_data_dir, exp_name)
    np.savez(exp_path, train_loss=train_losses, test_loss=test_losses,
             accuracy=metric_vals['accuracy'], precision=metric_vals['precision'],
             recall=metric_vals['recall'], f1=metric_vals['f1'], time=times, lr=lrs)
    print("Results are saved")


class OptimizersBenchmark():
    """
        Class that contains all data for all experiments. It has current
        state for one experiment and data that should be shared
        between experiments.
    """

    def __init__(self, optimizers_dict, schedulers_dict,
                 train_loader, test_loader, model_class, cfg):
        args = cfg.args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_class = model_class

        self.train_length = len(self.train_loader)

        exp_data_dir = os.path.join(cfg.path.data_path, cfg.path.exp_data_dir)

        if os.path.exists(exp_data_dir) and cfg.path.clean_data_dir:
            shutil.rmtree(exp_data_dir)

        if not os.path.exists(exp_data_dir):
            os.makedirs(exp_data_dir)
        self.exp_data_dir = exp_data_dir
        self.logdir = cfg.path.logdir

        self.lr = args.lr
        self.bs = args.batch_size
        self.gamma = cfg.train.gamma
        self.epochs = cfg.train.epochs
        self.log_interval = cfg.train.log_interval
        self.log_img_freq = cfg.train.log_img_freq
        self.device = args.device
        self.optimizers_dict = optimizers_dict
        self.schedulers_dict = schedulers_dict
        self.sched_step_note = None
        self.start_with = args.start_with
        self.has_schedulers_loop = args.schedulers

        self.opt = None
        self.opt_name = None
        self.sched = None
        self.sched_data = None

        self.train_iter_func = None
        self.test_iter_func = None
