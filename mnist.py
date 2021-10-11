import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from box import Box
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torchvision import datasets, transforms
from tqdm import tqdm

from common.benchmark import OptimizersBenchmark, run_benchmark
from common.models import Net
from common.optimizers import optimizers
from common.schedulers import get_schedulers_dict
from common.setup import setup


def train_iter_mnist(bench):
    """Train iteration for experiment on MNIST.

    Args:
        bench (OptimizersBenchmark): Class for benchmarking.

    Returns:
        [List, float]: List of losses, iteration time
    """
    bench.model.train()
    losses = []
    t1 = time()
    for batch_idx, (data, target) in tqdm(enumerate(bench.train_loader)):
        data, target = data.to(bench.device), target.to(bench.device)

        bench.optimizer.zero_grad()
        output = bench.model(data)
        loss = F.nll_loss(output, target)
        loss.backward(create_graph=True)
        bench.optimizer.step()

        if bench.sched_step_note in ['cyclic', 'cosine']:
            bench.scheduler.step()

        if batch_idx % bench.log_interval == 0:
            losses.append(loss.item())
    epoch_time = time() - t1
    return losses, epoch_time


def test_iter_mnist(bench):
    """Test iteration for experiment on MNIST.

    Args:
        bench (OptimizersBenchmark): Class with setup for benchmarking.

    Returns:
        [List, Dict]: List of losses, dictionary with metric values
    """
    bench.model.eval()
    losses = []
    targets = []
    preds = []
    with torch.no_grad():
        for data, target in bench.test_loader:
            targets += list(np.array(target))
            data, target = data.to(bench.device), target.to(bench.device)
            output = bench.model(data)
            loss_val = F.nll_loss(output, target, reduction='sum').item()
            losses.append(loss_val)
            pred = output.argmax(dim=1, keepdim=True)
            preds += list(pred.cpu().numpy().ravel())

    precision = precision_score(
        y_true=targets, y_pred=preds, average="macro", zero_division=0)
    recall = recall_score(y_true=targets, y_pred=preds, average="macro")
    accuracy = accuracy_score(y_true=targets, y_pred=preds)
    f1 = f1_score(y_true=targets, y_pred=preds, average="macro")
    metrics = {"precision": precision,
               "recall": recall, "accuracy": accuracy, "f1": f1}

    print(f'Average loss: {np.mean(losses):.4f},\
    Accuracy: {accuracy:.4f},\
    Precision: {precision:.4f},\
    Recall: {recall:.4f}\n')
    return losses, metrics


def get_dataloader_kwargs(cfg, batch_size):
    return {'num_workers': cfg.data.num_workers,
            'shuffle': cfg.data.shuffle,
            'batch_size': batch_size}


if __name__ == '__main__':
    args = setup()

    # read configuration from file
    with open(args.config_path, "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    cfg.args = args
    cfg.logdir = os.path.join(cfg.path.data_path, cfg.path.logdir)

    # get parameters for dataloaders from config and args
    train_kwargs = get_dataloader_kwargs(cfg, args.batch_size)
    test_kwargs = get_dataloader_kwargs(cfg, cfg.data.test_batch_size)

    # crate transform for data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.data.mnist_mean,), (cfg.data.mnist_std,))
    ])

    # train and test dataset
    dataset1 = datasets.MNIST(cfg.path.data_path, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(cfg.path.data_path, train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    schedulers_dict = get_schedulers_dict(cfg=cfg,
                                          samples_per_epoch=len(train_loader))

    # create class with setup for all experiments
    bench = OptimizersBenchmark(optimizers_dict=optimizers,
                                schedulers_dict=schedulers_dict,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                model_class=Net,
                                cfg=cfg)

    # run all experiments
    run_benchmark(bench, train_iter_mnist, test_iter_mnist, cfg)
