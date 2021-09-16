import argparse
import os
import random
import numpy as np
import torch


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_config_path', type=str,
                        default="/app/config.yml")
    parser.add_argument('--local_config_path', type=str,
                        default="~/Documents/rndbaseproject/prequel/cv/habr_optimizers/config.yml")
    parser.add_argument('--server', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gpu-num', type=int, default=0,
                        help='GPU number')
    parser.add_argument('--ix-sched', type=int, default=0,
                        help='Index of first scheduler in experiment')
    parser.add_argument('--schedulers', action='store_true', default=False,
                        help='Whether to loop over schedulers in experiment')
    parser.add_argument('--start-with', type=str, default="MADGRAD",
                        help='First optimizer, all previous will be skipped')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    seed = 2020
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.server:
        args.config_path = args.server_config_path
    else:
        args.config_path = os.path.expanduser(args.local_config_path)

    args.device = torch.device(f"cuda:{args.gpu_num}" if use_cuda else "cpu")

    return args
