# habr_optimizers
This repository contains code for article on ![Habr](https://habr.com).

Experiments are conducted on two tasks (code for multi-label classification is not published):
* toy task: classification on MNIST dataset
* real task: multi-label classification of images from mobile phone

For each task there are three types of experiments:
* changing learning rate, fixed batch size, no learning rate scheduler
* changing learning rate, changing batch size, no learning rate scheduler
* fixed learning rate, fixed batch size, changing learning rate schedulers

### Requires:
1) clone MADGRAD optimizer in common folder, ![actual commit](https://github.com/facebookresearch/madgrad/commit/71da2a48e738b76a29d3e33c3cadb4775cdf9e91):

### Build MNIST experiment:
`docker build . -f docker/mnist/Dockerfile -t "habr_mnist"`

### Run MNIST experiment:
`docker run --gpus all -it --cpuset-cpus 0-9 -v /path/to/data:/app/data --ipc=host "habr_mnist" python3 mnist.py --lr 0.001 --batch-size 8`


### Parameters (located in common/setup.py):
* config_path Path to config.yml
* batch-size Batch size
* lr Learning rate
* gpu-num Number of GPU
* ix-sched To execute experiments with different schedulers in parallel, they are
         splitted into groups. Group size can be changed in config.yml (n_sched).
         This is ordinal number of scheduler in list of schedulers (common/schedulers.py)
         to start execution with this scheduler. Used only if --schedulers is True.
* schedulers Whether to compare different schedulers in experiment.
* start-with Name of optimizer to start experiment with (names list are in
         common/optimizers.py)

### Repository structure:
* common
  * benchmark.py Contains trainloop and class for data that is shared between different models in one run.
  * models.py Model for MNIST.
  * optimizers.py List of optimizers.
  * schedulers.py List of schedulers and their parameters.
  * setup.py CLI for both types of experiments and some common preparation.
* config.yml All parameters that are not changed during different experiments.
* mnist.py Entrypoint for MNIST experiment, code for train and test iteration on MNIST.

