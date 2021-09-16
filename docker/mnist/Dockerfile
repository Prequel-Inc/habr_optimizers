FROM nvcr.io/nvidia/pytorch:20.12-py3
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch_optimizer python-box scikit-learn tensorboard albumentations

WORKDIR /app

ADD common/* /app/common/
ADD config.yml /app/config.yml
ADD mnist.py /app/mnist.py
