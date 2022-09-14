FROM ess-repos01.wisers.com:8889/imgrec/pytorch:1.8.0-cuda10.1-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y \
    vim \
    htop \
    git \
    less \
    tree \
    tmux \
    openssh-server \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/li

RUN pip install flake8 yapf lmdb pillow \
    natsort pytorch-lightning matplotlib \
    wandb jupyter pandas fire tqdm scipy \
    torchvision pytorch_metric_learning faiss-cpu \
    resnest efficientnet_pytorch optuna plotly kaleido \
    deepspeed scikit-image

EXPOSE 6006 8888
