FROM nvcr.io/nvidia/tritonserver:22.12-py3

# # RUN apt-get update && apt-get upgrade -y
# # RUN apt-get install -y libgl1-mesa-dev

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y build-essential gcc ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 python3-dev python3-pip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install torch==2.0.1
RUN pip3 install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip3 install torch torchvision torchaudio
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN pip install mmdet