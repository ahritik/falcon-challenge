FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Get falcon-challenge
RUN git clone https://github.com/snel-repo/falcon-challenge.git /opt/falcon-challenge
RUN pip3 install -e /opt/falcon-challenge

# Core deps
RUN pip3 install numpy tqdm tensorboard tensorboardX einops optax

# JAX CUDA wheels
# Note: EvalAI GPUs may differ; for CPU fallback replace with: pip install jax jaxlib
RUN pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Mandatory EvalAI paths
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH "/tmp/ground_truth.pkl"
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

# Defaults for remote run
ENV EVALUATION_LOC remote
ENV SPLIT "h2"
ENV PHASE "test"
ENV BATCH_SIZE 1

# Add local decoder entrypoint and any training artifacts if needed
ADD ./decoder_demos/ decoder_demos/
ADD ./data_demos/ data_demos/

# If you have a pre-trained checkpoint locally, add it here:
# ADD ./local_data/hritik_h2/model.arrays.npz /workspace/local_data/hritik_h2/model.arrays.npz

# Run
CMD ["/bin/bash", "-c", "python3 decoder_demos/hritik_sample.py eval --evaluation $EVALUATION_LOC --model-path /workspace/local_data/hritik_h2/model --phase $PHASE --split $SPLIT"]
