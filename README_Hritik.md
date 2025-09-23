# FALCON H2 Model Development (Hritik)

This package contains core code for developing and submitting decoders specifically for the H2 dataset in the FALCON challenge. H2 is a human intracortical brain-computer interface (iBCI) dataset. For a more general overview of FALCON, please see the [main website](https://snel-repo.github.io/falcon/).

**Note:** This README is customized for H2 model development. The setup focuses exclusively on the H2 dataset and the "hritik" decoder (JAX-based, session-aware encoders + Perceiver-like decoder with FiLM/LoRA options and MMD alignment).

## Installation

Install `falcon_challenge` with:

```powershell
pip install falcon-challenge
```

To create Docker containers for submission, you must have Docker installed.
See, e.g. [https://docs.docker.com/desktop/install/linux-install/](https://docs.docker.com/desktop/install/linux-install/).

## Getting started

### Data downloading

The H2 dataset is available on DANDI ([H2](https://dandiarchive.org/dandiset/000950?search=falcon&pos=4)). H2 is a human intracortical brain-computer interface (iBCI) dataset. You can download it by going to its DANDI page to find the DANDI download command, or you can run the following from project root:

```powershell
dandi download https://dandiarchive.org/dandiset/000950/draft
```

Data from the H2 dataset is broken down as follows:

- Held-in
  - Data from the first several recording sessions.
  - All non-evaluation data is released and split into calibration (large portion) and minival (small portion) sets.
  - Held-in calibration data is intended to train decoders from scratch.
  - Minival data enables validation of held-in decoders and submission debugging.
- Held-out:
  - Data from the latter several recording sessions.
  - A small portion of non-evaluation data is released for calibration.
  - Held-out calibration data is intentionally small to discourage training decoders from scratch on this data and provides an opportunity for few-shot recalibration.

The sample code expects your data directory to be set up in `./data`. Specifically, the following hierarchy is expected for H2:

```
data
- h2
  - held_in_calib
  - held_out_calib
  - minival (Copy dandiset minival folder into this folder)
```

Each of the lowest level dirs holds the data files (in Neurodata Without Borders (NWB) format). Data from some sessions is distributed across multiple NWB files. Some data from each file is allocated to calibration, minival, and evaluation splits as appropriate.

### Code

This codebase contains starter code for implementing your own method for the FALCON challenge, with a focus on H2 model development.

- The `falcon_challenge` folder contains the logic for the evaluator. Submitted solutions must conform to the interface specified in `falcon_challenge.interface`. During `reset`, `predict`, and `observe` methods, your approach has access to a new timestep of neural observations. To access and make use of trial timing signals, implement the `on_done` method. Only within-trial data will be considered for evaluation, but you are welcome to use data from the entire available time period.
- In `data_demos`, we provide notebooks that survey each dataset released as part of this challenge. The `h2.ipynb` notebook is particularly relevant for understanding the H2 dataset.
- In `decoder_demos`, we provide sample decoders and baselines that are formatted to be ready for submission to the challenge. To use them, see the comments in the header of each file ending in `_sample.py`. Your solutions should look similar once implemented! (Namely, you should have a `_decoder.py` file or class which conforms to `falcon_challenge.interface` as well as a `_sample.py` file that is the entry point for running your decoder.)

For example, you can prepare and evaluate a linear decoder for H2 by running:

```powershell
python decoder_demos/sklearn_decoder.py --training_dir data/h2/held_in_calib/ --calibration_dir data/h2/held_out_calib/ --mode all --task h2
# Should report: CV fit score

python decoder_demos/sklearn_sample.py --evaluation local --phase minival --split h2
# Should report: Held In Mean
```

Note: During evaluation, data file names are hashed into unique tags. Submitted solutions receive data to decode along with tags indicating the file from which the data originates in the call to their `reset` function. These tags are the keys of the the `DATASET_HELDINOUT_MAP` dictionary in `falcon_challenge/evaluator.py`. Submissions that intend to condition decoding on the data file from which the data comes should make use of these tags. For an example, see `fit_many_decoders` and `reset` in `decoder_demos/sklearn_decoder.py`.

## Your workflow with the Hritik decoder

We'll follow four steps:
1) EDA & Data Preprocessing
2) Model Training
3) Local Eval
4) EvalAI Submission

All commands below assume Windows PowerShell and project root at `c:\works\falcon-challenge`.

### 1) EDA & Data Preprocessing

- Explore H2 sessions and optionally save trial caches for faster training.
- Uses tqdm progress bars and writes summaries to `local_data/h2_preproc`.

```powershell
python data_demos/h2_eda_preprocessing.py --root data/h2 --out-dir local_data/h2_preproc --save-caches
```

Artifacts:

- `local_data/h2_preproc/h2_sessions.csv` summary
- `local_data/h2_preproc/trial_caches/*.npz` optional per-trial caches

### 2) Model Training (JAX)

- Hritik model: session-specific encoders, Perceiver-based decoder, FiLM conditioning, optional LoRA, and MMD loss for latent alignment. Training logs to TensorBoard and prints tqdm bars.

Install runtime deps if needed:

```powershell
pip install numpy tqdm tensorboard tensorboardX optax jax jaxlib
```

Train on held-in calibration data and save checkpoint to `local_data/hritik_h2`:

```powershell
python decoder_demos/hritik_sample.py train --training_dir data/h2/held_in_calib --save-path local_data/hritik_h2 --epochs 3 --batch-size 8 --lr 1e-3
```

Launch TensorBoard to monitor loss/metrics:

```powershell
tensorboard --logdir runs
```

### 3) Local Eval

- Evaluate on local minival using the saved model. This uses the Falcon evaluator and reports CER/WER.

```powershell
python decoder_demos/hritik_sample.py eval --evaluation local --model-path local_data/hritik_h2/model --phase minival --split h2
```

Expected output: a metrics dict with WER/CER and normalized latency printed in the console.

### 4) EvalAI Submission

- Package the decoder into a Docker image using `decoder_demos/hritik_sample.Dockerfile`.

Build image:

```powershell
docker build -t hritik_h2 -f .\decoder_demos\hritik_sample.Dockerfile .
```

Sanity check locally (mounts your `./data` as evaluation data and runs local mode):

```powershell
bash .\test_docker_local.sh --docker-name hritik_h2
```

Submit to EvalAI (after installing and configuring EvalAI CLI):

```powershell
evalai push hritik_h2:latest --phase few-shot-minival-2319 --private
```

Notes:

- If GPU wheels are unavailable for your CUDA, replace the JAX install in the Dockerfile with CPU-only `pip install jax jaxlib`.
- Ensure your user can run Docker without sudo if pushing from a Linux host.

### Docker Submission

To interface with our challenge, your code will need to be packaged in a Docker container that is submitted to EvalAI. Try this process by building and running the provided `sklearn_sample.Dockerfile`, to confirm your setup works. Do this with the following commands (once Docker is installed)

```powershell
# Build
docker build -t sk_smoke -f ./decoder_demos/sklearn_sample.Dockerfile .
bash test_docker_local.sh --docker-name sk_smoke
```

For an example Dockerfile with annotations regarding the necessity and function of each line, see `decoder_demos/template.Dockerfile`.

## EvalAI Submission

Please ensure that your submission runs locally before running remote evaluation. You can run the previously listed commands with your own Dockerfile (in place of sk_smoke). This should produce a log of nontrivial metrics (evaluation is run on locally available minival).

To submit to the FALCON benchmark once your decoder Docker container is ready, follow the instructions on the [EvalAI submission tab]((https://eval.ai/web/challenges/challenge-page/2319/submission)). This will instruct you to first install EvalAI, then add your token, and finally push the submission. It should look something like:
`
evalai push mysubmission:latest --phase --phase few-shot-<test/minival>-2319 --private
`
(Note that you will not see these instruction unless you have first created a team to submit. The phase should contain a specific challenge identifier. You may need to refresh the page before instructions will appear.)

Please note that all submissions are subject to a 6 hour time limit.

### Troubleshooting

Docker:

- If this is your first time with docker, note that `sudo` access is needed, or your user needs to be in the `docker` group. `docker info` should run without error.
- While `sudo` is sufficient for local development, the EvalAI submission step will ultimately require your user to be able to run `docker` commands without `sudo`.
- To do this, [add yourself to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/). Note you may [need vigr](https://askubuntu.com/questions/964040/usermod-says-account-doesnt-exist-but-adduser-says-it-does) to add your own user.

EvalAI:

- `pip install evalai` may fail on python 3.11, see [https://github.com/aio-libs/aiohttp/issues/6600](https://github.com/aio-libs/aiohttp/issues/6600). We recommend creating a separate env for submission in this case.
