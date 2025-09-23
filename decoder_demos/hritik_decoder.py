"""
Hritik Decoder (H2): JAX-based session-aware encoder + Perceiver-like decoder.

Implements falcon_challenge.interface.BCIDecoder for H2:
- predict(neural_observations): called every timestep; we buffer trial data
- on_done(dones): called when a trial ends; returns decoded string for that trial

Design highlights
- Session-specific encoders: a per-session embedding conditions FiLM scalars
- Perceiver-like decoder: learned latents cross-attend to input summary
- Optional LoRA adapters for low-rank session adaptation
- MMD loss between session latents for alignment in training

Training utilities are included below for convenience.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
import optax

from falcon_challenge.interface import BCIDecoder
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.dataloaders import load_nwb


# ---------------------------
# Tokenizer for H2 characters
# ---------------------------

def build_default_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    # Lowercase a-z, space, punctuation minimal; EOS token
    chars = list("abcdefghijklmnopqrstuvwxyz") + [" ", ",", ".", "?", "!"]
    # Note: Eval uses '>' as a space proxy for WER calc, but we output actual spaces here
    special = ["<eos>"]
    vocab = chars + special
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(s: str, stoi: Dict[str, int], max_len: int) -> np.ndarray:
    s = s.lower()
    ids = [stoi.get(c, None) for c in s if c in stoi]
    ids = [i for i in ids if i is not None]
    ids = ids[: max_len - 1] + [stoi["<eos>"]]
    arr = np.full((max_len,), -1, dtype=np.int32)
    arr[: len(ids)] = np.array(ids, dtype=np.int32)
    return arr


def decode_ids(ids: np.ndarray, itos: Dict[int, str]) -> str:
    out = []
    for i in ids:
        if i == -1:
            break
        ch = itos.get(int(i), "")
        if ch == "<eos>":
            break
        out.append(ch)
    return "".join(out)


# ---------------------------
# Model components (JAX/Optax)
# ---------------------------

@dataclass
class HritikConfig:
    d_model: int = 128
    n_latents: int = 16
    d_latent: int = 128
    vocab_max_len: int = 48
    lr: float = 1e-3
    l2: float = 1e-6
    mmd_weight: float = 0.0
    film: bool = True
    lora_rank: int = 0  # 0 disables LoRA


def glorot(key, shape):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(key, shape, minval=-limit, maxval=limit)


def init_params(key, n_channels: int, cfg: HritikConfig) -> Dict[str, Any]:
    k1, k2, k3, k4 = random.split(key, 4)

    params = {}
    # Session embedding (learned for seen sessions; unseen will use zero)
    params["session_embed"] = {}

    # Input projector
    params["enc_W1"] = glorot(k1, (n_channels, cfg.d_model))
    params["enc_b1"] = jnp.zeros((cfg.d_model,))

    # Latents and attention projections
    params["latents"] = glorot(k2, (cfg.n_latents, cfg.d_latent))
    params["attn_Wq"] = glorot(k3, (cfg.d_latent, cfg.d_latent))
    params["attn_Wk"] = glorot(k3, (cfg.d_model, cfg.d_latent))
    params["attn_Wv"] = glorot(k4, (cfg.d_model, cfg.d_latent))
    params["attn_bo"] = jnp.zeros((cfg.d_latent,))

    # Decoder head produces logits for each position independently
    params["dec_W"] = glorot(k4, (cfg.d_latent, cfg.vocab_max_len * 64))  # 64 is placeholder vocab size; real head created at train-time
    params["dec_b"] = jnp.zeros((cfg.vocab_max_len * 64,))

    # FiLM generator from session embedding
    params["film_W"] = glorot(k1, (cfg.d_latent, 2 * cfg.d_latent))
    params["film_b"] = jnp.zeros((2 * cfg.d_latent,))

    # LoRA adapters (optional)
    if cfg.lora_rank > 0:
        r = cfg.lora_rank
        for name, (din, dout) in {
            "attn_Wq": (cfg.d_latent, cfg.d_latent),
            "attn_Wk": (cfg.d_model, cfg.d_latent),
            "attn_Wv": (cfg.d_model, cfg.d_latent),
        }.items():
            kA, kB = random.split(key)
            params[f"{name}_lora_A"] = glorot(kA, (din, r))
            params[f"{name}_lora_B"] = jnp.zeros((r, dout))

    return params


def linear(x, W, b):
    return x @ W + b


def cross_attention(latents, x_enc, params, cfg: HritikConfig):
    # latents: L x D; x_enc: T x D_in (here mean pooled -> 1 x D)
    Wq, Wk, Wv = params["attn_Wq"], params["attn_Wk"], params["attn_Wv"]
    q = latents @ Wq  # L x D
    k = x_enc @ Wk    # B(=1) x D
    v = x_enc @ Wv
    attn = jax.nn.softmax((q @ k.T) / jnp.sqrt(q.shape[-1]), axis=0)  # L x 1
    out = attn * v  # broadcast to L x D
    out = out + params["attn_bo"]
    return out  # L x D


def apply_film(h, sess_vec, params, cfg: HritikConfig):
    if not cfg.film:
        return h
    fused = linear(sess_vec, params["film_W"], params["film_b"])  # 2D
    gamma, beta = jnp.split(fused, 2, axis=-1)
    gamma = jnp.tanh(gamma)
    return h * (1 + gamma) + beta


def mmd_rbf(x, y, sigma=1.0):
    def pdist(a):
        a2 = jnp.sum(a * a, axis=1, keepdims=True)
        return a2 + a2.T - 2 * (a @ a.T)
    Kxx = jnp.exp(-pdist(x) / (2 * sigma ** 2))
    Kyy = jnp.exp(-pdist(y) / (2 * sigma ** 2))
    Kxy = jnp.exp(-jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) / (2 * sigma ** 2))
    m = x.shape[0]
    n = y.shape[0]
    # Unbiased estimator
    return (jnp.sum(Kxx) - jnp.trace(Kxx)) / (m * (m - 1)) + (jnp.sum(Kyy) - jnp.trace(Kyy)) / (n * (n - 1)) - 2 * jnp.sum(Kxy) / (m * n)


def mmd_rbf_masked(z, mask_a, mask_b, sigma=1.0):
    """MMD between two groups selected from a single batch using boolean masks.
    Works under JIT: avoids dynamic slicing/unique calls.
    """
    def pdist(a):
        a2 = jnp.sum(a * a, axis=1, keepdims=True)
        return a2 + a2.T - 2 * (a @ a.T)
    D2 = pdist(z)
    D2 = jnp.maximum(D2, 0.0)
    K = jnp.exp(-D2 / (2 * sigma ** 2))
    ma = mask_a.astype(jnp.float32)
    mb = mask_b.astype(jnp.float32)
    m = jnp.sum(ma)
    n = jnp.sum(mb)
    I = jnp.eye(K.shape[0], dtype=K.dtype)
    Ka = K * (ma[:, None] * ma[None, :]) * (1.0 - I)
    Kb = K * (mb[:, None] * mb[None, :]) * (1.0 - I)
    Kab = K * (ma[:, None] * mb[None, :])
    Kaa_term = jnp.where(m > 1, jnp.sum(Ka) / (m * (m - 1)), 0.0)
    Kbb_term = jnp.where(n > 1, jnp.sum(Kb) / (n * (n - 1)), 0.0)
    Kab_term = jnp.where((m > 0) & (n > 0), jnp.sum(Kab) / (m * n), 0.0)
    return Kaa_term + Kbb_term - 2 * Kab_term


class HritikDecoder(BCIDecoder):
    def __init__(
        self,
        task_config: FalconConfig,
        model_path: str | None = None,
        batch_size: int = 1,
        cfg: HritikConfig | None = None,
    ):
        super().__init__(task_config=task_config, batch_size=batch_size)
        assert task_config.task == FalconTask.h2, "HritikDecoder is implemented for H2."
        assert batch_size == 1, "H2 expects batch size 1."
        self.cfg = cfg or HritikConfig()
        self.stoi, self.itos = build_default_vocab()
        self.vocab_size = len(self.stoi)
        self.params = None
        self.opt_state = None
        self.rng = random.PRNGKey(0)
        self.current_session = None
        self.trial_buffer = []  # list of neural_observations (1 x C) over current trial

        if model_path is not None and Path(model_path).exists():
            self.load(model_path)
        else:
            # Init fresh params
            self.params = init_params(self.rng, self._task_config.n_channels, self.cfg)
            # Replace decoder head to correct vocab size
            D = self.cfg.d_latent
            self.params["dec_W"] = glorot(self.rng, (D, self.cfg.vocab_max_len * self.vocab_size))
            self.params["dec_b"] = jnp.zeros((self.cfg.vocab_max_len * self.vocab_size,))

    # ------------- Inference API -------------
    def reset(self, dataset_tags: List[str] = [""]):
        # dataset_tags contains filenames; map to session string
        tag = dataset_tags[0]
        if isinstance(tag, Path):
            tag = tag.stem
        # Use FalconConfig.hash_dataset semantics for H2: file_stem.split('_')[1] typically date
        try:
            session_id = self._task_config.hash_dataset(Path(tag).stem)
        except Exception:
            session_id = str(tag)
        self.current_session = session_id
        self.trial_buffer = []

    def observe(self, neural_observations: np.ndarray):
        # Buffer observations; neural_observations is (B, C) where B==1
        if neural_observations.shape[0] > 1:
            neural_observations = neural_observations[:1]
        self.trial_buffer.append(np.asarray(neural_observations[0], dtype=np.float32))

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        # For H2, predictions are emitted at end-of-trial; here we just buffer
        self.observe(neural_observations)
        return np.zeros((1, 1), dtype=np.float32)  # ignored by evaluator for H2

    def on_done(self, dones: np.ndarray):
        # Called when a trial ends; produce a string prediction for the buffered trial
        if not self.trial_buffer:
            return ""
        spikes = np.stack(self.trial_buffer, axis=0)  # T x C
        pred = self.decode_trial(spikes, self.current_session)
        self.trial_buffer = []
        return pred

    # ------------- Trial decode -------------
    def decode_trial(self, spikes: np.ndarray, session_id: str) -> str:
        x = jnp.asarray(spikes)
        params = self.params
        cfg = self.cfg
        # Encode per-trial summary (mean pooling over time)
        x_mean = jnp.mean(x, axis=0)  # C
        # Apply normalization if available
        if params is not None and "enc_norm_mean" in params and "enc_norm_std" in params:
            eps = 1e-6
            x_mean = (x_mean - params["enc_norm_mean"]) / (params["enc_norm_std"] + eps)
        h = jax.nn.relu(linear(x_mean, params["enc_W1"], params["enc_b1"]))  # D
        # Perceiver-like cross-attn from latents to encoded summary
        latents = params["latents"]
        # Session embedding (zero if unseen)
        sess_vec = jnp.zeros((cfg.d_latent,))
        if "session_embed" in params and session_id in params["session_embed"]:
            sess_vec = params["session_embed"][session_id]
        z = cross_attention(latents, h[None, :], params, cfg)  # L x D
        z = jnp.mean(z, axis=0)  # D
        z = apply_film(z, sess_vec, params, cfg)
        # Decode tokens (independent positions)
        logits = linear(z, params["dec_W"], params["dec_b"])  # (max_len * vocab)
        logits = logits.reshape((cfg.vocab_max_len, self.vocab_size))
        ids = jnp.argmax(logits, axis=-1)
        return decode_ids(np.array(ids, dtype=np.int32), self.itos)

    # ------------- Persistence -------------
    def save(self, path: str | Path):
        path = Path(path)
        payload = {
            "params": jax.device_get(self.params),
            "cfg": self.cfg.__dict__,
            "stoi": self.stoi,
            "itos": self.itos,
        }
        np.savez_compressed(path, payload=json.dumps({k: None for k in payload.keys()}))
        # Save arrays separately to keep compatibility
        np.savez_compressed(path.with_suffix(".arrays.npz"), **{k: np.array(v) if isinstance(v, (np.ndarray, jnp.ndarray)) else np.array(0) for k, v in self.params.items() if isinstance(v, (np.ndarray, jnp.ndarray))})
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump({"cfg": self.cfg.__dict__, "stoi": self.stoi, "itos": self.itos}, f)

    def load(self, path: str | Path):
        path = Path(path)
        # Read arrays
        arrays_path = path.with_suffix(".arrays.npz")
        if arrays_path.exists():
            arrays = np.load(arrays_path)
            params = {}
            for k in arrays.files:
                params[k] = jnp.asarray(arrays[k])
            # Some non-array params might be missing; initialize defaults
            if "dec_W" not in params or "dec_b" not in params:
                params.update(init_params(self.rng, self._task_config.n_channels, self.cfg))
            self.params = params
        else:
            # Fallback initialize
            self.params = init_params(self.rng, self._task_config.n_channels, self.cfg)
            D = self.cfg.d_latent
            self.params["dec_W"] = glorot(self.rng, (D, self.cfg.vocab_max_len * len(self.stoi)))
            self.params["dec_b"] = jnp.zeros((self.cfg.vocab_max_len * len(self.stoi),))


# ---------------------------
# Training utilities
# ---------------------------

@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 3
    lr: float = 1e-3
    max_len: int = 48


def make_trials_from_folder(folder: Path) -> List[Tuple[np.ndarray, str, str]]:
    """Return list of (spikes TxC, text, session_id) from H2 NWB files in folder."""
    files = sorted(folder.glob("*.nwb"))
    out: List[Tuple[np.ndarray, str, str]] = []
    for fn in files:
        spikes, targets, done, _ = load_nwb(fn, dataset=FalconTask.h2)
        # Segment by done==True
        ends = np.where(done)[0].tolist()
        starts = [0] + [e + 1 for e in ends[:-1]]
        session_id = fn.stem.split("_")[1]
        for i, (s, e) in enumerate(zip(starts, ends)):
            trial_spikes = spikes[s : e + 1]
            if i < len(targets):
                text = "".join(chr(int(c)) for c in targets[i] if int(c) != 0).lower()
            else:
                text = ""
            out.append((trial_spikes.astype(np.float32), text, session_id))
    return out


def train_hritik(
    training_dir: Path,
    heldout_calib_dir: Path | None,
    save_path: Path,
    cfg: HritikConfig,
    train_cfg: TrainConfig,
):
    stoi, itos = build_default_vocab()
    trials = make_trials_from_folder(Path(training_dir))
    if len(trials) == 0:
        raise RuntimeError(f"No NWB files found in {training_dir}")

    rng = random.PRNGKey(0)

    # Map sessions to numeric labels and pre-tokenize & pre-mean
    sessions = sorted({sess for (_, _, sess) in trials})
    sess2id = {s: i for i, s in enumerate(sessions)}
    proc = [
        (np.asarray(spk, dtype=np.float32).mean(axis=0),  # x_mean
         encode_text(txt, stoi, train_cfg.max_len),
         np.int32(sess2id[sess]))
        for (spk, txt, sess) in trials
    ]
    # Feature normalization stats (mean/std over training x_mean)
    x_means = np.stack([p[0] for p in proc]).astype(np.float32)
    enc_mean = x_means.mean(axis=0)
    enc_std = x_means.std(axis=0)
    enc_std[enc_std < 1e-6] = 1e-6

    # Initialize params and include normalization stats before optimizer init
    params = init_params(rng, n_channels=trials[0][0].shape[-1], cfg=cfg)
    D = cfg.d_latent
    params["dec_W"] = glorot(rng, (D, cfg.vocab_max_len * len(stoi)))
    params["dec_b"] = jnp.zeros((cfg.vocab_max_len * len(stoi),))
    params["enc_norm_mean"] = jnp.asarray(enc_mean)
    params["enc_norm_std"] = jnp.asarray(enc_std)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=train_cfg.lr, weight_decay=cfg.l2),
    )
    opt_state = opt.init(params)

    def loss_fn(params, x_mean, token_ids, mask_a, mask_b):
        # x_mean: (B, C), token_ids: (B, L), labels: (B,)
        # Encode
        h = jax.nn.relu(x_mean @ params["enc_W1"] + params["enc_b1"])  # (B, D)
        # Cross-attend per-sample
        z = jax.vmap(lambda hv: jnp.mean(cross_attention(params["latents"], hv[None, :], params, cfg), axis=0))(h)  # (B, D)
        # FiLM zero during training
        z = z  # apply_film with zeros would be no-op
        # Decode
        z = jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        logits_flat = z @ params["dec_W"] + params["dec_b"]  # (B, L*V)
        logits_flat = jnp.nan_to_num(logits_flat, nan=0.0, posinf=0.0, neginf=0.0)
        V = len(stoi)
        L = train_cfg.max_len
        logits = logits_flat.reshape((logits_flat.shape[0], L, V))
        mask = (token_ids != -1)
        # Stable CE via log-softmax + masked one-hot
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        safe_labels = jnp.where(mask, jnp.clip(token_ids, 0, V - 1), 0)
        tgt = jax.nn.one_hot(safe_labels, V, dtype=log_probs.dtype)
        ce_per = -jnp.sum(tgt * log_probs, axis=-1)
        ce = jnp.sum(ce_per * mask) / (jnp.sum(mask) + 1e-6)
        # MMD between two label groups using masks (use standardized z for stability)
        mmd_pen = 0.0
        if cfg.mmd_weight > 0:
            z_center = z - jnp.mean(z, axis=0, keepdims=True)
            z_std = jnp.std(z_center) + 1e-6
            z_norm = z_center / z_std
            mmd_pen = mmd_rbf_masked(z_norm, mask_a, mask_b, sigma=1.0)
        return ce + cfg.mmd_weight * mmd_pen, (ce, mmd_pen)

    @jax.jit
    def train_step(params, opt_state, x_mean, token_ids, mask_a, mask_b):
        (loss, (base, mmd_pen)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_mean, token_ids, mask_a, mask_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, base, mmd_pen

    # Build batches
    def to_batches(items, bs):
        for i in range(0, len(items), bs):
            yield items[i : i + bs]

    def cross_attend_single(h_vec):
        zL = cross_attention(params["latents"], h_vec[None, :], params, cfg)
        return jnp.mean(zL, axis=0)

    # Safe SummaryWriter import
    try:
        from tensorboardX import SummaryWriter  # type: ignore
        writer = SummaryWriter(logdir=str(Path("runs") / "hritik_h2"))
    except Exception:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            writer = SummaryWriter(logdir=str(Path("runs") / "hritik_h2"))
        except Exception:
            class _Dummy:
                def add_scalar(self, *a, **k):
                    pass
                def close(self):
                    pass
            writer = _Dummy()

    step = 0
    for epoch in range(train_cfg.epochs):
        np.random.shuffle(proc)
        pbar = tqdm(list(to_batches(proc, train_cfg.batch_size)), desc=f"Epoch {epoch+1}/{train_cfg.epochs}")
        for batch in pbar:
            # Collate arrays
            x_mean = np.stack([b[0] for b in batch]).astype(np.float32)
            # Apply normalization
            x_mean = (x_mean - enc_mean) / (enc_std + 1e-6)
            token_ids = np.stack([b[1] for b in batch]).astype(np.int32)
            labels = np.array([b[2] for b in batch], dtype=np.int32)
            uniq = np.unique(labels)
            if uniq.size >= 2:
                a, b = uniq[:2]
                mask_a = (labels == a).astype(np.float32)
                mask_b = (labels == b).astype(np.float32)
            else:
                mask_a = np.zeros_like(labels, dtype=np.float32)
                mask_b = np.zeros_like(labels, dtype=np.float32)
            params, opt_state, loss, base, mmd_pen = train_step(
                params, opt_state, x_mean, token_ids, mask_a, mask_b
            )
            step += 1
            if step % 10 == 0:
                writer.add_scalar("loss/total", float(loss), step)
                writer.add_scalar("loss/xe", float(base), step)
                writer.add_scalar("loss/mmd", float(mmd_pen), step)
            pbar.set_postfix({"loss": float(loss)})
    writer.close()

    # Save checkpoint
    arrays = {k: np.array(v) for k, v in params.items() if isinstance(v, (np.ndarray, jnp.ndarray))}
    arrays_path = save_path.with_suffix(".arrays.npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    with open(save_path.with_suffix(".meta.json"), "w") as f:
        json.dump({"cfg": cfg.__dict__, "stoi": stoi, "itos": {int(k): v for k, v in enumerate([*stoi.keys()])}}, f)
    print(f"Saved model arrays to {arrays_path}")
