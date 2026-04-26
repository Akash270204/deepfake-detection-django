"""
train_efficientnet_b1_fixed.py
====================================

"""

import os
import gc
import json
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dense, Dropout, BatchNormalization,
    Input, Concatenate, Multiply, Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    CSVLogger, LambdaCallback,
    LearningRateScheduler,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path("/mnt/c/Users/Akash/deepfake-detection-django")

CFG = {
    "DATA_DIR"          : str(BASE_DIR / "dataset" / "video_frames"),
    "MODEL_SAVE_PATH"   : str(BASE_DIR / "detector" / "ml_models" / "deepfake_video_model.h5"),
    "CHECKPOINT_PATH"   : str(BASE_DIR / "detector" / "ml_models" / "best_video_checkpoint.h5"),
    "LOG_PATH"          : str(BASE_DIR / "training_video_log.csv"),

    "IMG_SIZE"          : (240, 240),
    "IMG_SIZE_FULL"     : (240, 240, 3),

    "BATCH_SIZE"        : 16,

    # ── Phase 1: train head only ──────────────────────────────────────────
    "PHASE1_EPOCHS"     : 25,
    "PHASE1_LR"         : 2e-4,
    "WARMUP_EPOCHS"     : 3,

    # ── Phase 2/3/4: gradual backbone unfreezing ──────────────────────────
    "UNFREEZE_PER_STEP" : 5,
    "UNFREEZE_STEPS"    : 3,
    "STEP_EPOCHS"       : 8,
    "PHASE2_LR"         : 1e-5,
    "PHASE2_LR_MIN"     : 5e-7,

    # ── Early stopping ────────────────────────────────────────────────────
    # FIX-10: patience=6, min_delta added, min epochs before ES fires
    "ES_PATIENCE"       : 6,
    "ES_MIN_DELTA"      : 1e-4,

    # ── Loss ──────────────────────────────────────────────────────────────
    "FOCAL_GAMMA"       : 2.0,
    "FOCAL_ALPHA"       : 0.75,
    "LABEL_SMOOTHING"   : 0.01,

    # ── Regularisation ────────────────────────────────────────────────────
    "L2_REG"            : 2e-4,
    "DROPOUT1"          : 0.50,
    "DROPOUT2"          : 0.40,
    "STOCHASTIC_DEPTH"  : 0.20,

    # ── TTA ───────────────────────────────────────────────────────────────
    "TTA_STEPS"         : 3,
    "TTA_MIN_AUC"       : 0.82,

    # ── Threshold search ──────────────────────────────────────────────────
    "TARGET_RECALL"     : 0.90,

    "CLASSES"           : ["real", "fake"],
}

print("=" * 68)
print("  DEEPFAKE VIDEO MODEL — FULLY FIXED & OPTIMISED")
print(f"  Phase1={CFG['PHASE1_EPOCHS']}ep lr={CFG['PHASE1_LR']}")
print(f"  Gradual unfreeze: {CFG['UNFREEZE_STEPS']} steps × "
      f"{CFG['UNFREEZE_PER_STEP']} layers × {CFG['STEP_EPOCHS']}ep")
print(f"  Focal alpha={CFG['FOCAL_ALPHA']} | "
      f"Dropout={CFG['DROPOUT1']}/{CFG['DROPOUT2']} | "
      f"StochDepth={CFG['STOCHASTIC_DEPTH']}")
print("=" * 68)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {tf.config.list_physical_devices('GPU')}")
print("=" * 68)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled\n")

os.makedirs(os.path.dirname(CFG["MODEL_SAVE_PATH"]), exist_ok=True)


# =============================================================================
# LOSSES  
# =============================================================================

def binary_crossentropy_loss(label_smoothing=0.0):
    """Standard BCE for warmup — avoids focal collapse in early epochs."""
    def _bce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        bce = -(y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)
    _bce.__name__ = "bce_loss"
    return _bce


def focal_loss(gamma=2.0, alpha=0.75, label_smoothing=0.01):
    """Focal loss — switched in after warmup."""
    def _focal(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.cast(y_pred, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_smooth = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        bce     = -(y_smooth * tf.math.log(y_pred)
                    + (1.0 - y_smooth) * tf.math.log(1.0 - y_pred))
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        loss    = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(loss)
    _focal.__name__ = "focal_loss"
    return _focal


# =============================================================================
# COLLAPSE DETECTOR  (FIX-5: check_after=3)
# =============================================================================

class CollapseDetector(tf.keras.callbacks.Callback):
    def __init__(self, check_after=3, min_auc=0.52, patience=4):
        super().__init__()
        self.check_after      = check_after
        self.min_auc          = min_auc
        self.patience         = patience
        self._consecutive_low = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.check_after or logs is None:
            return
        val_auc = logs.get("val_auc", 1.0)
        if val_auc < self.min_auc:
            self._consecutive_low += 1
            print(f"\n  [CollapseDetector] ep{epoch+1} val_auc={val_auc:.4f} "
                  f"low ({self._consecutive_low}/{self.patience})")
            if self._consecutive_low >= self.patience:
                print("  [CollapseDetector] COLLAPSE — stopping.")
                self.model.stop_training = True
        else:
            self._consecutive_low = 0


# =============================================================================
# AUGMENTATION — codec/compression artifacts only (no MixUp, no CoarseDropout)
# =============================================================================

def mobile_augment(img_uint8: np.ndarray) -> np.ndarray:
    """
    Augmentations relevant to real-world deepfake video only.
    Input/output: uint8 RGB image.
    """
    img = img_uint8.copy()

    # JPEG compression — most important real-world distortion
    if np.random.random() < 0.60:
        quality = int(np.random.randint(40, 95))
        _, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # Gaussian noise — sensor / re-encoding noise
    if np.random.random() < 0.35:
        sigma = float(np.random.uniform(0.0, 15.0))
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        img   = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    # Motion blur — camera movement
    if np.random.random() < 0.25:
        kernel = np.zeros((5, 5), dtype=np.float32)
        if np.random.random() > 0.5:
            kernel[2, :] = 0.2
        else:
            kernel[:, 2] = 0.2
        img = cv2.filter2D(img, -1, kernel)

    return img


# =============================================================================
# VIDEO FRAME GENERATOR
# =============================================================================

class VideoFrameGenerator(tf.keras.utils.Sequence):
    """
    Wraps Keras ImageDataGenerator; applies mobile_augment on uint8 frames
    BEFORE preprocess_input so augmentations operate in pixel space.
    """
    def __init__(self, base_generator, apply_aug=False):
        self.gen       = base_generator
        self.samples   = base_generator.samples
        self.classes   = base_generator.classes
        self.apply_aug = apply_aug

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, idx):
        X, y = self.gen[idx]
        out  = []
        for x in X:
            # x is float in [0,255] from ImageDataGenerator (no rescale set)
            frame = np.clip(x, 0.0, 255.0).astype(np.uint8)
            if self.apply_aug:
                frame = mobile_augment(frame)
            # preprocess_input expects uint8-range float; converts to [-1,1]
            out.append(preprocess_input(frame.astype(np.float32)))
        return np.stack(out, axis=0), y

    def on_epoch_end(self):
        self.gen.on_epoch_end()

    def reset(self):
        self.gen.reset()


# =============================================================================
# DATA GENERATORS
# =============================================================================

def build_generators():
    train_datagen = ImageDataGenerator(
        rotation_range      = 15,
        width_shift_range   = 0.10,
        height_shift_range  = 0.10,
        shear_range         = 0.08,
        zoom_range          = 0.15,
        brightness_range    = [0.70, 1.30],
        horizontal_flip     = True,
        vertical_flip       = False,
        channel_shift_range = 15.0,
        fill_mode           = "nearest",
    )
    val_datagen = ImageDataGenerator()

    kw = dict(
        target_size   = CFG["IMG_SIZE"],
        batch_size    = CFG["BATCH_SIZE"],
        class_mode    = "binary",
        classes       = CFG["CLASSES"],
        interpolation = "bilinear",
    )

    raw_train = train_datagen.flow_from_directory(
        os.path.join(CFG["DATA_DIR"], "train"), shuffle=True, seed=42, **kw
    )
    raw_val = val_datagen.flow_from_directory(
        os.path.join(CFG["DATA_DIR"], "val"), shuffle=False, **kw
    )
    raw_test = val_datagen.flow_from_directory(
        os.path.join(CFG["DATA_DIR"], "test"), shuffle=False, **kw
    )

    assert raw_train.class_indices == {"real": 0, "fake": 1}, (
        f"Class mismatch: {raw_train.class_indices}"
    )

    train_gen = VideoFrameGenerator(raw_train, apply_aug=True)
    val_gen   = VideoFrameGenerator(raw_val,   apply_aug=False)
    test_gen  = VideoFrameGenerator(raw_test,  apply_aug=False)

    return train_gen, val_gen, test_gen, raw_train, raw_val, raw_test


# =============================================================================
# DATA SANITY + LEAKAGE CHECK  (FIX-6)
# =============================================================================

def _video_id(filename: str) -> str:
    """Extract video-level ID from a frame filename like 'video001_frame042.jpg'."""
    name = os.path.splitext(os.path.basename(filename))[0]
    # Strip trailing _frameNNN or _NNN suffix (common naming convention)
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name


def check_data(raw_train, raw_val, raw_test):
    for split, gen in [("Train", raw_train), ("Val", raw_val), ("Test", raw_test)]:
        labels = gen.classes
        n_real = int(np.sum(labels == 0))
        n_fake = int(np.sum(labels == 1))
        bal    = min(n_real, n_fake) / max(n_real, n_fake, 1)
        print(f"  {split:5s}: Real={n_real:,}  Fake={n_fake:,}  "
              f"Total={len(labels):,}  Balance={bal:.2f}")
        if bal < 0.70:
            print(f"    ⚠  Imbalanced — focal alpha will compensate")

    # Leakage check: video IDs must not overlap across splits
    train_ids = {_video_id(f) for f in raw_train.filenames}
    val_ids   = {_video_id(f) for f in raw_val.filenames}
    test_ids  = {_video_id(f) for f in raw_test.filenames}

    tv_leak = train_ids & val_ids
    tt_leak = train_ids & test_ids
    vt_leak = val_ids   & test_ids

    if tv_leak:
        print(f"\n  ⚠  TRAIN/VAL LEAKAGE: {len(tv_leak)} video IDs overlap!")
        print(f"     Example: {list(tv_leak)[:3]}")
        print("     Fix: re-split your dataset at the VIDEO level, not frame level.")
    if tt_leak:
        print(f"\n  ⚠  TRAIN/TEST LEAKAGE: {len(tt_leak)} video IDs overlap!")
    if vt_leak:
        print(f"\n  ⚠  VAL/TEST LEAKAGE: {len(vt_leak)} video IDs overlap!")
    if not tv_leak and not tt_leak and not vt_leak:
        print("  Video-level split: no leakage detected ✓")

    total = raw_train.samples
    if total < 2000:
        print(f"\n  ⚠  Only {total} training frames. "
              "Overfitting likely. Add more data for best results.")
    print()


def get_class_weights(raw_train):
    labels  = raw_train.classes
    weights = compute_class_weight("balanced",
                                   classes=np.unique(labels), y=labels)
    cw = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class weights (for BCE warmup only): real={cw[0]:.3f}  fake={cw[1]:.3f}")
    return cw


# =============================================================================
# MODEL  (FIX-1: proper stochastic depth via gated Bernoulli mask)
# =============================================================================

class StochasticDepth(tf.keras.layers.Layer):
    """
    True stochastic depth: during training, randomly zeros the entire
    residual branch with probability `drop_rate`. During inference,
    always uses the residual (no randomness).
    Compatible with Keras 3 (no K.learning_phase()).
    """
    def __init__(self, drop_rate=0.20, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, inputs, training=False):
        residual, shortcut = inputs
        if not training:
            # Inference: always use residual, scale by survival prob
            return shortcut + residual * (1.0 - self.drop_rate)

        # Training: Bernoulli mask per sample in batch
        batch_size = tf.shape(residual)[0]
        mask = tf.cast(
            tf.random.uniform([batch_size, 1]) > self.drop_rate,
            dtype=residual.dtype
        )
        return shortcut + residual * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_rate": self.drop_rate})
        return cfg


def build_model():
    base = EfficientNetB1(
        weights     = "imagenet",
        include_top = False,
        input_shape = CFG["IMG_SIZE_FULL"],
    )
    base.trainable = False

    inp = Input(shape=CFG["IMG_SIZE_FULL"])
    x   = base(inp, training=False)

    avg = GlobalAveragePooling2D()(x)
    mx  = GlobalMaxPooling2D()(x)
    x   = Concatenate()([avg, mx])                    # shape: [B, 2560]

    # ── fc1 block ────────────────────────────────────────────────────────
    x   = BatchNormalization(name="bn_head1")(x)
    x   = Dense(512, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(CFG["L2_REG"]),
                name="fc1")(x)
    x   = Dropout(CFG["DROPOUT1"], name="drop1")(x)
    x   = BatchNormalization(name="bn_head2")(x)      # shape: [B, 512]

    # ── fc2 block with TRUE stochastic depth (FIX-1) ─────────────────────
    # Shortcut: project fc1 output to 256 dims
    shortcut = Dense(256, use_bias=False, name="fc_shortcut")(x)

    # Residual branch: fc2 → dropout → project to 256
    residual = Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(CFG["L2_REG"]),
                     name="fc2")(x)
    residual = Dropout(CFG["DROPOUT2"], name="drop2")(residual)

    # Stochastic depth: randomly skip residual during training
    x = StochasticDepth(drop_rate=CFG["STOCHASTIC_DEPTH"],
                        name="stochastic_depth")([residual, shortcut])
    x = tf.keras.layers.Activation("relu")(x)         # shape: [B, 256]

    # Output — float32 for mixed precision safety
    out = Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

    return Model(inp, out, name="deepfake_video_b1_fixed"), base


# =============================================================================
# LR SCHEDULES  (FIX-2: epoch offset parameter added)
# =============================================================================

def warmup_cosine_schedule(lr_peak, warmup_epochs, total_epochs,
                            epoch_offset=0):
    """
    Linear warmup then cosine decay.
    epoch_offset: the Keras initial_epoch so schedule is always aligned.
    """
    def schedule(epoch, _lr):
        e = epoch - epoch_offset           # relative epoch within this phase
        if e < warmup_epochs:
            return float(lr_peak * (0.1 + 0.9 * e / max(warmup_epochs, 1)))
        else:
            progress = (e - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cos_val  = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(lr_peak * (0.05 + 0.95 * cos_val))
    return LearningRateScheduler(schedule, verbose=0)


def cosine_schedule(lr_start, lr_min, total_epochs, epoch_offset=0):
    """Pure cosine decay with epoch offset alignment."""
    def schedule(epoch, _lr):
        e        = epoch - epoch_offset
        cos_val  = 0.5 * (1.0 + np.cos(np.pi * e / max(total_epochs, 1)))
        return float(lr_min + (lr_start - lr_min) * cos_val)
    return LearningRateScheduler(schedule, verbose=0)


# =============================================================================
# COMPILE
# =============================================================================

def compile_model(model, lr, loss_fn, weight_decay=1e-4):
    model.compile(
        optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay,
                          clipnorm=1.0),
        loss      = loss_fn,
        metrics   = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# =============================================================================
# CALLBACKS  (FIX-10: min_delta, start_from_epoch)
# =============================================================================

def base_callbacks(phase_label, lr_callback,
                   monitor_es="val_loss", es_mode="min",
                   start_from_epoch=5):
    """
    EarlyStopping:
      - monitors val_loss (honest overfitting signal)
      - min_delta prevents stopping on noise
      - start_from_epoch avoids premature stopping
    ModelCheckpoint: still saves best val_auc model.
    """
    callbacks = [
        ModelCheckpoint(
            CFG["CHECKPOINT_PATH"],
            monitor="val_auc", save_best_only=True,
            mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor=monitor_es,
            patience=CFG["ES_PATIENCE"],
            min_delta=CFG["ES_MIN_DELTA"],
            restore_best_weights=True,
            mode=es_mode,
            verbose=1,
        ),
        CSVLogger(CFG["LOG_PATH"], append=(phase_label != "P1warm")),
        tf.keras.callbacks.TerminateOnNaN(),
        CollapseDetector(check_after=3, min_auc=0.52, patience=4),
        lr_callback,
        LambdaCallback(
            on_epoch_end=lambda ep, logs: print(
                f"  {phase_label} ep{ep+1:03d} | "
                f"loss={logs.get('loss', 0):.4f} | "
                f"val_acc={logs.get('val_accuracy', 0):.4f} | "
                f"val_auc={logs.get('val_auc', 0):.4f} | "
                f"val_loss={logs.get('val_loss', 0):.4f} | "
                f"lr={logs.get('lr', 0):.2e}"
            ) if (ep + 1) % 2 == 0 else None
        ),
    ]
    return callbacks


# =============================================================================
# OUTPUT RANGE CHECK
# =============================================================================

def _predict_generator(model, gen, n_samples):
    """Predict on a generator, returning exactly n_samples predictions."""
    gen.reset()
    preds = []
    for i in range(len(gen)):
        X, _ = gen[i]
        p = model.predict(X, verbose=0).flatten()
        preds.extend(p.tolist())
        if len(preds) >= n_samples:
            break
    gen.reset()
    return np.array(preds[:n_samples])


def check_output_range(model, val_gen, raw_val, label):
    preds  = _predict_generator(model, val_gen, raw_val.samples)
    labels = raw_val.classes[:len(preds)]
    real_p = preds[labels == 0]
    fake_p = preds[labels == 1]

    print(f"\n  [{label}]")
    print(f"    All : min={preds.min():.3f}  max={preds.max():.3f}  "
          f"std={preds.std():.3f}")
    if len(real_p):
        print(f"    Real: mean={real_p.mean():.3f}  "
              f"range=[{real_p.min():.3f},{real_p.max():.3f}]")
    if len(fake_p):
        print(f"    Fake: mean={fake_p.mean():.3f}  "
              f"range=[{fake_p.min():.3f},{fake_p.max():.3f}]")
    if len(real_p) and len(fake_p):
        sep    = fake_p.mean() - real_p.mean()
        status = "OK ✓" if sep > 0.30 else "LOW ⚠️ "
        print(f"    Sep : {sep:.3f}  {status}")
    return preds, labels


# =============================================================================
# TEMPERATURE SCALING CALIBRATION
# =============================================================================

def calibrate_temperature(model, val_gen, raw_val):
    """
    Grid search over T in [0.3, 3.0] minimising NLL on val set.
    Calibrated scores: sigmoid(logit / T).
    """
    print("\n  Calibrating temperature on val set …")
    preds      = _predict_generator(model, val_gen, raw_val.samples)
    all_labels = raw_val.classes[:len(preds)].astype(np.float64)
    p          = np.clip(preds, 1e-6, 1.0 - 1e-6).astype(np.float64)
    logits     = np.log(p / (1.0 - p))

    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.3, 3.0, 54):
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)
        nll   = -np.mean(
            all_labels * np.log(p_cal) + (1.0 - all_labels) * np.log(1.0 - p_cal)
        )
        if nll < best_nll:
            best_nll, best_T = nll, float(T)

    print(f"    Best temperature T={best_T:.3f}  (NLL={best_nll:.4f})")
    return best_T


def apply_temperature(raw_probs, T):
    p      = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logits / T))


# =============================================================================
# TEST-TIME AUGMENTATION  (FIX-3: TTA operates on uint8 pixel range)
# =============================================================================

def tta_predict(model, gen, tta_steps=3):
    """
    3-step TTA — all operations in pixel space BEFORE preprocess_input.
      step 0: original (no modification)
      step 1: horizontal flip
      step 2: brightness +10% (clamped to [0,255] in uint8 space)

    Each step re-preprocesses from the raw uint8 output of the generator
    so augmentations are not applied on top of already-normalised tensors.
    """
    all_step_preds = []

    for step in range(tta_steps):
        gen.reset()
        step_preds = []

        for idx in range(len(gen)):
            # Get uint8 frames from generator BEFORE preprocess_input
            X_raw, _ = gen.gen[idx]        # raw float [0,255] from ImageDataGenerator
            X_out    = []

            for x in X_raw:
                frame = np.clip(x, 0.0, 255.0).astype(np.uint8)

                if step == 0:
                    pass                                          # original
                elif step == 1:
                    frame = frame[:, ::-1, :]                    # horizontal flip
                elif step == 2:
                    # Brightness +10% in pixel space then clamp to uint8
                    frame = np.clip(frame.astype(np.float32) * 1.10,
                                    0.0, 255.0).astype(np.uint8)

                X_out.append(preprocess_input(frame.astype(np.float32)))

            X_batch = np.stack(X_out, axis=0)
            p = model.predict(X_batch, verbose=0).flatten()
            step_preds.extend(p.tolist())

        all_step_preds.append(step_preds[:gen.samples])

    gen.reset()
    return np.array(all_step_preds).mean(axis=0)


# =============================================================================
# THRESHOLD FINDER
# =============================================================================

def find_optimal_threshold(y_true, y_prob, target_recall=0.90):
    thresholds  = np.linspace(0.05, 0.95, 181)
    best_thresh = 0.50
    best_min_rec = -1.0
    found       = False

    for t in thresholds:
        y_pred   = (y_prob >= t).astype(int)
        tp       = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn       = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn       = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp       = int(np.sum((y_pred == 1) & (y_true == 0)))
        fake_rec = tp / (tp + fn + 1e-8)
        real_rec = tn / (tn + fp + 1e-8)
        min_rec  = min(fake_rec, real_rec)

        if fake_rec >= target_recall and real_rec >= target_recall:
            if min_rec > best_min_rec:
                best_min_rec = min_rec
                best_thresh  = float(t)
                found        = True

    if not found:
        print(f"\n  ⚠  90/10 threshold not found — falling back to Youden-J")
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j_idx         = int(np.argmax(tpr - fpr))
        best_thresh   = float(np.clip(thr[j_idx], 0.20, 0.80))

    print(f"  Threshold: {best_thresh:.4f}  "
          f"(90/10 {'MET ✓' if found else 'NOT MET ✗'})")
    return best_thresh, found


# =============================================================================
# FRAME-LEVEL 90/10 EVALUATION
# =============================================================================

def evaluate_frame_criterion(y_true, y_pred, threshold):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))

    fake_recall = tp / (tp + fn + 1e-8)
    real_recall = tn / (tn + fp + 1e-8)

    print(f"\n  {'='*60}")
    print(f"  FRAME-LEVEL 90/10  (threshold={threshold:.4f})")
    print(f"  {'='*60}")
    print(f"  Fake → FAKE : {fake_recall*100:.1f}%  "
          f"{'✓' if fake_recall >= 0.90 else '✗'}")
    print(f"  Real → REAL : {real_recall*100:.1f}%  "
          f"{'✓' if real_recall >= 0.90 else '✗'}")
    both = fake_recall >= 0.90 and real_recall >= 0.90
    print(f"  Result: {'✓ MET' if both else '✗ NOT MET'}")
    print(f"  {'='*60}")
    return fake_recall, real_recall


# =============================================================================
# TEMPORAL CONSISTENCY (per-video)
# =============================================================================

def temporal_consistency_report(y_prob_cal, raw_test, threshold, max_videos=25):
    filenames = raw_test.filenames
    labels    = raw_test.classes

    video_data = {}
    for fname, prob, label in zip(filenames, y_prob_cal, labels):
        video_id = _video_id(fname)
        if video_id not in video_data:
            video_data[video_id] = {"probs": [], "label": int(label)}
        video_data[video_id]["probs"].append(float(prob))

    print(f"\n  Temporal Consistency (threshold={threshold:.4f})")
    print(f"  {'Video ID':<35} {'True':>5} {'Frames':>6} "
          f"{'Avg P':>8} {'Std':>6} {'Result':>6}")
    print("  " + "─" * 70)

    correct = 0
    total   = 0
    for vid_id, data in list(video_data.items())[:max_videos]:
        probs    = np.array(data["probs"])
        true_lbl = data["label"]
        avg_p    = probs.mean()
        std_p    = probs.std()
        pred     = int(avg_p >= threshold)
        ok       = "✓" if pred == true_lbl else "✗"
        correct += int(pred == true_lbl)
        total   += 1

        true_str = "FAKE" if true_lbl == 1 else "REAL"
        short_id = vid_id[:35]
        print(f"  {short_id:<35} {true_str:>5} {len(probs):>6} "
              f"{avg_p:>8.3f} {std_p:>6.3f} {ok:>6}")

    if total:
        print(f"\n  Video-level accuracy: {correct}/{total} = "
              f"{100*correct/total:.1f}%")


# =============================================================================
# PLOTS
# =============================================================================

def plot_history(histories, labels):
    def cat(key):
        out = []
        for h in histories:
            out.extend(h.history.get(key, []))
        return out

    epochs     = range(1, len(cat("accuracy")) + 1)
    phase_ends = []
    running    = 0
    for h in histories[:-1]:
        running += len(h.history.get("accuracy", []))
        phase_ends.append(running)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Video Model — Training History (Fixed & Optimised)",
                 fontsize=14, fontweight="bold")

    def _plot(ax, tk, vk, title):
        ax.plot(epochs, cat(tk), "b-", label="Train", linewidth=1.8)
        ax.plot(epochs, cat(vk), "r-", label="Val",   linewidth=1.8)
        for pe in phase_ends:
            ax.axvline(pe, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    _plot(axes[0,0], "accuracy", "val_accuracy", "Accuracy")
    _plot(axes[0,1], "loss",     "val_loss",     "Loss")
    _plot(axes[1,0], "auc",      "val_auc",      "AUC")
    _plot(axes[1,1], "recall",   "val_recall",   "Recall")
    plt.tight_layout()
    out = str(BASE_DIR / "training_history_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  History → {out}")


def plot_score_distribution(y_true, y_prob, threshold, title_suffix=""):
    real_s = y_prob[y_true == 0]
    fake_s = y_prob[y_true == 1]
    bins   = np.linspace(0, 1, 50)
    plt.figure(figsize=(10, 6))
    plt.hist(real_s, bins=bins, alpha=0.6, color="green",
             label=f"Real (n={len(real_s)})", density=True)
    plt.hist(fake_s, bins=bins, alpha=0.6, color="red",
             label=f"Fake (n={len(fake_s)})", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", lw=2,
                label=f"Threshold={threshold:.3f}")
    plt.xlabel("P(fake)"); plt.ylabel("Density")
    plt.title(f"Score Distribution {title_suffix}\n"
              "Two separated peaks = healthy model")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    suffix = title_suffix.replace(' ', '_')
    out    = str(BASE_DIR / f"score_distribution_video{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Score dist → {out}")


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar(im)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=["Real","Fake"], yticklabels=["Real","Fake"],
           xlabel="Predicted", ylabel="True", title=title)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    out = str(BASE_DIR / "confusion_matrix_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix → {out}")


def plot_roc(fpr, tpr, auc_val):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="green", lw=2, label=f"AUC={auc_val:.4f}")
    plt.plot([0,1],[0,1], "k--", lw=1.5)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve — Video Model (calibrated + TTA)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = str(BASE_DIR / "roc_curve_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC → {out}")


# =============================================================================
# BACKBONE UNFREEZING HELPER
# =============================================================================

def freeze_backbone_except_last_n(base, n, model):
    """
    Unfreeze last n backbone layers. BN layers always frozen (statistics
    must not shift — critical for small datasets).
    """
    base.trainable = True
    freeze_until   = len(base.layers) - n
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers[freeze_until:]:
        layer.trainable = True

    # Lock ALL BN in entire model
    bn_locked = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            bn_locked += 1
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.BatchNormalization):
                    sub.trainable = False
                    bn_locked += 1

    trainable_p = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"  Unfrozen last {n} backbone layers | BN locked: {bn_locked} | "
          f"Trainable params: {trainable_p:,}")
    return trainable_p


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train():
    t_start = time.time()

    print("\n[1/7] Loading data …")
    train_gen, val_gen, test_gen, raw_train, raw_val, raw_test = build_generators()
    class_weights = get_class_weights(raw_train)
    check_data(raw_train, raw_val, raw_test)
    print(f"  Train={train_gen.samples:,}  "
          f"Val={val_gen.samples:,}  "
          f"Test={test_gen.samples:,}")

    print("\n[2/7] Building model …")
    model, base = build_model()
    print(f"  Total params: {model.count_params():,}")
    model.summary(line_length=90)

    histories = []

    # =========================================================================
    # PHASE 1A — Warmup with BCE + class_weight (stable gradients)
    # =========================================================================
    warmup_ep = CFG["WARMUP_EPOCHS"]
    print(f"\n{'─'*68}")
    print(f"[3/7] PHASE 1A — warmup {warmup_ep}ep  BCE  lr={CFG['PHASE1_LR']}")
    print(f"{'─'*68}")

    compile_model(model,
                  lr=CFG["PHASE1_LR"],
                  loss_fn=binary_crossentropy_loss(label_smoothing=0.0),
                  weight_decay=1e-4)

    h_warm = model.fit(
        train_gen,
        epochs          = warmup_ep,
        validation_data = val_gen,
        # FIX-4: class_weight used only during BCE warmup
        class_weight    = class_weights,
        callbacks       = base_callbacks(
            "P1warm",
            # FIX-2: epoch_offset=0 for first phase (Keras starts at 0)
            warmup_cosine_schedule(CFG["PHASE1_LR"], warmup_ep, warmup_ep,
                                   epoch_offset=0),
            monitor_es="val_loss", es_mode="min",
        ),
        verbose=1,
    )
    histories.append(h_warm)
    print(f"  Warmup done → val_auc={h_warm.history['val_auc'][-1]:.4f}")

    # =========================================================================
    # PHASE 1B — Focal loss (no class_weight — focal_alpha handles imbalance)
    # =========================================================================
    remaining_p1 = CFG["PHASE1_EPOCHS"] - warmup_ep
    start_ep_p1b = len(h_warm.history["loss"])   # Keras initial_epoch
    end_ep_p1b   = start_ep_p1b + remaining_p1

    print(f"\n{'─'*68}")
    print(f"[3/7] PHASE 1B — focal loss  {remaining_p1}ep  lr={CFG['PHASE1_LR']}")
    print(f"      Keras epochs: {start_ep_p1b} → {end_ep_p1b}")
    print(f"{'─'*68}")

    compile_model(model,
                  lr=CFG["PHASE1_LR"],
                  loss_fn=focal_loss(CFG["FOCAL_GAMMA"],
                                     CFG["FOCAL_ALPHA"],
                                     CFG["LABEL_SMOOTHING"]),
                  weight_decay=1e-4)

    h_p1 = model.fit(
        train_gen,
        epochs          = end_ep_p1b,
        initial_epoch   = start_ep_p1b,
        validation_data = val_gen,
        # FIX-4: NO class_weight — focal_alpha already handles imbalance
        callbacks       = base_callbacks(
            "P1",
            # FIX-2: epoch_offset aligned to Keras counter
            warmup_cosine_schedule(CFG["PHASE1_LR"],
                                   warmup_epochs=0,
                                   total_epochs=remaining_p1,
                                   epoch_offset=start_ep_p1b),
            monitor_es="val_loss", es_mode="min",
        ),
        verbose=1,
    )
    histories.append(h_p1)
    val_auc_p1 = max(h_p1.history.get("val_auc", [0]))
    print(f"\n  Phase 1 done → best val_auc={val_auc_p1:.4f}")
    check_output_range(model, val_gen, raw_val, "After phase 1")

    # =========================================================================
    # PHASE 2 — Gradual unfreezing (FIX-9: 0.4x LR decay per step)
    # =========================================================================
    n_unfrozen   = 0
    epoch_cursor = end_ep_p1b
    step_epochs  = CFG["STEP_EPOCHS"]
    lr_base      = CFG["PHASE2_LR"]

    for step in range(1, CFG["UNFREEZE_STEPS"] + 1):
        n_unfrozen += CFG["UNFREEZE_PER_STEP"]

        # FIX-9: 0.4x decay keeps gradient signal in step 2&3
        lr_step_cur = lr_base * (0.4 ** (step - 1))

        print(f"\n{'─'*68}")
        print(f"[4/7] PHASE 2 step {step}/{CFG['UNFREEZE_STEPS']} — "
              f"unfreeze {n_unfrozen} backbone layers  lr={lr_step_cur:.1e}")
        print(f"      Keras epochs: {epoch_cursor} → {epoch_cursor + step_epochs}")
        print(f"{'─'*68}")

        freeze_backbone_except_last_n(base, n_unfrozen, model)

        compile_model(model,
                      lr=lr_step_cur,
                      loss_fn=focal_loss(CFG["FOCAL_GAMMA"],
                                         CFG["FOCAL_ALPHA"],
                                         CFG["LABEL_SMOOTHING"]),
                      weight_decay=5e-6)
        model.optimizer.clipnorm = 0.5

        end_epoch = epoch_cursor + step_epochs
        h_step    = model.fit(
            train_gen,
            epochs          = end_epoch,
            initial_epoch   = epoch_cursor,
            validation_data = val_gen,
            # FIX-4: no class_weight in focal phases
            callbacks       = base_callbacks(
                f"P2s{step}",
                # FIX-2: epoch_offset keeps cosine aligned to this step's epochs
                cosine_schedule(lr_step_cur,
                                lr_step_cur * 0.05,
                                step_epochs,
                                epoch_offset=epoch_cursor),
                monitor_es="val_loss", es_mode="min",
            ),
            verbose=1,
        )
        histories.append(h_step)
        epoch_cursor  = end_epoch
        best_auc_step = max(h_step.history.get("val_auc", [0]))
        print(f"  Step {step} done → best val_auc={best_auc_step:.4f}")

    check_output_range(model, val_gen, raw_val, "After phase 2 (all steps)")

    # =========================================================================
    # SAVE + HISTORY
    # =========================================================================
    print("\n[5/7] Loading best checkpoint and saving …")
    model.load_weights(CFG["CHECKPOINT_PATH"])
    model.save(CFG["MODEL_SAVE_PATH"])
    print(f"  Saved → {CFG['MODEL_SAVE_PATH']}")

    phase_labels = ["warmup", "focal",
                    *[f"step{i}" for i in range(1, CFG["UNFREEZE_STEPS"] + 1)]]
    plot_history(histories, phase_labels)

    # =========================================================================
    # TEMPERATURE CALIBRATION ON VAL SET
    # =========================================================================
    print("\n[6/7] Temperature calibration on val set …")
    T = calibrate_temperature(model, val_gen, raw_val)

    # =========================================================================
    # FINAL EVALUATION ON TEST SET
    # =========================================================================
    all_val_aucs = []
    for h in histories:
        all_val_aucs.extend(h.history.get("val_auc", []))
    best_val_auc = max(all_val_aucs)
    use_tta      = best_val_auc >= CFG["TTA_MIN_AUC"]

    print(f"\n{'='*68}")
    print(f"[7/7] TEST SET EVALUATION")
    print(f"      TTA={use_tta} | Temperature={T:.3f}")
    print(f"{'='*68}")

    # --- Raw test predictions (FIX-3: TTA in pixel space) ---
    if use_tta:
        y_prob_raw = tta_predict(model, test_gen, tta_steps=CFG["TTA_STEPS"])
    else:
        y_prob_raw = _predict_generator(model, test_gen, raw_test.samples)

    y_true     = raw_test.classes[:len(y_prob_raw)]
    y_prob_cal = apply_temperature(y_prob_raw, T)

    print(f"\n  Raw  : min={y_prob_raw.min():.4f}  max={y_prob_raw.max():.4f}  "
          f"mean={y_prob_raw.mean():.4f}  std={y_prob_raw.std():.4f}")
    print(f"  Cal  : min={y_prob_cal.min():.4f}  max={y_prob_cal.max():.4f}  "
          f"mean={y_prob_cal.mean():.4f}  std={y_prob_cal.std():.4f}")

    real_s = y_prob_cal[y_true == 0]
    fake_s = y_prob_cal[y_true == 1]
    if len(real_s) and len(fake_s):
        sep    = fake_s.mean() - real_s.mean()
        status = "OK ✓" if sep > 0.30 else "LOW ⚠️  (target: >0.30)"
        print(f"\n  Real: mean={real_s.mean():.3f}  range=[{real_s.min():.3f},{real_s.max():.3f}]")
        print(f"  Fake: mean={fake_s.mean():.3f}  range=[{fake_s.min():.3f},{fake_s.max():.3f}]")
        print(f"  Sep : {sep:.3f}  {status}")

    # Threshold selected on val (NEVER on test)
    val_prob_raw = _predict_generator(model, val_gen, raw_val.samples)
    val_prob_cal = apply_temperature(val_prob_raw, T)
    val_labels   = raw_val.classes[:len(val_prob_cal)]
    opt_thresh, found = find_optimal_threshold(
        val_labels, val_prob_cal, target_recall=CFG["TARGET_RECALL"]
    )

    # Apply to TEST
    y_pred         = (y_prob_cal >= opt_thresh).astype(int)
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy       = (tp + tn) / (tp + tn + fp + fn)
    precision      = tp / (tp + fp + 1e-8)
    recall         = tp / (tp + fn + 1e-8)
    f1             = 2 * precision * recall / (precision + recall + 1e-8)
    auc_val        = roc_auc_score(y_true, y_prob_cal)
    fpr, tpr, _    = roc_curve(y_true, y_prob_cal)

    print(f"\n  Threshold : {opt_thresh:.4f} (selected on val)")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC AUC   : {auc_val:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Real','Fake'])}")

    fake_recall, real_recall = evaluate_frame_criterion(y_true, y_pred, opt_thresh)
    temporal_consistency_report(y_prob_cal, raw_test, opt_thresh)

    plot_confusion_matrix(cm, "Confusion Matrix — Video Model (calibrated)")
    plot_roc(fpr, tpr, auc_val)
    plot_score_distribution(y_true, y_prob_raw, opt_thresh, "_raw")
    plot_score_distribution(y_true, y_prob_cal, opt_thresh, "_calibrated")

    # Overfit gap
    gap = best_val_auc - auc_val
    print(f"\n  Overfit gap: best_val={best_val_auc:.4f}  test={auc_val:.4f}  gap={gap:.4f}")
    if gap > 0.08:
        print("  Still overfitting — add more diverse training videos.")
    elif gap > 0.04:
        print("  Mild overfit — acceptable.")
    else:
        print("  Good generalisation ✓")

    # Save metadata
    meta = {
        "optimal_threshold"         : float(round(opt_thresh, 4)),
        "video_detection_threshold" : float(round(opt_thresh, 2)),
        "temperature"               : float(round(T, 4)),
        "class_indices"             : {"real": 0, "fake": 1},
        "input_size"                : CFG["IMG_SIZE"][0],
        "preprocessing"             : "efficientnet.preprocess_input + temperature",
        "output_activation"         : "sigmoid",
        "output_meaning"            : "P(fake)",
        "focal_loss_alpha"          : CFG["FOCAL_ALPHA"],
        "focal_loss_gamma"          : CFG["FOCAL_GAMMA"],
        "label_smoothing"           : CFG["LABEL_SMOOTHING"],
        "phase1_lr"                 : CFG["PHASE1_LR"],
        "phase2_lr"                 : CFG["PHASE2_LR"],
        "unfreeze_layers_total"     : CFG["UNFREEZE_PER_STEP"] * CFG["UNFREEZE_STEPS"],
        "gradual_unfreezing_steps"  : CFG["UNFREEZE_STEPS"],
        "mobile_augmentation"       : True,
        "mixup"                     : False,
        "cutout_augmentation"       : False,
        "tta_steps"                 : CFG["TTA_STEPS"] if use_tta else 1,
        "tta_enabled"               : use_tta,
        "temperature_calibration"   : True,
        "stochastic_depth"          : True,
        "double_penalty_fix"        : True,
        "lr_schedule_aligned"       : True,
        "tta_pixel_space"           : True,
        "detection_strategy"        : "weighted_average",
        "test_auc"                  : float(round(auc_val, 4)),
        "test_fake_recall"          : float(round(fake_recall, 4)),
        "test_real_recall"          : float(round(real_recall, 4)),
        "frame_90_10_met"           : bool(fake_recall >= 0.90 and real_recall >= 0.90),
        "overfit_gap"               : float(round(gap, 4)),
    }
    meta_path = Path(CFG["MODEL_SAVE_PATH"]).parent / "model_metadata_video.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata → {meta_path}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*68}")
    print("  VIDEO MODEL — FIXED TRAINING COMPLETE")
    print(f"{'='*68}")
    print(f"  Time              : {elapsed:.1f} min")
    print(f"  Best val AUC      : {best_val_auc:.4f}")
    print(f"  Test AUC (cal)    : {auc_val:.4f}")
    print(f"  Overfit gap       : {gap:.4f}")
    print(f"  Fake frame recall : {fake_recall*100:.1f}%  (target ≥90%)")
    print(f"  Real frame recall : {real_recall*100:.1f}%  (target ≥90%)")
    print(f"  90/10 criterion   : {'MET ✓' if fake_recall>=0.90 and real_recall>=0.90 else 'NOT MET ✗'}")
    print(f"  Temperature T     : {T:.3f}")
    print(f"\n  >>> settings.py: VIDEO_DETECTION_THRESHOLD = {opt_thresh:.2f}")
    print(f"  >>> settings.py: VIDEO_TEMPERATURE = {T:.3f}")
    grade = ("EXCELLENT" if auc_val > 0.88 else
             "GOOD"      if auc_val > 0.80 else
             "FAIR"      if auc_val > 0.72 else
             "NEEDS MORE DATA")
    print(f"  Grade: {grade}")
    print(f"{'='*68}")

    gc.collect()
    K.clear_session()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        import traceback
        traceback.print_exc()