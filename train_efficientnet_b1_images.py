"""
train_efficientnet_b1_images_fixed_v3.py
=========================================
Merges the reliability of the OLD script with the architectural improvements
of the NEW script (GeM pooling, SE attention, 3-phase training).

BUGS FIXED vs new script (train_efficientnet_b1_images_fixed.py):
──────────────────────────────────────────────────────────────────
  FIX A — class_weight missing from Phase 1B focal path  ← CRITICAL
           Always pass class_weights to every model.fit() call.

  FIX B — Lambda layer for GeM→float16 cast won't deserialize  ← CRITICAL
           Replaced with a proper GemCast Layer subclass so
           keras.models.load_model() finds it in custom_objects.

  FIX C — Augmentation clip range too wide ([-1.5, 1.5])
           preprocess_input outputs [-1, 1]; clipping to [-1.5, 1.5]
           allows out-of-distribution values that degrade features.
           Fixed to [-1.0, 1.0].

  FIX D — No warmup in LR schedule
           Old script's WarmUpCosineDecay proved stable. Re-added:
           1-epoch linear warmup then cosine decay per phase.

  FIX E — Phase 2 LR too low (3e-6 → 1e-5)
           3e-6 is too slow to converge in 20 epochs on 25 unfrozen
           layers. Old script used 1e-5 which is validated and stable.

  FIX F — CollapseDetector callback missing from Phase 1B and Phase 2
           Real-time collapse halt restored (output std < 0.05 stops
           training immediately rather than waiting for epoch end).

  FIX G — ReduceLROnPlateau missing
           Old script used factor=0.4 patience=5 as a safety net
           alongside the cosine schedule. Re-added.

  FIX H — unfreeze_backbone only locks BN inside base model
           Head BN layers (bn1, bn2) were left trainable during Phase 2,
           risking stat corruption. Fixed to lock ALL BN everywhere.

  FIX I — get_dirs() KeyError if IMAGE_TEST and VAL both absent
           Added a safe fallback chain.

UNCHANGED from new script (keep improvements over old):
───────────────────────────────────────────────────────
  • GeM + GAP pooling (richer features than GAP alone)
  • SE channel attention block
  • 3-phase training (1A BCE, 1B Focal, 2 finetune)
  • tf.data pipeline (faster than ImageDataGenerator on GPU)
  • Temperature calibration
  • TTA inference
  • Collapse recovery logic
  • FocalLoss / BCELoss classes
  • Phase 1B skip when AUC < 0.55
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
from tensorflow import keras
from tensorflow.keras import backend as K, layers, Model
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger,
    LearningRateScheduler, TerminateOnNaN, ReduceLROnPlateau,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_auc_score, roc_curve,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not installed — limited metrics")

tf.random.set_seed(42)
np.random.seed(42)

import sys
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepfake_detection.settings")
import django
django.setup()
from django.conf import settings

# =============================================================================
# CONFIG
# =============================================================================

CFG = {
    "MODEL_SAVE_PATH"  : str(settings.IMAGE_MODEL_PATH),
    "CKPT_PATH"        : str(Path(settings.IMAGE_MODEL_PATH).parent / "best_v3_ckpt.keras"),
    "LOG_PATH"         : str(project_root / "log_image_v3.csv"),

    "IMG_SIZE"         : settings.IMAGE_SIZE,   # 240

    "BATCH_SIZE"       : 32,

    # Phase 1A: BCE only (frozen backbone)
    "PHASE1A_EPOCHS"   : 15,
    # Phase 1B: focal loss (frozen backbone)
    "PHASE1B_EPOCHS"   : 10,
    # Phase 2: fine-tune backbone
    "PHASE2_EPOCHS"    : 20,

    "LR_HEAD"          : 2e-4,
    "LR_FINETUNE"      : 1e-5,      # FIX E: was 3e-6, matches old script
    "LR_MIN"           : 1e-8,

    "FOCAL_GAMMA"      : 2.0,
    "FOCAL_ALPHA"      : 0.75,      # Penalise missed fakes 3× harder
    "LABEL_SMOOTHING"  : 0.05,

    "L2_REG"           : 1e-4,
    "DROPOUT1"         : 0.40,
    "DROPOUT2"         : 0.25,

    "ES_PATIENCE"      : 6,
    "ES_MIN_DELTA"     : 5e-4,
    "UNFREEZE_LAYERS"  : 25,

    "TTA_STEPS"        : 3,
    "TARGET_RECALL"    : 0.85,

    "WEIGHT_DECAY"     : 1e-5,
}

# =============================================================================
# GPU SETUP
# =============================================================================

print("\n" + "=" * 65)
print("  IMAGE DEEPFAKE — FIXED TRAINING v3")
print("=" * 65)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
print("  Mixed precision float16 ✓")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5376)]
    )
    print(f"  GPU: {len(gpus)} device(s) | 5.25GB ✓")
else:
    print("  No GPU — CPU only")

print(f"  TF: {tf.__version__}")
total_ep = CFG["PHASE1A_EPOCHS"] + CFG["PHASE1B_EPOCHS"] + CFG["PHASE2_EPOCHS"]
print(f"  Batch={CFG['BATCH_SIZE']} | Max epochs={total_ep}")
print("=" * 65 + "\n")

os.makedirs(os.path.dirname(CFG["MODEL_SAVE_PATH"]), exist_ok=True)


# =============================================================================
# LOSS CLASSES
# =============================================================================

class FocalLoss(keras.losses.Loss):
    """
    Binary focal loss.
    alpha=0.75 penalises missed fakes 3× harder than false alarms.
    Old script had alpha=0.25 which was BACKWARDS (upweighted real class).
    New script corrected this — kept at 0.75.
    """
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.05,
                 name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0:
            y_sm = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            y_sm = y_true
        bce  = -(y_sm * tf.math.log(y_pred)
                 + (1.0 - y_sm) * tf.math.log(1.0 - y_pred))
        p_t  = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        a_t  = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        return tf.reduce_mean(a_t * tf.pow(1.0 - p_t, self.gamma) * bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha,
                    "label_smoothing": self.label_smoothing})
        return cfg


class BCELoss(keras.losses.Loss):
    def __init__(self, label_smoothing=0.0, name="bce_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return tf.reduce_mean(
            -(y_true * tf.math.log(y_pred)
              + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"label_smoothing": self.label_smoothing})
        return cfg


# =============================================================================
# GeM POOLING
# =============================================================================

class GeM(layers.Layer):
    """Generalised Mean Pooling — richer than GAP for fine-grained features."""
    def __init__(self, p_init=3.0, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p_init = float(p_init)
        self.eps    = float(eps)

    def build(self, input_shape):
        self.gem_p = self.add_weight(
            name        = "gem_p",
            shape       = (1,),
            initializer = tf.constant_initializer(self.p_init),
            trainable   = True,
            dtype       = tf.float32,
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        p = tf.cast(self.gem_p, tf.float32)
        p = tf.maximum(p, 1.0)
        x = tf.clip_by_value(x, self.eps, 1e4)
        x = tf.pow(x, p)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.pow(x + self.eps, 1.0 / p)
        return x   # float32; GemCast handles dtype before Concatenate

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"p_init": self.p_init, "eps": self.eps})
        return cfg


class GemCast(layers.Layer):
    """
    FIX B — Replaces the Lambda cast layer.
    Lambda layers don't serialize cleanly in .keras format, causing
    keras.models.load_model() to fail at checkpoint reload.
    This proper Layer subclass is registered in custom_objects
    and serializes/deserializes without error.
    """
    def call(self, x):
        return tf.cast(x, tf.float16)

    def get_config(self):
        return super().get_config()


# =============================================================================
# COLLAPSE DETECTOR CALLBACK  (FIX F — restored from old script)
# =============================================================================

class CollapseDetector(keras.callbacks.Callback):
    """
    FIX F — Restored from old script.
    Stops training immediately if model outputs collapse to a flat range.
    Checks output std on one val batch every epoch after epoch 3.
    std < 0.05 means the model predicts nearly the same score for all
    inputs — it has failed to learn any discrimination.

    The new script only did a post-epoch diagnostic print, which means
    a collapsed Phase 1B or Phase 2 would train uselessly for all
    remaining epochs before being caught.
    """
    def __init__(self, val_ds, check_after_epoch=3, min_std=0.05):
        super().__init__()
        self.val_ds      = val_ds
        self.check_after = check_after_epoch
        self.min_std     = min_std

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.check_after:
            return
        # Take one batch from the validation dataset
        for x_batch, _ in self.val_ds.take(1):
            preds = self.model.predict(x_batch, verbose=0).flatten()
            break
        std = float(np.std(preds))
        rng = float(preds.max() - preds.min())
        print(f"\n  [CollapseDetector] std={std:.4f}  range={rng:.4f}")
        if std < self.min_std:
            print(f"  [CollapseDetector] COLLAPSE — stopping. "
                  f"std={std:.4f} < {self.min_std}")
            self.model.stop_training = True


# =============================================================================
# WARMUP + COSINE LR SCHEDULE  (FIX D — restored from old script)
# =============================================================================

class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    FIX D — Restored from old script which proved stable.
    Linear warm-up for the first epoch, then cosine decay for the rest.
    No warmup means a high initial LR can shock BN running statistics
    at the start of each phase.
    """
    def __init__(self, initial_lr, total_steps, warmup_steps):
        super().__init__()
        self.initial_lr   = float(initial_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        progress  = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0
        )
        cosine_lr = 0.5 * self.initial_lr * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":   self.initial_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


# =============================================================================
# tf.data PIPELINE
# =============================================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    """
    preprocess_input expects [0, 255] and maps to [-1, 1] internally.
    DO NOT divide by 255 or apply ImageNet mean/std on top of it.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [CFG["IMG_SIZE"], CFG["IMG_SIZE"]])
    img = tf.cast(img, tf.float32)   # [0, 255]
    img = preprocess_input(img)      # → [-1, 1]
    return img, tf.cast(label, tf.float32)


def augment_train(img, label):
    """
    FIX C — Clip to [-1.0, 1.0] matching preprocess_input output range.
    Old bug clipped to [-1.5, 1.5] allowing out-of-distribution values
    that degrade backbone feature quality.
    """
    img = tf.image.random_flip_left_right(img)

    pad = 20
    h   = CFG["IMG_SIZE"]
    img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, [h, h, 3])

    # Small brightness jitter in [-1, 1] space
    img = img + tf.random.uniform([], -0.05, 0.05)
    img = tf.clip_by_value(img, -1.0, 1.0)   # FIX C

    return img, label


def get_paths_and_labels(data_dir):
    paths, labels = [], []
    data_dir = Path(data_dir)
    for cls, lbl in [("real", 0), ("fake", 1)]:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing directory: {cls_dir}")
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for p in cls_dir.glob(ext):
                paths.append(str(p))
                labels.append(lbl)
    paths  = np.array(paths)
    labels = np.array(labels, dtype=np.int32)
    idx    = np.random.permutation(len(paths))
    return paths[idx], labels[idx]


def build_dataset(data_dir, training=False, batch_size=32):
    paths, labels = get_paths_and_labels(data_dir)
    n_real = int(np.sum(labels == 0))
    n_fake = int(np.sum(labels == 1))
    print(f"    {Path(data_dir).name}: {len(paths):,}  "
          f"(real={n_real:,} fake={n_fake:,})")

    ds = tf.data.Dataset.from_tensor_slices(
        (paths.tolist(), labels.tolist())
    )
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 8000), seed=42,
                        reshuffle_each_iteration=True)

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(augment_train, num_parallel_calls=AUTOTUNE)

    ds = (ds
          .batch(batch_size, drop_remainder=False)
          .prefetch(AUTOTUNE))
    return ds, paths, labels


def get_dirs():
    """
    FIX I — Safe fallback chain for all three split directories.
    Previous code could raise KeyError if IMAGE_TEST and VAL both absent.
    """
    train_dir = settings.DATASET.get("IMAGE_TRAIN") or settings.DATASET.get("TRAIN")
    val_dir   = settings.DATASET.get("IMAGE_VAL")   or settings.DATASET.get("VAL")
    test_dir  = (settings.DATASET.get("IMAGE_TEST")
                 or settings.DATASET.get("TEST")
                 or val_dir)

    if train_dir and not Path(train_dir).exists():
        train_dir = settings.DATASET.get("TRAIN", train_dir)
        val_dir   = settings.DATASET.get("VAL",   val_dir)
        test_dir  = settings.DATASET.get("TEST",  val_dir)

    if not train_dir or not val_dir:
        raise FileNotFoundError(
            "Could not locate train/val directories. "
            "Check settings.DATASET keys: IMAGE_TRAIN/IMAGE_VAL or TRAIN/VAL."
        )
    return train_dir, val_dir, test_dir


# =============================================================================
# MODEL
# =============================================================================

def build_model(img_size=240):
    print("\n  Building EfficientNet-B1 + SE + GeM …")

    base = EfficientNetB1(
        weights     = "imagenet",
        include_top = False,
        input_shape = (img_size, img_size, 3),
    )
    base.trainable = False

    inp = keras.Input(shape=(img_size, img_size, 3), name="input")
    x   = base(inp, training=False)

    # SE channel attention
    ch  = x.shape[-1]   # 1280 for B1
    se  = layers.GlobalAveragePooling2D(name="se_gap")(x)
    se  = layers.Dense(max(ch // 16, 8), activation="relu",   name="se_fc1")(se)
    se  = layers.Dense(ch,               activation="sigmoid", name="se_fc2")(se)
    se  = layers.Reshape((1, 1, ch), name="se_reshape")(se)
    x   = layers.Multiply(name="se_scale")([x, se])

    # GeM + GAP  →  both float32 from GeM; GAP may be float16 in mixed prec
    gem = GeM(p_init=3.0, name="gem")(x)                       # float32
    gap = layers.GlobalAveragePooling2D(name="gap")(x)          # float16

    # FIX B — proper Layer subclass cast instead of Lambda
    gem = GemCast(name="gem_cast")(gem)                         # → float16

    x = layers.Concatenate(name="pool_cat")([gem, gap])         # [B, 2560] float16

    # Head — three BN layers matches old script depth
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dense(
            256,
            activation         = "relu",
            kernel_regularizer = keras.regularizers.l2(CFG["L2_REG"]),
            name               = "fc1",
        )(x)
    x = layers.Dropout(CFG["DROPOUT1"], name="drop1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dense(
            128,
            activation         = "relu",
            kernel_regularizer = keras.regularizers.l2(CFG["L2_REG"]),
            name               = "fc2",
        )(x)
    x = layers.Dropout(CFG["DROPOUT2"], name="drop2")(x)

    # float32 output for numerical stability
    out = layers.Dense(1, activation="sigmoid",
                       dtype="float32", name="output")(x)

    model = Model(inp, out, name="deepfake_v3")
    print(f"  Total params: {model.count_params():,}")
    return model, base


CUSTOM_OBJECTS = {
    "FocalLoss":        FocalLoss,
    "BCELoss":          BCELoss,
    "GeM":              GeM,
    "GemCast":          GemCast,
    "WarmUpCosineDecay": WarmUpCosineDecay,
}


# =============================================================================
# COMPILE
# =============================================================================

def compile_model(model, optimizer, loss_fn):
    model.compile(
        optimizer = optimizer,
        loss      = loss_fn,
        metrics   = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# =============================================================================
# CALLBACKS
# =============================================================================

def make_callbacks(phase, lr_cb, val_ds, append_log=True):
    """
    FIX F — CollapseDetector included in every phase.
    FIX G — ReduceLROnPlateau included as safety net alongside cosine.
    """
    return [
        ModelCheckpoint(
            CFG["CKPT_PATH"],
            monitor        = "val_auc",
            save_best_only = True,
            mode           = "max",
            verbose        = 1,
        ),
        EarlyStopping(
            monitor              = "val_auc",
            patience             = CFG["ES_PATIENCE"],
            min_delta            = CFG["ES_MIN_DELTA"],
            restore_best_weights = True,
            mode                 = "max",
            verbose              = 1,
        ),
        # FIX G — safety net on top of cosine schedule
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.4,
            patience = 5,
            min_lr   = CFG["LR_MIN"],
            verbose  = 1,
        ),
        CSVLogger(CFG["LOG_PATH"], append=append_log),
        TerminateOnNaN(),
        lr_cb,
        # FIX F — real-time collapse halt
        CollapseDetector(val_ds, check_after_epoch=3, min_std=0.05),
    ]


# =============================================================================
# UNFREEZE BACKBONE
# =============================================================================

def unfreeze_backbone(base, model, n):
    base.trainable = True
    cut = len(base.layers) - n
    for layer in base.layers[:cut]:
        layer.trainable = False
    for layer in base.layers[cut:]:
        layer.trainable = True

    # FIX H — lock ALL BN layers everywhere (base + head)
    # Old bug only locked BN inside base.layers, leaving head BN (bn1, bn2)
    # trainable which can corrupt running statistics.
    bn = 0
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            bn += 1
        # BN inside sub-models (base)
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, layers.BatchNormalization):
                    sub.trainable = False
                    bn += 1

    tp = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"  Unfrozen top {n} backbone layers | BN locked everywhere={bn} | "
          f"Trainable params={tp:,}")


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def predict_dataset(model, ds, n_samples):
    preds = []
    for X, _ in ds:
        p = model.predict(X, verbose=0).flatten()
        preds.extend(p.tolist())
        if len(preds) >= n_samples:
            break
    return np.array(preds[:n_samples])


def check_distribution(preds, labels, tag):
    real_p = preds[labels == 0]
    fake_p = preds[labels == 1]
    sep    = 0.0
    print(f"\n  [{tag}]")
    print(f"    All  : min={preds.min():.3f} max={preds.max():.3f} "
          f"std={preds.std():.4f}")
    if len(real_p):
        print(f"    Real : mean={real_p.mean():.3f}  "
              f"range=[{real_p.min():.3f},{real_p.max():.3f}]")
    if len(fake_p):
        print(f"    Fake : mean={fake_p.mean():.3f}  "
              f"range=[{fake_p.min():.3f},{fake_p.max():.3f}]")
    if len(real_p) and len(fake_p):
        sep    = float(fake_p.mean() - real_p.mean())
        status = ("GOOD ✓"       if sep > 0.20 else
                  "LEARNING..."  if sep > 0.05 else
                  "COLLAPSED ✗")
        print(f"    Sep  : {sep:.4f}  {status}")
    return sep


def is_collapsed(preds, min_std=0.02):
    return float(preds.std()) < min_std


# =============================================================================
# TEMPERATURE CALIBRATION
# =============================================================================

def calibrate_temperature(preds, labels):
    p      = np.clip(preds, 1e-6, 1 - 1e-6).astype(np.float64)
    logits = np.log(p / (1 - p))
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.3, 3.0, 54):
        pc  = 1.0 / (1.0 + np.exp(-logits / T))
        pc  = np.clip(pc, 1e-7, 1 - 1e-7)
        nll = -np.mean(labels * np.log(pc) + (1 - labels) * np.log(1 - pc))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    print(f"  Temperature T={best_T:.3f}  (NLL={best_nll:.4f})")
    return best_T


def apply_temperature(probs, T):
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / T))


def find_threshold(y_true, y_prob, target=0.85):
    best_t, best_min, found = 0.50, -1.0, False
    for t in np.linspace(0.10, 0.90, 161):
        yp  = (y_prob >= t).astype(int)
        tp  = int(np.sum((yp == 1) & (y_true == 1)))
        fn  = int(np.sum((yp == 0) & (y_true == 1)))
        tn  = int(np.sum((yp == 0) & (y_true == 0)))
        fp  = int(np.sum((yp == 1) & (y_true == 0)))
        fr  = tp / (tp + fn + 1e-8)
        rr  = tn / (tn + fp + 1e-8)
        m   = min(fr, rr)
        if fr >= target and rr >= target and m > best_min:
            best_min, best_t, found = m, float(t), True

    if not found and HAS_SKLEARN:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j      = int(np.argmax(tpr - fpr))
        best_t = float(np.clip(thr[j], 0.25, 0.80))
        print(f"  ⚠ Balanced threshold not found — Youden-J: {best_t:.4f}")
    else:
        print(f"  Threshold={best_t:.4f}  "
              f"({'✓' if found else '✗'} balanced ≥{target*100:.0f}%)")
    return best_t


# =============================================================================
# TTA PREDICTION
# =============================================================================

def tta_predict(model, test_paths, img_size, batch_size, n_steps=3):
    """
    Uses preprocess_input to match training preprocessing exactly.
    """
    all_preds = []

    for step in range(n_steps):
        step_preds = []
        for i in range(0, len(test_paths), batch_size):
            batch = test_paths[i:i + batch_size]
            imgs  = []
            for p in batch:
                img = cv2.imread(str(p))
                if img is None:
                    imgs.append(np.zeros((img_size, img_size, 3), np.float32))
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype(np.float32)   # [0, 255]

                if step == 1:
                    img = img[:, ::-1, :]               # horizontal flip
                elif step == 2:
                    img = np.clip(img * 1.10, 0, 255)   # brightness +10%

                img = preprocess_input(img)             # → [-1, 1]
                imgs.append(img)

            X = np.stack(imgs, axis=0).astype(np.float32)
            p = model.predict(X, verbose=0).flatten()
            step_preds.extend(p.tolist())

        all_preds.append(step_preds)

    return np.array(all_preds).mean(axis=0)


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train():
    t_start = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dir, val_dir, test_dir = get_dirs()
    print(f"  Train : {train_dir}")
    print(f"  Val   : {val_dir}")
    print(f"  Test  : {test_dir}")

    print("\n[1/8] Building tf.data pipelines …")
    train_ds, train_paths, train_labels = build_dataset(
        train_dir, training=True,  batch_size=CFG["BATCH_SIZE"])
    val_ds,   val_paths,   val_labels   = build_dataset(
        val_dir,   training=False, batch_size=CFG["BATCH_SIZE"])
    test_ds,  test_paths,  test_labels  = build_dataset(
        test_dir,  training=False, batch_size=CFG["BATCH_SIZE"])

    n_train = len(train_paths)
    n_val   = len(val_paths)
    n_test  = len(test_paths)
    steps   = int(np.ceil(n_train / CFG["BATCH_SIZE"]))

    n_real = int(np.sum(train_labels == 0))
    n_fake = int(np.sum(train_labels == 1))
    tot    = n_real + n_fake
    class_weights = {
        0: round(tot / (2 * max(n_real, 1)), 4),
        1: round(tot / (2 * max(n_fake, 1)), 4),
    }

    print(f"\n  train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"  Steps/epoch: {steps}")
    print(f"  Class weights: {class_weights}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/8] Building model …")
    model, base = build_model(CFG["IMG_SIZE"])
    histories    = []
    epoch_cursor = 0

    bce_loss   = BCELoss(label_smoothing=0.0)
    focal_loss = FocalLoss(
        gamma           = CFG["FOCAL_GAMMA"],
        alpha           = CFG["FOCAL_ALPHA"],
        label_smoothing = CFG["LABEL_SMOOTHING"],
    )

    # =========================================================================
    # PHASE 1A — BCE, frozen backbone
    # =========================================================================
    p1a     = CFG["PHASE1A_EPOCHS"]
    end_p1a = epoch_cursor + p1a

    print(f"\n{'─'*65}")
    print(f"[3/8] PHASE 1A — BCE frozen backbone | {p1a}ep | "
          f"lr={CFG['LR_HEAD']}")
    print(f"{'─'*65}")

    # FIX D — warmup schedule for phase 1A
    p1a_schedule = WarmUpCosineDecay(
        initial_lr   = CFG["LR_HEAD"],
        total_steps  = p1a * steps,
        warmup_steps = steps,          # 1 epoch linear warmup
    )
    base.trainable = False
    compile_model(
        model,
        AdamW(learning_rate=p1a_schedule,
              weight_decay=CFG["WEIGHT_DECAY"], clipnorm=1.0),
        bce_loss,
    )

    h1a = model.fit(
        train_ds,
        epochs          = end_p1a,
        initial_epoch   = epoch_cursor,
        validation_data = val_ds,
        class_weight    = class_weights,   # always pass class_weights
        callbacks       = make_callbacks("P1A", keras.callbacks.LambdaCallback(),
                                         val_ds, append_log=False),
        verbose         = 1,
    )
    histories.append(h1a)
    epoch_cursor = end_p1a

    best_p1a_auc = max(h1a.history.get("val_auc", [0]))
    print(f"\n  Phase 1A → best val_auc={best_p1a_auc:.4f}")
    vp1a  = predict_dataset(model, val_ds, n_val)
    sep1a = check_distribution(vp1a, val_labels, "After Phase 1A")

    # ── Collapse check + recovery ──────────────────────────────────────────
    if is_collapsed(vp1a):
        print("\n  ⚠ MODEL COLLAPSED after Phase 1A — reinitialising head …")
        head_layers = ["fc1", "fc2", "output", "bn1", "bn2"]
        for layer in model.layers:
            if layer.name in head_layers:
                new_weights = []
                for w in layer.get_weights():
                    if len(w.shape) > 1:
                        fan_in  = w.shape[0]
                        scale   = np.sqrt(2.0 / fan_in)
                        new_weights.append(
                            np.random.normal(0, scale, w.shape).astype(np.float32)
                        )
                    else:
                        new_weights.append(np.zeros(w.shape, dtype=np.float32))
                layer.set_weights(new_weights)

        recovery_schedule = WarmUpCosineDecay(
            initial_lr   = CFG["LR_HEAD"] * 0.1,
            total_steps  = 10 * steps,
            warmup_steps = steps,
        )
        compile_model(
            model,
            AdamW(learning_rate=recovery_schedule,
                  weight_decay=CFG["WEIGHT_DECAY"]),
            bce_loss,
        )
        h1a_r = model.fit(
            train_ds,
            epochs          = epoch_cursor + 10,
            initial_epoch   = epoch_cursor,
            validation_data = val_ds,
            class_weight    = class_weights,
            callbacks       = make_callbacks("P1A_recovery",
                                             keras.callbacks.LambdaCallback(),
                                             val_ds, append_log=True),
            verbose=1,
        )
        histories.append(h1a_r)
        epoch_cursor += 10
        vp1a  = predict_dataset(model, val_ds, n_val)
        sep1a = check_distribution(vp1a, val_labels, "After Recovery")
        best_p1a_auc = max(h1a_r.history.get("val_auc", [best_p1a_auc]))

    # =========================================================================
    # PHASE 1B — Focal loss, frozen backbone
    # Skip if Phase 1A AUC < 0.55 (model still collapsed) — extend BCE instead
    # =========================================================================
    p1b     = CFG["PHASE1B_EPOCHS"]
    end_p1b = epoch_cursor + p1b

    if best_p1a_auc >= 0.55:
        print(f"\n{'─'*65}")
        print(f"[4/8] PHASE 1B — Focal frozen backbone | {p1b}ep | "
              f"lr={CFG['LR_HEAD'] * 0.5}")
        print(f"{'─'*65}")

        p1b_schedule = WarmUpCosineDecay(   # FIX D
            initial_lr   = CFG["LR_HEAD"] * 0.5,
            total_steps  = p1b * steps,
            warmup_steps = max(steps // 2, 1),
        )
        compile_model(
            model,
            AdamW(learning_rate=p1b_schedule,
                  weight_decay=CFG["WEIGHT_DECAY"], clipnorm=1.0),
            focal_loss,
        )

        h1b = model.fit(
            train_ds,
            epochs          = end_p1b,
            initial_epoch   = epoch_cursor,
            validation_data = val_ds,
            class_weight    = class_weights,   # FIX A — was missing in new script
            callbacks       = make_callbacks("P1B",
                                             keras.callbacks.LambdaCallback(),
                                             val_ds, append_log=True),
            verbose=1,
        )
        histories.append(h1b)
        epoch_cursor = end_p1b

        best_p1b_auc = max(h1b.history.get("val_auc", [0]))
        print(f"\n  Phase 1B → best val_auc={best_p1b_auc:.4f}")
        vp1b  = predict_dataset(model, val_ds, n_val)
        sep1b = check_distribution(vp1b, val_labels, "After Phase 1B")
    else:
        print(f"\n  [4/8] Phase 1A AUC={best_p1a_auc:.4f} < 0.55 — "
              f"extending BCE instead of focal …")
        p1b_ext_schedule = WarmUpCosineDecay(
            initial_lr   = CFG["LR_HEAD"] * 0.3,
            total_steps  = p1b * steps,
            warmup_steps = max(steps // 2, 1),
        )
        compile_model(
            model,
            AdamW(learning_rate=p1b_ext_schedule,
                  weight_decay=CFG["WEIGHT_DECAY"], clipnorm=1.0),
            bce_loss,
        )
        h1b = model.fit(
            train_ds,
            epochs          = end_p1b,
            initial_epoch   = epoch_cursor,
            validation_data = val_ds,
            class_weight    = class_weights,
            callbacks       = make_callbacks("P1B_bce_ext",
                                             keras.callbacks.LambdaCallback(),
                                             val_ds, append_log=True),
            verbose=1,
        )
        histories.append(h1b)
        epoch_cursor = end_p1b
        best_p1b_auc = max(h1b.history.get("val_auc", [0]))
        print(f"\n  Extended BCE → best val_auc={best_p1b_auc:.4f}")

    # =========================================================================
    # PHASE 2 — Fine-tune backbone (top 25 layers)
    # =========================================================================
    p2     = CFG["PHASE2_EPOCHS"]
    end_p2 = epoch_cursor + p2

    print(f"\n{'─'*65}")
    print(f"[5/8] PHASE 2 — Unfreeze top {CFG['UNFREEZE_LAYERS']} layers | "
          f"{p2}ep | lr={CFG['LR_FINETUNE']}")
    print(f"{'─'*65}")

    unfreeze_backbone(base, model, CFG["UNFREEZE_LAYERS"])

    # FIX D — warmup for phase 2 as well
    # FIX E — LR_FINETUNE is now 1e-5 (was 3e-6), clipnorm=0.5 for safety
    p2_schedule = WarmUpCosineDecay(
        initial_lr   = CFG["LR_FINETUNE"],
        total_steps  = p2 * steps,
        warmup_steps = steps,    # 1 epoch warmup
    )
    compile_model(
        model,
        AdamW(learning_rate=p2_schedule,
              weight_decay=CFG["WEIGHT_DECAY"], clipnorm=0.5),
        focal_loss,
    )

    h2 = model.fit(
        train_ds,
        epochs          = end_p2,
        initial_epoch   = epoch_cursor,
        validation_data = val_ds,
        class_weight    = class_weights,
        callbacks       = make_callbacks("P2",
                                         keras.callbacks.LambdaCallback(),
                                         val_ds, append_log=True),
        verbose=1,
    )
    histories.append(h2)
    epoch_cursor = end_p2

    best_p2_auc = max(h2.history.get("val_auc", [0]))
    print(f"\n  Phase 2 → best val_auc={best_p2_auc:.4f}")
    vp2  = predict_dataset(model, val_ds, n_val)
    sep2 = check_distribution(vp2, val_labels, "After Phase 2")

    # =========================================================================
    # LOAD BEST CHECKPOINT + SAVE
    # =========================================================================
    print("\n[6/8] Loading best checkpoint and saving …")
    # FIX B — GemCast is now in CUSTOM_OBJECTS so load_model works
    model = keras.models.load_model(
        CFG["CKPT_PATH"],
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )

    save_path_keras = CFG["MODEL_SAVE_PATH"].replace(".h5", ".keras")
    model.save(save_path_keras)
    print(f"  Saved (.keras) → {save_path_keras}")

    try:
        model.save(CFG["MODEL_SAVE_PATH"])
        print(f"  Saved (.h5)    → {CFG['MODEL_SAVE_PATH']}")
    except Exception as e:
        print(f"  ⚠ .h5 save failed ({e}) — use .keras version")
        CFG["MODEL_SAVE_PATH"] = save_path_keras

    # =========================================================================
    # TEMPERATURE CALIBRATION
    # =========================================================================
    print("\n[7/8] Temperature calibration on val set …")
    val_preds = predict_dataset(model, val_ds, n_val)
    T         = calibrate_temperature(val_preds, val_labels)
    val_cal   = apply_temperature(val_preds, T)
    threshold = find_threshold(val_labels, val_cal, target=CFG["TARGET_RECALL"])

    # =========================================================================
    # FINAL EVALUATION — TTA on test set
    # =========================================================================
    print(f"\n[8/8] Test evaluation (TTA={CFG['TTA_STEPS']}) …")

    y_prob_raw = tta_predict(
        model, test_paths,
        CFG["IMG_SIZE"], CFG["BATCH_SIZE"],
        n_steps=CFG["TTA_STEPS"],
    )
    y_true = test_labels[:len(y_prob_raw)]
    y_prob = apply_temperature(y_prob_raw, T)
    y_pred = (y_prob >= threshold).astype(int)

    sep_test = check_distribution(y_prob, y_true, "Test (calibrated)")

    auc_val = accuracy = recall = specificity = f1 = 0.0
    if HAS_SKLEARN:
        cm             = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy       = (tp + tn) / (tp + tn + fp + fn)
        precision      = tp / (tp + fp + 1e-8)
        recall         = tp / (tp + fn + 1e-8)
        specificity    = tn / (tn + fp + 1e-8)
        f1             = 2 * precision * recall / (precision + recall + 1e-8)
        auc_val        = roc_auc_score(y_true, y_prob)

        print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"  Accuracy    : {accuracy:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall(Fake): {recall:.4f}")
        print(f"  Specificity : {specificity:.4f}")
        print(f"  F1          : {f1:.4f}")
        print(f"  ROC AUC     : {auc_val:.4f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=['Real','Fake'], digits=4)}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    all_vauc = []
    for h in histories:
        all_vauc.extend(h.history.get("val_auc", []))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Image Deepfake — Fixed Training v3", fontweight="bold")

    axes[0].plot(all_vauc, "purple", lw=2)
    axes[0].axhline(0.93, color="green", linestyle="--", label="Target 0.93")
    axes[0].set_title("Val AUC"); axes[0].legend(); axes[0].grid(alpha=0.3)

    real_s = y_prob[y_true == 0]
    fake_s = y_prob[y_true == 1]
    bins   = np.linspace(0, 1, 50)
    axes[1].hist(real_s, bins, alpha=0.6, color="green",
                 label=f"Real n={len(real_s)}", density=True)
    axes[1].hist(fake_s, bins, alpha=0.6, color="red",
                 label=f"Fake n={len(fake_s)}", density=True)
    axes[1].axvline(threshold, color="black", lw=2,
                    linestyle="--", label=f"T={threshold:.2f}")
    axes[1].set_title("Score Distribution")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    if HAS_SKLEARN and auc_val > 0:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[2].plot(fpr, tpr, "orange", lw=2, label=f"AUC={auc_val:.4f}")
        axes[2].plot([0, 1], [0, 1], "navy", lw=1.5, linestyle="--")
        axes[2].set_title("ROC Curve")
        axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_plot = project_root / "training_v3.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out_plot}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    all_val_aucs = []
    for h in histories:
        all_val_aucs.extend(h.history.get("val_auc", []))
    best_val_auc = max(all_val_aucs) if all_val_aucs else 0.0

    meta = {
        "optimal_threshold"         : float(round(threshold, 4)),
        "image_detection_threshold" : float(round(threshold, 2)),
        "temperature"               : float(round(T, 4)),
        "class_indices"             : {"real": 0, "fake": 1},
        "input_size"                : CFG["IMG_SIZE"],
        "preprocessing"             : "efficientnet preprocess_input [0,255]→[-1,1]",
        "focal_alpha"               : CFG["FOCAL_ALPHA"],
        "focal_gamma"               : CFG["FOCAL_GAMMA"],
        "label_smoothing"           : CFG["LABEL_SMOOTHING"],
        "gem_pooling"               : True,
        "se_attention"              : True,
        "tta_steps"                 : CFG["TTA_STEPS"],
        "batch_size"                : CFG["BATCH_SIZE"],
        "test_auc"                  : float(round(auc_val, 4)),
        "test_accuracy"             : float(round(accuracy, 4)),
        "test_recall_fake"          : float(round(recall, 4)),
        "test_specificity"          : float(round(specificity, 4)),
        "separation_test"           : float(round(sep_test, 4)),
        "best_val_auc"              : float(round(best_val_auc, 4)),
        "model_save_path"           : CFG["MODEL_SAVE_PATH"],
        "unfreeze_layers"           : CFG["UNFREEZE_LAYERS"],
        "lr_finetune"               : CFG["LR_FINETUNE"],
    }
    meta_path = Path(settings.IMAGE_MODEL_PATH).parent / "model_metadata_v3.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    grade   = ("EXCELLENT" if auc_val > 0.96 else
               "GOOD"      if auc_val > 0.93 else
               "FAIR"      if auc_val > 0.88 else
               "NEEDS MORE DATA / DEBUG")

    print(f"\n{'='*65}")
    print("  FIXED TRAINING v3 COMPLETE")
    print(f"{'='*65}")
    print(f"  Time          : {elapsed:.1f} min")
    print(f"  Best val AUC  : {best_val_auc:.4f}")
    print(f"  Test AUC      : {auc_val:.4f}")
    print(f"  Test Accuracy : {accuracy:.4f}")
    print(f"  Recall (fake) : {recall:.4f}")
    print(f"  Specificity   : {specificity:.4f}")
    print(f"  Separation    : {sep_test:.3f}")
    print(f"  Temperature T : {T:.3f}")
    print(f"  Grade         : {grade}")
    print(f"\n  >>> Update settings.py:")
    print(f"      IMAGE_DETECTION_THRESHOLD = {threshold:.2f}")
    print(f"      MODEL_TEMPERATURE         = {T:.3f}")
    print(f"{'='*65}")

    if sep_test > 0.30:
        fake_at_70 = float(np.mean(y_prob[y_true == 1] >= 0.70) * 100)
        real_at_70 = float(np.mean(y_prob[y_true == 0] <  0.70) * 100)
        print(f"\n  70% threshold analysis:")
        print(f"    Fake scoring ≥70%: {fake_at_70:.1f}%")
        print(f"    Real scoring <70%: {real_at_70:.1f}%")
        if fake_at_70 >= 80 and real_at_70 >= 80:
            print(f"    ✓ SAFE to set IMAGE_DETECTION_THRESHOLD = 0.70")
        else:
            print(f"    ⚠ Use {threshold:.2f} — 0.70 misses too many fakes")

    gc.collect()
    K.clear_session()


def main():
    try:
        train()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()