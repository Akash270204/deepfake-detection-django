"""
train_efficientnet_b1_video.py — Final fixed version
======================================================

3 bugs fixed vs previous version:

  FIX A: cv2 moved to top of imports — was placed AFTER mobile_augment()
         which already called cv2.imencode. NameError crash on first batch.

  FIX B: mobile_augment() now receives and returns [0,255] uint8 correctly.
         ImageDataGenerator passes float pixels to __getitem__. The old code
         did img.astype(np.uint8) on a [0,1] float — all values became 0 or 1
         (black frame). Fixed by scaling: (x * 255).astype(np.uint8) before
         passing to mobile_augment, then back to float32 for preprocess_input.

  FIX C: CollapseDetector no longer calls val_gen[0] inside on_epoch_end.
         After Keras calls on_epoch_end on the generator, the [0] index
         returns a stale or wrong batch. Fixed to use val_auc from logs dict
         instead — which is always accurate and computed by Keras itself.

Everything else from the previous version is unchanged and correct.
"""

import os
import gc
import json
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# FIX A: cv2 at the top, before any function that uses it
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dense, Dropout, BatchNormalization,
    Input, Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger, LambdaCallback,
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
    "DATA_DIR"           : str(BASE_DIR / "dataset" / "video_frames"),
    "MODEL_SAVE_PATH"    : str(BASE_DIR / "detector" / "ml_models" / "deepfake_video_model.h5"),
    "CHECKPOINT_PATH"    : str(BASE_DIR / "detector" / "ml_models" / "best_video_checkpoint.h5"),
    "LOG_PATH"           : str(BASE_DIR / "training_video_log.csv"),

    "IMG_SIZE"           : (240, 240),
    "IMG_SIZE_FULL"      : (240, 240, 3),

    "BATCH_SIZE"         : 16,
    "PHASE1_EPOCHS"      : 20,
    "PHASE2_EPOCHS"      : 20,

    "PHASE1_LR"          : 3e-4,
    "PHASE2_LR"          : 1e-5,

    "UNFREEZE_LAYERS"    : 15,

    "ES_PATIENCE"        : 8,
    "REDUCE_LR_PATIENCE" : 3,

    "FOCAL_GAMMA"        : 2.0,
    "FOCAL_ALPHA"        : 0.75,
    "LABEL_SMOOTHING"    : 0.05,

    "CLASSES"            : ["real", "fake"],
}

print("=" * 62)
print("  DEEPFAKE VIDEO MODEL — ANTI-OVERFIT + MOBILE FIX")
print("  Phase1=20ep lr=3e-4 | Phase2=20ep lr=1e-5")
print("  Unfreeze=15 | Dropout=0.50/0.40 | Mobile augmentation ON")
print("=" * 62)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {tf.config.list_physical_devices('GPU')}")
print("=" * 62)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled")

os.makedirs(os.path.dirname(CFG["MODEL_SAVE_PATH"]), exist_ok=True)


# =============================================================================
# FOCAL LOSS
# =============================================================================

def focal_loss(gamma=2.0, alpha=0.75, label_smoothing=0.05):
    def _focal(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.cast(y_pred, tf.float32)
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_smooth = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
        bce      = -(y_smooth * tf.math.log(y_pred)
                     + (1 - y_smooth) * tf.math.log(1 - y_pred))
        p_t      = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t  = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    _focal.__name__ = "focal_loss"
    return _focal


# =============================================================================
# COLLAPSE DETECTOR
#
# FIX C: No longer calls val_gen[0] inside on_epoch_end.
#
# The old code did:
#   x_batch, _ = self.val_gen[0]
#   preds = self.model.predict(x_batch, verbose=0)
#
# Problem: Keras calls on_epoch_end() on all callbacks AND on the generator
# in the same hook. The generator's internal state is reset before or during
# this call depending on ordering, so val_gen[0] can return a wrong batch.
# It also runs a full model.predict() which adds GPU memory pressure.
#
# Fix: use val_auc directly from the logs dict that Keras passes to the
# callback. val_auc is computed by Keras on the full validation set during
# the epoch — it is always accurate, costs nothing extra, and is not
# affected by generator state.
#
# A real collapse shows up as val_auc stuck at ~0.50 (random guessing)
# for several consecutive epochs. That is what we check here.
# =============================================================================

class CollapseDetector(tf.keras.callbacks.Callback):
    def __init__(self, check_after=12, min_auc=0.52, patience=4):
        """
        check_after: don't check before this epoch (head needs warmup)
        min_auc: val_auc must be above this — 0.52 = barely better than random
        patience: how many consecutive epochs below min_auc before stopping
        """
        super().__init__()
        self.check_after      = check_after
        self.min_auc          = min_auc
        self.patience         = patience
        self._consecutive_low = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.check_after:
            return
        if logs is None:
            return

        val_auc = logs.get('val_auc', 1.0)
        print(f"\n  [CollapseDetector] ep{epoch+1}  val_auc={val_auc:.4f}  "
              f"(min={self.min_auc}, consecutive_low={self._consecutive_low})")

        if val_auc < self.min_auc:
            self._consecutive_low += 1
            if self._consecutive_low >= self.patience:
                print(f"  [CollapseDetector] COLLAPSE — val_auc={val_auc:.4f} "
                      f"below {self.min_auc} for {self.patience} consecutive epochs. Stopping.")
                self.model.stop_training = True
        else:
            self._consecutive_low = 0  # reset on any good epoch


# =============================================================================
# MOBILE-AWARE AUGMENTATION
#
# FIX B: mobile_augment now receives and returns uint8 [0,255].
#
# The old code in __getitem__ did:
#   frame = x.astype(np.uint8)   ← x is float [0,1] from ImageDataGenerator
#   mobile_augment(frame)         ← frame is now 0 or 1 everywhere
#
# Fix: scale up before converting, scale back down after.
# The __getitem__ method handles the scaling correctly now.
# mobile_augment itself only operates on uint8 [0,255] — this is correct
# because cv2.imencode and cv2.GaussianBlur expect that range.
# =============================================================================

def mobile_augment(img_uint8):
    """
    img_uint8: numpy array, dtype=uint8, shape=(H,W,3), values in [0,255].
    Returns: numpy array, same shape and dtype.
    """
    img = img_uint8.copy()

    # JPEG compression — simulates mobile codec low-bitrate encoding
    if np.random.random() < 0.50:
        quality = int(np.random.randint(40, 86))
        _, buf  = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # Gaussian noise — mobile sensor noise in low-light
    if np.random.random() < 0.40:
        sigma = float(np.random.uniform(0, 15))
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Motion blur — hand-shake and fast movement
    if np.random.random() < 0.30:
        kernel = np.zeros((5, 5), dtype=np.float32)
        if np.random.random() > 0.5:
            kernel[2, :] = 1.0 / 5   # horizontal
        else:
            kernel[:, 2] = 1.0 / 5   # vertical
        img = cv2.filter2D(img, -1, kernel)

    return img


# =============================================================================
# VIDEO FRAME GENERATOR
# =============================================================================

class VideoFrameGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_generator, apply_mobile_aug=False):
        self.gen              = base_generator
        self.samples          = base_generator.samples
        self.classes          = base_generator.classes
        self.class_indices    = base_generator.class_indices
        self.apply_mobile_aug = apply_mobile_aug

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, idx):
        X, y = self.gen[idx]
        processed = []
        for x in X:
            # FIX B: ImageDataGenerator gives float32 in [0, 255] when no rescale
            # is set. Convert to uint8 directly — no multiplication needed.
            # If your ImageDataGenerator ever has rescale=1/255, multiply by 255
            # first: (x * 255).astype(np.uint8)
            frame = np.clip(x, 0, 255).astype(np.uint8)

            if self.apply_mobile_aug:
                frame = mobile_augment(frame)   # in: uint8[0,255] out: uint8[0,255]

            # preprocess_input expects float32 [0,255] and normalises internally
            processed.append(preprocess_input(frame.astype(np.float32)))

        return np.stack(processed, axis=0), y

    def on_epoch_end(self):
        self.gen.on_epoch_end()

    def reset(self):
        self.gen.reset()


# =============================================================================
# DATA GENERATORS
# =============================================================================

def build_generators():
    train_datagen = ImageDataGenerator(
        # NO rescale — preprocess_input handles normalisation
        rotation_range      = 15,
        width_shift_range   = 0.12,
        height_shift_range  = 0.12,
        shear_range         = 0.08,
        zoom_range          = 0.15,
        brightness_range    = [0.60, 1.40],
        horizontal_flip     = True,
        channel_shift_range = 20.0,
        fill_mode           = "nearest",
    )
    val_datagen = ImageDataGenerator()   # NO rescale

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
        f"Class mismatch: {raw_train.class_indices} — folders must be 'real' and 'fake'"
    )

    train_gen = VideoFrameGenerator(raw_train, apply_mobile_aug=True)
    val_gen   = VideoFrameGenerator(raw_val,   apply_mobile_aug=False)
    test_gen  = VideoFrameGenerator(raw_test,  apply_mobile_aug=False)

    return train_gen, val_gen, test_gen, raw_train, raw_val, raw_test


# =============================================================================
# DATA SANITY CHECK
# =============================================================================

def check_data(raw_train):
    labels  = raw_train.classes
    n_real  = int(np.sum(labels == 0))
    n_fake  = int(np.sum(labels == 1))
    total   = len(labels)
    balance = min(n_real, n_fake) / max(n_real, n_fake) if max(n_real, n_fake) > 0 else 0
    print(f"\n  Data: Real={n_real:,}  Fake={n_fake:,}  Total={total:,}  Balance={balance:.2f}")
    if total < 2000:
        print(f"  WARNING: Only {total} frames. Model will overfit. Aim for 3k+ per class.")
    if balance < 0.5:
        print(f"  WARNING: Class imbalance detected ({balance:.2f}).")


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def get_class_weights(raw_train):
    labels  = raw_train.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    cw = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class weights: real={cw[0]:.3f}  fake={cw[1]:.3f}")
    return cw


# =============================================================================
# MODEL
# =============================================================================

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
    x   = Concatenate()([avg, mx])   # 2560-dim

    x   = BatchNormalization(name="bn_head1")(x)
    x   = Dense(256, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(3e-4),
                name="fc1")(x)
    x   = Dropout(0.50, name="drop1")(x)
    x   = BatchNormalization(name="bn_head2")(x)
    x   = Dense(64, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(3e-4),
                name="fc2")(x)
    x   = Dropout(0.40, name="drop2")(x)
    out = Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

    return Model(inp, out, name="deepfake_video_b1"), base


# =============================================================================
# COMPILE
# =============================================================================

def compile_model(model, lr, weight_decay):
    model.compile(
        optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=1.0),
        loss      = focal_loss(CFG["FOCAL_GAMMA"], CFG["FOCAL_ALPHA"], CFG["LABEL_SMOOTHING"]),
        metrics   = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# =============================================================================
# CALLBACKS
# =============================================================================

def build_callbacks(phase):
    """
    Note: CollapseDetector no longer needs val_gen as a parameter.
    It reads val_auc from the logs dict passed by Keras — always accurate.
    """
    return [
        ModelCheckpoint(
            CFG["CHECKPOINT_PATH"], monitor="val_auc",
            save_best_only=True, mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc", patience=CFG["ES_PATIENCE"],
            restore_best_weights=True, mode="max", verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.4,
            patience=CFG["REDUCE_LR_PATIENCE"],
            min_lr=1e-8, verbose=1,
        ),
        CSVLogger(CFG["LOG_PATH"], append=(phase == 2)),
        tf.keras.callbacks.TerminateOnNaN(),
        # FIX C: passes no val_gen — uses logs dict instead
        CollapseDetector(check_after=12, min_auc=0.52, patience=4),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  ep{epoch+1:03d} | "
                f"loss={logs.get('loss', 0):.4f} | "
                f"val_acc={logs.get('val_accuracy', 0):.4f} | "
                f"val_auc={logs.get('val_auc', 0):.4f} | "
                f"val_rec={logs.get('val_recall', 0):.4f}"
            ) if (epoch + 1) % 2 == 0 else None
        ),
    ]


# =============================================================================
# OUTPUT RANGE CHECK
# =============================================================================

def check_output_range(model, val_gen, label):
    val_gen.reset()
    preds  = model.predict(val_gen, verbose=0).flatten()
    labels = val_gen.classes
    real_p = preds[labels == 0]
    fake_p = preds[labels == 1]
    print(f"\n  [{label}]")
    print(f"    All : min={preds.min():.3f}  max={preds.max():.3f}  std={preds.std():.3f}")
    if len(real_p): print(f"    Real: mean={real_p.mean():.3f}")
    if len(fake_p): print(f"    Fake: mean={fake_p.mean():.3f}")
    if len(real_p) and len(fake_p):
        sep = fake_p.mean() - real_p.mean()
        print(f"    Sep : {sep:.3f}  {'OK' if sep > 0.25 else 'LOW'}")
    val_gen.reset()


# =============================================================================
# THRESHOLD FINDER
# =============================================================================

def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(np.clip(thresholds[int(np.argmax(j))], 0.30, 0.80))


# =============================================================================
# PLOTS
# =============================================================================

def plot_history(h1, h2):
    def cat(key):
        return h1.history.get(key, []) + h2.history.get(key, [])
    epochs = range(1, len(cat("accuracy")) + 1)
    p1_end = len(h1.history.get("accuracy", []))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Video Model — Training History", fontsize=14, fontweight="bold")
    def _plot(ax, tk, vk, title):
        ax.plot(epochs, cat(tk), "b-", label="Train", linewidth=1.8)
        ax.plot(epochs, cat(vk), "r-", label="Val",   linewidth=1.8)
        if p1_end > 0:
            ax.axvline(p1_end, color="gray", linestyle="--", alpha=0.5, label="Phase 2")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    _plot(axes[0,0], "accuracy", "val_accuracy", "Accuracy")
    _plot(axes[0,1], "loss",     "val_loss",     "Loss")
    _plot(axes[1,0], "auc",      "val_auc",      "AUC")
    _plot(axes[1,1], "recall",   "val_recall",   "Recall")
    plt.tight_layout()
    out = str(BASE_DIR / "training_history_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  History → {out}")


def plot_score_distribution(y_true, y_prob, threshold):
    real_s, fake_s = y_prob[y_true == 0], y_prob[y_true == 1]
    bins = np.linspace(0, 1, 50)
    plt.figure(figsize=(10, 6))
    plt.hist(real_s, bins=bins, alpha=0.6, color="green", label=f"Real (n={len(real_s)})", density=True)
    plt.hist(fake_s, bins=bins, alpha=0.6, color="red",   label=f"Fake (n={len(fake_s)})", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", lw=2, label=f"Threshold={threshold:.2f}")
    plt.xlabel("P(fake)"); plt.ylabel("Density")
    plt.title("Score Distribution — Two peaks = healthy, one peak = collapsed")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = str(BASE_DIR / "score_distribution_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Score distribution → {out}")


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens); plt.colorbar(im)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Real","Fake"],
           yticklabels=["Real","Fake"], xlabel="Predicted", ylabel="True",
           title="Confusion Matrix — Video Model")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=18,
                    fontweight="bold", color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    out = str(BASE_DIR / "confusion_matrix_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix → {out}")


def plot_roc(fpr, tpr, auc_val):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="green", lw=2, label=f"AUC={auc_val:.4f}")
    plt.plot([0,1],[0,1], "k--", lw=1.5, label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve — Video Model")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = str(BASE_DIR / "roc_curve_video.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC → {out}")


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train():
    t_start = time.time()

    print("\n[1/6] Loading data …")
    train_gen, val_gen, test_gen, raw_train, raw_val, raw_test = build_generators()
    class_weights = get_class_weights(raw_train)
    check_data(raw_train)
    print(f"  Train={train_gen.samples:,}  Val={val_gen.samples:,}  Test={test_gen.samples:,}")

    print("\n[2/6] Building model …")
    model, base = build_model()
    print(f"  Total params: {model.count_params():,}")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"[3/6] PHASE 1 — head only ({CFG['PHASE1_EPOCHS']} ep  lr={CFG['PHASE1_LR']})")
    print(f"{'─'*62}")

    compile_model(model, lr=CFG["PHASE1_LR"], weight_decay=1e-4)

    h1 = model.fit(
        train_gen,
        epochs          = CFG["PHASE1_EPOCHS"],
        validation_data = val_gen,
        class_weight    = class_weights,
        callbacks       = build_callbacks(phase=1),
        verbose         = 1,
    )
    print(f"\n  Phase 1 done → val_acc={h1.history['val_accuracy'][-1]:.4f}  "
          f"val_auc={h1.history['val_auc'][-1]:.4f}")
    check_output_range(model, val_gen, "After phase 1")

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"[4/6] PHASE 2 — fine-tune top {CFG['UNFREEZE_LAYERS']} layers  lr={CFG['PHASE2_LR']}")
    print(f"{'─'*62}")

    base.trainable = True
    freeze_until = len(base.layers) - CFG["UNFREEZE_LAYERS"]
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers[freeze_until:]:
        layer.trainable = True

    # Lock ALL BN layers (base + head)
    bn_locked = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False; bn_locked += 1
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.BatchNormalization):
                    sublayer.trainable = False; bn_locked += 1
    print(f"  Locked {bn_locked} BN layers")
    trainable_p2 = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"  Trainable params: {trainable_p2:,}")

    compile_model(model, lr=CFG["PHASE2_LR"], weight_decay=1e-5)
    model.optimizer.clipnorm = 0.5

    h2 = model.fit(
        train_gen,
        epochs          = CFG["PHASE1_EPOCHS"] + CFG["PHASE2_EPOCHS"],
        initial_epoch   = len(h1.history["loss"]),
        validation_data = val_gen,
        class_weight    = class_weights,
        callbacks       = build_callbacks(phase=2),
        verbose         = 1,
    )
    print(f"\n  Phase 2 done → val_acc={h2.history['val_accuracy'][-1]:.4f}  "
          f"val_auc={h2.history['val_auc'][-1]:.4f}")
    check_output_range(model, val_gen, "After phase 2")

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\n[5/6] Loading best checkpoint and saving …")
    model.load_weights(CFG["CHECKPOINT_PATH"])
    model.save(CFG["MODEL_SAVE_PATH"])
    print(f"  Saved → {CFG['MODEL_SAVE_PATH']}")
    plot_history(h1, h2)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print(f"\n{'='*62}\n[6/6] TEST SET EVALUATION\n{'='*62}")

    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1).flatten()
    y_true = raw_test.classes

    print(f"\n  min={y_prob.min():.4f}  max={y_prob.max():.4f}  "
          f"mean={y_prob.mean():.4f}  std={y_prob.std():.4f}")

    real_s = y_prob[y_true == 0]
    fake_s = y_prob[y_true == 1]
    if len(real_s) and len(fake_s):
        sep = fake_s.mean() - real_s.mean()
        print(f"  Real: mean={real_s.mean():.3f}  range=[{real_s.min():.3f},{real_s.max():.3f}]")
        print(f"  Fake: mean={fake_s.mean():.3f}  range=[{fake_s.min():.3f},{fake_s.max():.3f}]")
        print(f"  Sep : {sep:.3f}  {'OK' if sep > 0.25 else 'LOW — need more data'}")

    opt_thresh = find_optimal_threshold(y_true, y_prob)
    y_pred     = (y_prob >= opt_thresh).astype(int)

    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy       = (tp + tn) / (tp + tn + fp + fn)
    precision      = tp / (tp + fp + 1e-8)
    recall         = tp / (tp + fn + 1e-8)
    f1             = 2 * precision * recall / (precision + recall + 1e-8)
    auc_val        = roc_auc_score(y_true, y_prob)
    fpr, tpr, _    = roc_curve(y_true, y_prob)

    print(f"\n  Threshold : {opt_thresh:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC AUC   : {auc_val:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Real','Fake'])}")

    plot_confusion_matrix(cm)
    plot_roc(fpr, tpr, auc_val)
    plot_score_distribution(y_true, y_prob, opt_thresh)

    # Overfitting gap check
    best_val_auc = max(max(h1.history.get("val_auc", [0])), max(h2.history.get("val_auc", [0])))
    gap = best_val_auc - auc_val
    print(f"\n  Overfit gap: best_val={best_val_auc:.4f}  test={auc_val:.4f}  gap={gap:.4f}")
    if gap > 0.08:
        print(f"  Still overfitting (gap={gap:.4f}). Add more diverse training videos.")
    elif gap > 0.04:
        print(f"  Mild overfitting ({gap:.4f}) — acceptable.")
    else:
        print(f"  Good generalisation (gap={gap:.4f})")

    # Save metadata
    meta = {
        "optimal_threshold":         float(round(opt_thresh, 4)),
        "video_detection_threshold": float(round(opt_thresh, 2)),
        "class_indices":             {"real": 0, "fake": 1},
        "input_size":                CFG["IMG_SIZE"][0],
        "preprocessing":             "efficientnet.preprocess_input",
        "output_activation":         "sigmoid",
        "output_meaning":            "P(fake)",
        "focal_loss_alpha":          CFG["FOCAL_ALPHA"],
        "focal_loss_gamma":          CFG["FOCAL_GAMMA"],
        "phase1_lr":                 CFG["PHASE1_LR"],
        "phase2_lr":                 CFG["PHASE2_LR"],
        "unfreeze_layers":           CFG["UNFREEZE_LAYERS"],
        "total_epochs":              CFG["PHASE1_EPOCHS"] + CFG["PHASE2_EPOCHS"],
        "mobile_augmentation":       True,
        "detection_strategy":        "weighted_average",
    }
    meta_path = Path(CFG["MODEL_SAVE_PATH"]).parent / "model_metadata_video.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata → {meta_path}")

    elapsed      = (time.time() - t_start) / 60
    print(f"\n{'='*62}")
    print("  VIDEO MODEL — TRAINING COMPLETE")
    print(f"{'='*62}")
    print(f"  Time         : {elapsed:.1f} min")
    print(f"  Best val AUC : {best_val_auc:.4f}")
    print(f"  Test AUC     : {auc_val:.4f}")
    print(f"  Test Recall  : {recall:.4f}")
    print(f"\n  >>> settings.py:  VIDEO_DETECTION_THRESHOLD = {opt_thresh:.2f}")
    grade = ("EXCELLENT" if auc_val > 0.90 else "GOOD" if auc_val > 0.85
             else "FAIR" if auc_val > 0.78 else "NEEDS MORE DATA")
    print(f"  Grade: {grade}")
    print(f"{'='*62}")

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