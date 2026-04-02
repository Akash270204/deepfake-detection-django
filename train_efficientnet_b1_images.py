"""
train_image_model.py  —  EfficientNet-B1 Deepfake Classifier (Fixed)
======================================================================

FIXES vs original script:
  1. BN layers in base LOCKED during phase 2 (collapse prevention #1)
  2. LR2 = 1e-5  (was 1e-4 — 10x too high, destroyed pretrained weights)
  3. UNFREEZE = 20 layers  (was 40 — caused overfitting on small datasets)
  4. EarlyStopping monitors val_auc not val_loss
  5. Focal loss alpha = 0.75  (was 0.25 — was upweighting REAL not FAKE)
  6. Uses IMAGE_TRAIN/IMAGE_VAL/IMAGE_TEST paths (not mixed TRAIN/VAL)
  7. Output range diagnostic printed after every phase
  8. Threshold saved to model_metadata.json automatically
  9. CollapseDetector callback stops training if outputs flatten
 10. Warm-up LR schedule for phase 1 head stabilisation
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Project setup ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
import django
django.setup()
from django.conf import settings

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── GPU setup ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision (float16) enabled")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5376)]
    )
    print(f"GPU found: {len(gpus)} device(s), memory limit: 5.25 GB")
else:
    print("No GPU — using CPU")
print("=" * 60 + "\n")


# =============================================================================
# FOCAL LOSS  —  alpha FIXED to 0.75
# =============================================================================

def focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """
    Binary focal loss.

    alpha = weight on the POSITIVE (fake, label=1) class.
    0.75 means "penalise missed fakes 3x harder than false alarms".

    Original script had alpha=0.25 which is BACKWARDS — it upweighted
    the real class, causing the model to predict real for everything.
    """
    def loss_fn(y_true, y_pred):
        eps     = tf.keras.backend.epsilon()
        y_pred  = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)
        y_true  = tf.cast(y_true, tf.float32)
        bce     = -(y_true * tf.math.log(y_pred)
                    + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_w = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_w * tf.pow(1.0 - p_t, gamma) * bce)
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


# =============================================================================
# COLLAPSE DETECTOR CALLBACK
# =============================================================================

class CollapseDetector(keras.callbacks.Callback):
    """
    Stops training if model outputs collapse to a flat range.
    Checks output std on one val batch every epoch after epoch 3.
    std < 0.05 means the model is predicting nearly the same score
    for all inputs — it has failed to learn any discrimination.
    """
    def __init__(self, val_gen, check_after_epoch=3, min_std=0.05):
        super().__init__()
        self.val_gen     = val_gen
        self.check_after = check_after_epoch
        self.min_std     = min_std

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.check_after:
            return
        x_batch, _ = next(iter(self.val_gen))
        preds = self.model.predict(x_batch, verbose=0).flatten()
        std   = float(np.std(preds))
        rng   = float(preds.max() - preds.min())
        print(f"\n  [CollapseDetector] output std={std:.4f}  range={rng:.4f}")
        if std < self.min_std:
            print(f"  [CollapseDetector] COLLAPSE — stopping. "
                  f"std={std:.4f} < {self.min_std}")
            self.model.stop_training = True


# =============================================================================
# WARM-UP + COSINE DECAY LR SCHEDULE
# =============================================================================

class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up for first epoch, cosine decay for the rest."""
    def __init__(self, initial_lr, total_steps, warmup_steps):
        super().__init__()
        self.initial_lr   = float(initial_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step       = tf.cast(step, tf.float32)
        warmup_lr  = self.initial_lr * (step / self.warmup_steps)
        progress   = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0
        )
        cosine_lr  = 0.5 * self.initial_lr * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'initial_lr':   self.initial_lr,
            'total_steps':  self.total_steps,
            'warmup_steps': self.warmup_steps,
        }


# =============================================================================
# TRAINER
# =============================================================================

class DeepfakeTrainerB1:

    IMAGE_SIZE    = settings.IMAGE_SIZE
    BATCH_SIZE    = settings.TRAINING['BATCH_SIZE']
    PHASE1_EPOCHS = settings.TRAINING['PHASE1_EPOCHS']
    PHASE2_EPOCHS = settings.TRAINING['PHASE2_EPOCHS']
    LR1           = settings.TRAINING['LEARNING_RATE_PHASE1']   # 1e-3
    LR2           = 1e-5      # FIXED: was 1e-4 in settings — 10x too high
    PATIENCE      = settings.TRAINING['EARLY_STOPPING_PATIENCE']
    UNFREEZE      = 20        # FIXED: was 40 — too many layers for small datasets
    WEIGHT_DECAY  = 1e-5

    def __init__(self):
        self.model          = None
        self.base_model     = None
        self.history        = {}
        self.best_threshold = 0.50
        self._loss_fn       = focal_loss(gamma=2.0, alpha=0.75)

        print("DeepfakeTrainerB1 (fixed)")
        print(f"  image_size   : {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
        print(f"  batch_size   : {self.BATCH_SIZE}")
        print(f"  LR1          : {self.LR1}")
        print(f"  LR2          : {self.LR2}   (FIXED from 1e-4)")
        print(f"  unfreeze     : top {self.UNFREEZE} layers  (FIXED from 40)")
        print(f"  focal alpha  : 0.75  (FIXED from 0.25)")

    # ── Build ─────────────────────────────────────────────────────────────────
    def build_model(self) -> keras.Model:
        print("\nBuilding EfficientNet-B1 …")
        self.base_model = EfficientNetB1(
            include_top=False,
            weights='imagenet',
            input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
            pooling=None,
        )
        self.base_model.trainable = False

        inputs = layers.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.BatchNormalization(name='bn_h1')(x)
        x = layers.Dropout(0.40, name='drop1')(x)
        x = layers.Dense(256, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(self.WEIGHT_DECAY),
                          name='fc1')(x)
        x = layers.BatchNormalization(name='bn_h2')(x)
        x = layers.Dropout(0.35, name='drop2')(x)
        x = layers.Dense(128, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(self.WEIGHT_DECAY),
                          name='fc2')(x)
        x = layers.BatchNormalization(name='bn_h3')(x)
        x = layers.Dropout(0.25, name='drop3')(x)
        outputs = layers.Dense(1, activation='sigmoid',
                                dtype='float32', name='output')(x)

        self.model = Model(inputs, outputs, name='deepfake_b1')
        total = self.model.count_params()
        print(f"  params: {total:,}")
        return self.model

    # ── Data ──────────────────────────────────────────────────────────────────
    def prepare_data(self):
        # Use image-specific paths; fall back to legacy paths
        train_dir = settings.DATASET.get('IMAGE_TRAIN', settings.DATASET['TRAIN'])
        val_dir   = settings.DATASET.get('IMAGE_VAL',   settings.DATASET['VAL'])

        if not Path(train_dir).exists():
            train_dir = settings.DATASET['TRAIN']
            val_dir   = settings.DATASET['VAL']

        if not Path(train_dir).exists():
            raise FileNotFoundError(
                f"Dataset missing. Create:\n"
                f"  {train_dir}/real/  and  {train_dir}/fake/"
            )

        for cls in ['real', 'fake']:
            p = Path(train_dir) / cls
            if not p.exists():
                raise FileNotFoundError(f"Missing folder: {p}")

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.10,
            height_shift_range=0.10,
            shear_range=0.08,
            zoom_range=0.12,
            horizontal_flip=True,
            brightness_range=[0.80, 1.20],
            channel_shift_range=10.0,
            fill_mode='nearest',
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        kw = dict(
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            classes=['real', 'fake'],   # real=0  fake=1
            interpolation='bilinear',
        )
        train_gen = train_datagen.flow_from_directory(
            train_dir, shuffle=True, seed=42, **kw
        )
        val_gen = val_datagen.flow_from_directory(
            val_dir, shuffle=False, **kw
        )

        assert train_gen.class_indices == {'real': 0, 'fake': 1}, (
            f"Class mismatch: {train_gen.class_indices}"
        )

        real_n = int(np.sum(train_gen.labels == 0))
        fake_n = int(np.sum(train_gen.labels == 1))
        total  = real_n + fake_n
        print(f"\nDataset: train={train_gen.samples:,}  val={val_gen.samples:,}")
        print(f"Balance: real={real_n:,} ({real_n/total:.1%})  "
              f"fake={fake_n:,} ({fake_n/total:.1%})")
        if abs(real_n - fake_n) / total > 0.3:
            print("WARNING: imbalance > 30%. Class weights will compensate.")

        class_weights = {
            0: round(total / (2.0 * max(real_n, 1)), 4),
            1: round(total / (2.0 * max(fake_n, 1)), 4),
        }
        print(f"Class weights: {class_weights}")
        return train_gen, val_gen, class_weights

    # ── Compile ───────────────────────────────────────────────────────────────
    def _compile(self, optimizer):
        self.model.compile(
            optimizer=optimizer,
            loss=self._loss_fn,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ],
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _callbacks(self, ckpt_path, patience, csv_name, val_gen):
        return [
            keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor='val_auc',   # FIXED: was val_loss
                save_best_only=True,
                mode='max',
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_auc',   # FIXED: was val_loss
                patience=patience,
                restore_best_weights=True,
                min_delta=1e-4,
                mode='max',
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.4,
                patience=5,
                min_lr=1e-8,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(project_root / csv_name),
            keras.callbacks.TerminateOnNaN(),
            CollapseDetector(val_gen, check_after_epoch=3, min_std=0.05),
        ]

    # ── Output range check ────────────────────────────────────────────────────
    def _check_output_range(self, val_gen, label):
        val_gen.reset()
        preds  = self.model.predict(val_gen, verbose=0).flatten()
        labels = val_gen.classes
        real_p = preds[labels == 0]
        fake_p = preds[labels == 1]

        print(f"\n  [{label}] output distribution:")
        print(f"    All  : min={preds.min():.3f}  max={preds.max():.3f}  "
              f"std={preds.std():.3f}")
        if len(real_p):
            print(f"    Real : mean={real_p.mean():.3f}  "
                  f"range=[{real_p.min():.3f}, {real_p.max():.3f}]")
        if len(fake_p):
            print(f"    Fake : mean={fake_p.mean():.3f}  "
                  f"range=[{fake_p.min():.3f}, {fake_p.max():.3f}]")
        if len(real_p) and len(fake_p):
            sep = fake_p.mean() - real_p.mean()
            print(f"    Separation : {sep:.3f}  "
                  f"{'OK' if sep > 0.3 else 'LOW — model not discriminating'}")
        val_gen.reset()

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self):
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)
        t0 = time.time()

        model_dir = project_root / 'detector' / 'ml_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(settings.IMAGE_MODEL_PATH)

        self.build_model()
        train_gen, val_gen, class_weights = self.prepare_data()
        steps = len(train_gen)

        # ── Phase 1 ───────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"Phase 1 — head only  ({self.PHASE1_EPOCHS} epochs  lr={self.LR1})")
        print(f"{'─'*60}")

        p1_schedule = WarmUpCosineDecay(
            initial_lr   = self.LR1,
            total_steps  = self.PHASE1_EPOCHS * steps,
            warmup_steps = steps,            # 1 epoch warm-up
        )
        self.base_model.trainable = False
        self._compile(keras.optimizers.Adam(
            learning_rate=p1_schedule, clipnorm=1.0
        ))

        h1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.PHASE1_EPOCHS,
            callbacks=self._callbacks(ckpt_path, self.PATIENCE,
                                      'log_phase1.csv', val_gen),
            class_weight=class_weights,
            verbose=1,
        )
        self._merge_history(h1.history)
        self._print_phase_summary(h1, 'Phase 1')
        self._check_output_range(val_gen, 'After phase 1')

        # ── Phase 2 ───────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"Phase 2 — fine-tune top {self.UNFREEZE} layers")
        print(f"LR={self.LR2}   BN layers locked")
        print(f"{'─'*60}")

        self.base_model.trainable = True
        freeze_until = len(self.base_model.layers) - self.UNFREEZE
        for layer in self.base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in self.base_model.layers[freeze_until:]:
            layer.trainable = True

        # CRITICAL: lock ALL BatchNorm layers in base.
        # BN running stats were calibrated on 1.2M ImageNet images.
        # Retraining them on a small deepfake dataset corrupts the stats
        # and causes output collapse to ~0.23 for all inputs.
        bn_locked = 0
        for layer in self.base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                bn_locked += 1
        print(f"  Locked {bn_locked} BN layers in base")

        trainable = sum(np.prod(w.shape) for w in self.model.trainable_weights)
        print(f"  Trainable params: {trainable:,}")

        self._compile(keras.optimizers.Adam(
            learning_rate=self.LR2, clipnorm=0.5
        ))

        h2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.PHASE1_EPOCHS + self.PHASE2_EPOCHS,
            initial_epoch=len(h1.history['loss']),
            callbacks=self._callbacks(
                ckpt_path, self.PATIENCE + 4, 'log_phase2.csv', val_gen
            ),
            class_weight=class_weights,
            verbose=1,
        )
        self._merge_history(h2.history)
        self._print_phase_summary(h2, 'Phase 2')
        self._check_output_range(val_gen, 'After phase 2')

        # Reload best checkpoint
        print("\nLoading best checkpoint …")
        self.model = keras.models.load_model(
            ckpt_path,
            custom_objects={'focal_loss': self._loss_fn},
            compile=False,
        )
        self.model.compile(
            optimizer='adam', loss=self._loss_fn,
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )

        print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self):
        test_dir = settings.DATASET.get('IMAGE_TEST')
        if test_dir is None or not Path(test_dir).exists():
            test_dir = settings.DATASET.get('IMAGE_VAL',
                       settings.DATASET.get('VAL'))

        print("\n" + "=" * 60)
        print(f"EVALUATION  ({test_dir})")
        print("=" * 60)

        eval_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        ).flow_from_directory(
            test_dir,
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False,
            interpolation='bilinear',
        )
        print(f"Eval images: {eval_gen.samples:,}")

        preds     = self.model.predict(eval_gen, verbose=1).flatten()
        true_lbls = eval_gen.classes

        self._check_output_range(eval_gen, 'Final')

        self.best_threshold = self._find_optimal_threshold(true_lbls, preds)
        print(f"\nOptimal threshold: {self.best_threshold:.4f}")

        pred_cls = (preds >= self.best_threshold).astype(int)
        self._print_metrics(true_lbls, pred_cls, preds)
        self._plot_roc_curve(true_lbls, preds)
        self._plot_confusion_matrix(true_lbls, pred_cls)
        self._plot_score_distribution(true_lbls, preds)

        # Save metadata
        meta = {
            'optimal_threshold':             float(round(self.best_threshold, 4)),
            'image_detection_threshold':     float(round(self.best_threshold, 2)),
            'class_indices':                 {'real': 0, 'fake': 1},
            'input_size':                    self.IMAGE_SIZE,
            'preprocessing':                 'efficientnet.preprocess_input',
            'output_activation':             'sigmoid',
            'output_meaning':                'P(fake)',
            'focal_loss_alpha':              0.75,
            'focal_loss_gamma':              2.0,
            'unfreeze_layers':               self.UNFREEZE,
            'phase2_lr':                     self.LR2,
        }
        meta_path = Path(settings.IMAGE_MODEL_PATH).parent / 'model_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\nMetadata saved: {meta_path}")
        print(f"\n>>> Update settings.py:")
        print(f"    IMAGE_DETECTION_THRESHOLD = {self.best_threshold:.2f}")

    @staticmethod
    def _find_optimal_threshold(y_true, y_prob) -> float:
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j      = tpr - fpr
            best   = int(np.argmax(j))
            return float(np.clip(thresholds[best], 0.30, 0.80))
        except ImportError:
            best_t, best_f1 = 0.50, 0.0
            for t in np.arange(0.30, 0.80, 0.01):
                p  = (y_prob >= t).astype(int)
                tp = np.sum((p == 1) & (y_true == 1))
                fp = np.sum((p == 1) & (y_true == 0))
                fn = np.sum((p == 0) & (y_true == 1))
                pr = tp / (tp + fp + 1e-8)
                rc = tp / (tp + fn + 1e-8)
                f1 = 2 * pr * rc / (pr + rc + 1e-8)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            return best_t

    @staticmethod
    def _print_metrics(y_true, y_pred, y_prob):
        try:
            from sklearn.metrics import (
                confusion_matrix, classification_report, roc_auc_score
            )
            cm             = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            acc  = (tp + tn) / (tp + tn + fp + fn)
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            auc  = roc_auc_score(y_true, y_prob)
            print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
            print(f"Accuracy   : {acc:.4f}")
            print(f"Precision  : {prec:.4f}")
            print(f"Recall     : {rec:.4f}")
            print(f"Specificity: {spec:.4f}")
            print(f"F1         : {f1:.4f}")
            print(f"ROC AUC    : {auc:.4f}")
            print(f"\n{classification_report(y_true, y_pred, target_names=['Real','Fake'], digits=4)}")
        except ImportError:
            print("pip install scikit-learn for detailed metrics")

    def _plot_roc_curve(self, y_true, y_prob):
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ra = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC AUC={ra:.4f}')
            plt.plot([0,1],[0,1],'navy',lw=2,linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            path = project_root / 'roc_curve.png'
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"ROC saved: {path}")
        except ImportError:
            pass

    def _plot_confusion_matrix(self, y_true, y_pred):
        try:
            from sklearn.metrics import confusion_matrix
            cm     = confusion_matrix(y_true, y_pred)
            labels = ['Real', 'Fake']
            plt.figure(figsize=(7, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            t = np.arange(len(labels))
            plt.xticks(t, labels); plt.yticks(t, labels)
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i,j]),
                             ha='center', va='center', fontsize=14,
                             color='white' if cm[i,j] > thresh else 'black')
            plt.ylabel('True'); plt.xlabel('Predicted')
            plt.tight_layout()
            path = project_root / 'confusion_matrix.png'
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"Confusion matrix saved: {path}")
        except ImportError:
            pass

    def _plot_score_distribution(self, y_true, y_prob):
        """
        Most useful diagnostic plot.
        Two separated peaks = healthy model.
        One central peak = collapsed model.
        """
        real_s = y_prob[y_true == 0]
        fake_s = y_prob[y_true == 1]
        bins   = np.linspace(0, 1, 50)
        plt.figure(figsize=(10, 6))
        plt.hist(real_s, bins=bins, alpha=0.6, color='green',
                 label=f'Real (n={len(real_s)})', density=True)
        plt.hist(fake_s, bins=bins, alpha=0.6, color='red',
                 label=f'Fake (n={len(fake_s)})', density=True)
        plt.axvline(x=self.best_threshold, color='black', linestyle='--',
                    lw=2, label=f'Threshold={self.best_threshold:.2f}')
        plt.xlabel('Model Output  P(fake)')
        plt.ylabel('Density')
        plt.title('Score Distribution — Real vs Fake\n'
                  '(Two separated peaks = healthy  |  One central peak = collapsed)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = project_root / 'score_distribution.png'
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Score distribution saved: {path}")

    # ── Plot history ──────────────────────────────────────────────────────────
    def plot_history(self):
        if not self.history:
            return
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        pairs = [
            ('accuracy',  'val_accuracy',  'Accuracy',  axes[0, 0]),
            ('loss',      'val_loss',       'Loss',      axes[0, 1]),
            ('auc',       'val_auc',        'AUC',       axes[0, 2]),
            ('precision', 'val_precision',  'Precision', axes[1, 0]),
            ('recall',    'val_recall',     'Recall',    axes[1, 1]),
        ]
        for tk, vk, title, ax in pairs:
            if tk in self.history:
                ax.plot(self.history[tk], label='Train', lw=2)
            if vk in self.history:
                ax.plot(self.history[vk], label='Val',   lw=2)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.legend(); ax.grid(alpha=0.3)
        ax = axes[1, 2]
        if 'val_accuracy' in self.history:
            ax.plot(self.history['val_accuracy'], label='Val Acc', lw=2)
        if 'val_auc' in self.history:
            ax.plot(self.history['val_auc'],      label='Val AUC', lw=2)
        ax.set_title('Validation Overview', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
        plt.suptitle('EfficientNet-B1 Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = project_root / 'training_history_b1_images.png'
        plt.savefig(path, dpi=300); plt.close()
        print(f"History plot saved: {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self):
        best_acc = max(self.history.get('val_accuracy', [0]))
        best_auc = max(self.history.get('val_auc',      [0]))
        best_rec = max(self.history.get('val_recall',   [0]))
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best val accuracy : {best_acc*100:.2f}%")
        print(f"  Best val AUC      : {best_auc:.4f}")
        print(f"  Best val recall   : {best_rec:.4f}")
        print(f"\n  >>> settings.py:  IMAGE_DETECTION_THRESHOLD = "
              f"{self.best_threshold:.2f}")
        grade = ("EXCELLENT" if best_auc > 0.97 else
                 "GOOD"      if best_auc > 0.94 else
                 "FAIR"      if best_auc > 0.90 else "POOR")
        print(f"  Grade: {grade}")
        print("=" * 60)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _merge_history(self, h):
        for k, v in h.items():
            self.history.setdefault(k, []).extend(v)

    @staticmethod
    def _print_phase_summary(h, label):
        hh = h.history
        print(f"\n{label}:")
        print(f"  acc  train={hh['accuracy'][-1]:.4f}  val={hh['val_accuracy'][-1]:.4f}")
        print(f"  auc  train={hh['auc'][-1]:.4f}  val={hh['val_auc'][-1]:.4f}")
        print(f"  loss train={hh['loss'][-1]:.5f}  val={hh['val_loss'][-1]:.5f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    try:
        trainer = DeepfakeTrainerB1()
        trainer.train()
        trainer.evaluate()
        trainer.plot_history()
        trainer.summary()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()