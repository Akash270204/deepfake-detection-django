"""
verify_model.py  —  Run after training to confirm model works correctly
=======================================================================

This script:
  1. Loads the trained model exactly as ml_utils.py does
  2. Runs it on a sample of your test images
  3. Prints per-image predictions so you can manually verify
  4. Confirms the preprocessing pipeline matches training
  5. Checks the output range is still healthy after loading

Run:  python verify_model.py

If this script shows correct predictions, your Django app will too.
If it shows collapse or wrong predictions, the issue is in model saving.
"""

import os
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
import django
django.setup()
from django.conf import settings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image


# ── Load model exactly as ml_utils.py does ───────────────────────────────────
print("\n" + "=" * 60)
print("MODEL VERIFICATION")
print("=" * 60)

model_path = settings.IMAGE_MODEL_PATH
threshold  = settings.IMAGE_DETECTION_THRESHOLD

print(f"Model path : {model_path}")
print(f"Threshold  : {threshold}")

if not Path(model_path).exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

def focal_loss(y_true, y_pred):
    gamma, alpha = 2.0, 0.75
    eps    = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)
    y_true = tf.cast(y_true, tf.float32)
    bce    = -(y_true * tf.math.log(y_pred)
               + (1 - y_true) * tf.math.log(1 - y_pred))
    p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_w = y_true * alpha + (1 - y_true) * (1 - alpha)
    return tf.reduce_mean(alpha_w * tf.pow(1.0 - p_t, gamma) * bce)

try:
    model = keras.models.load_model(
        str(model_path),
        custom_objects={'focal_loss': focal_loss},
        compile=False,
    )
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(f"Model loaded OK  input={model.input_shape[1]}x{model.input_shape[1]}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

input_size = model.input_shape[1]
is_binary  = (model.output_shape[-1] == 1)
print(f"Output type: {'binary sigmoid' if is_binary else 'softmax'}")


# ── Preprocessing function (identical to ml_utils.py) ────────────────────────
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_size, input_size), Image.LANCZOS)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    return preprocess_input(arr)


# ── Quick sanity: random inputs ───────────────────────────────────────────────
print("\n--- Random input sanity check ---")
rand = preprocess_input(
    np.random.randint(0, 256, (10, input_size, input_size, 3)).astype(np.float32)
)
out = model.predict(rand, verbose=0).flatten()
print(f"Random inputs: min={out.min():.3f}  max={out.max():.3f}  std={out.std():.3f}")
if out.max() - out.min() < 0.05:
    print("WARNING: Still collapsed on random inputs. Model may be broken.")
else:
    print("OK: Model gives varied outputs on random inputs.")


# ── Test on actual images ─────────────────────────────────────────────────────
test_dir  = settings.DATASET.get('IMAGE_TEST', settings.DATASET.get('TEST'))
real_dir  = Path(test_dir) / 'real'
fake_dir  = Path(test_dir) / 'fake'

if not real_dir.exists() or not fake_dir.exists():
    print(f"\nTest dir not found — skipping image tests")
    print(f"Expected: {real_dir} and {fake_dir}")
    sys.exit(0)

EXTS = {'.jpg', '.jpeg', '.png'}

real_images = [f for f in real_dir.iterdir() if f.suffix.lower() in EXTS][:15]
fake_images = [f for f in fake_dir.iterdir() if f.suffix.lower() in EXTS][:15]

print(f"\n--- Per-image predictions (threshold={threshold}) ---")
print(f"{'File':<45} {'Score':>7}  {'Pred':<6}  {'True':<6}  {'OK?'}")
print("─" * 75)

correct = 0
total   = 0
scores_real = []
scores_fake = []

for img_path in real_images:
    try:
        arr   = preprocess(img_path)
        score = float(model.predict(arr, verbose=0)[0][0])
        pred  = 'FAKE' if score > threshold else 'REAL'
        ok    = '✓' if pred == 'REAL' else '✗'
        if pred == 'REAL':
            correct += 1
        total += 1
        scores_real.append(score)
        print(f"{img_path.name:<45} {score:>7.3f}  {pred:<6}  {'REAL':<6}  {ok}")
    except Exception as e:
        print(f"{img_path.name:<45} ERROR: {e}")

print()
for img_path in fake_images:
    try:
        arr   = preprocess(img_path)
        score = float(model.predict(arr, verbose=0)[0][0])
        pred  = 'FAKE' if score > threshold else 'REAL'
        ok    = '✓' if pred == 'FAKE' else '✗'
        if pred == 'FAKE':
            correct += 1
        total += 1
        scores_fake.append(score)
        print(f"{img_path.name:<45} {score:>7.3f}  {pred:<6}  {'FAKE':<6}  {ok}")
    except Exception as e:
        print(f"{img_path.name:<45} ERROR: {e}")

print("─" * 75)
acc = correct / total * 100 if total else 0
print(f"Accuracy on sample: {correct}/{total} = {acc:.1f}%")

if scores_real and scores_fake:
    print(f"\nScore summary on this sample:")
    print(f"  Real images: mean={np.mean(scores_real):.3f}  "
          f"range=[{min(scores_real):.3f}, {max(scores_real):.3f}]")
    print(f"  Fake images: mean={np.mean(scores_fake):.3f}  "
          f"range=[{min(scores_fake):.3f}, {max(scores_fake):.3f}]")
    sep = np.mean(scores_fake) - np.mean(scores_real)
    print(f"  Separation: {sep:.3f}  "
          f"{'OK' if sep > 0.2 else 'LOW — model may struggle'}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
if acc >= 75:
    print(f"Model appears to be working correctly ({acc:.1f}% on sample).")
    print("Ready to use in the Django app.")
else:
    print(f"Low accuracy on sample ({acc:.1f}%). "
          f"Check threshold or consider retraining.")
print("=" * 60)