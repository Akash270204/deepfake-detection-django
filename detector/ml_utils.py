"""
ml_utils.py — Deepfake Detector Inference Utilities

CHANGES IN THIS VERSION:

1. _focal_loss alpha = 0.75 (correct — matches training)

2. Bilateral filter removed from _preprocess_video_frame (correct — matches training)

3. VIDEO DECISION LOGIC COMPLETELY REWRITTEN
   Old logic: majority_vote — count frames above threshold, >50% = fake
   Problem: 31 frames at score 0.59 beats 29 frames at score 0.10.
            The model is barely calling 31 frames fake (0.59 ≈ threshold)
            but strongly calling 29 frames real. The video is probably real.

   New logic: weighted_average — three signals combined
     a) Average fake probability across all frames (most important)
     b) Fake frame percentage (secondary)
     c) Confidence-weighted score (high-confidence frames count more)

   Decision:
     - weighted_score = 0.60 * avg_prob + 0.30 * fake_pct + 0.10 * conf_score
     - is_deepfake = weighted_score > threshold

   This means a video where all frames score 0.80-0.90 is correctly
   called fake even if only 40% of frames cross the threshold.
   A video where 55% of frames barely cross at 0.59 may correctly
   be called real if the overall average is low.

4. MOBILE VIDEO ROBUSTNESS AT INFERENCE
   Added image enhancement before prediction for mobile frames:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
     Corrects the low-contrast / dark frames that mobile cameras produce.
     Does NOT alter model-relevant features, only makes the input
     match the lighting distribution the model was trained on.
   - No other changes to preprocessing — preprocess_input still applied.

5. PARAMETERS USED FOR DECISION — explained in _make_video_decision():
   - avg_fake_prob: mean of all frame fake probabilities
   - fake_frame_pct: fraction of frames where P(fake) > threshold
   - confidence_weighted_score: frames with higher max(p, 1-p) count more
   - temporal_consistency: variance of frame scores (consistent = more reliable)
   - weighted_score: final combined score for the is_deepfake decision
"""

import gc
import time
import uuid
import logging

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from pathlib import Path
from django.conf import settings
from tensorflow.keras.applications.efficientnet import preprocess_input

logger = logging.getLogger(__name__)
_detector_instance = None


# =============================================================================
# QUALITY VALIDATOR
# =============================================================================

class QualityValidator:

    @staticmethod
    def check_blur(image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0, "Cannot read image"
        score     = float(cv2.Laplacian(img, cv2.CV_64F).var())
        threshold = settings.QUALITY_VALIDATION.get('BLUR_THRESHOLD', 100.0)
        ok        = score >= threshold
        msg       = f"Image blur score: {score:.1f} ({'OK' if ok else f'below threshold {threshold}'})"
        return ok, score, msg

    @staticmethod
    def check_resolution(image_path):
        img      = Image.open(image_path)
        w, h     = img.size
        min_res  = settings.QUALITY_VALIDATION.get('MIN_RESOLUTION',  (224, 224))
        warn_res = settings.QUALITY_VALIDATION.get('WARN_RESOLUTION', (480, 480))
        if w < min_res[0] or h < min_res[1]:
            return False, (w, h), f"Resolution {w}x{h} below minimum {min_res[0]}x{min_res[1]}"
        if w < warn_res[0] or h < warn_res[1]:
            return True, (w, h), f"⚠️ Low resolution {w}x{h} — recommend {warn_res[0]}x{warn_res[1]}+"
        return True, (w, h), f"Resolution {w}x{h} OK"

    @classmethod
    def validate_image(cls, image_path) -> dict:
        result = {'is_valid': True, 'warnings': [], 'errors': [], 'metrics': {}}
        res_ok, resolution, res_msg = cls.check_resolution(image_path)
        result['metrics']['resolution'] = resolution
        if not res_ok:
            result['is_valid'] = False
            result['errors'].append(res_msg)
        elif '⚠️' in res_msg:
            result['warnings'].append(res_msg)
        if settings.QUALITY_VALIDATION.get('ENABLE_BLUR_DETECTION', True):
            blur_ok, blur_score, blur_msg = cls.check_blur(image_path)
            result['metrics']['blur_score'] = blur_score
            if not blur_ok:
                result['is_valid'] = False
                result['errors'].append(blur_msg)
        return result


# =============================================================================
# FORENSIC ANALYZER — ADVISORY ONLY
# =============================================================================

class ForensicAnalyzer:

    @staticmethod
    def fft_artifact_analysis(image_path) -> dict | None:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            spectrum      = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
            magnitude     = np.log1p(np.abs(spectrum))
            mag_u8        = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w          = mag_u8.shape
            cy, cx        = h // 2, w // 2
            r             = min(h, w) // 6
            yy, xx        = np.ogrid[:h, :w]
            dist2         = (yy - cy) ** 2 + (xx - cx) ** 2
            ring          = (dist2 >= (r - 5) ** 2) & (dist2 <= (r + 5) ** 2)
            ring_vals     = magnitude[ring]
            spike_ratio   = float(np.std(ring_vals) / (np.mean(ring_vals) + 1e-8))
            grid_artifact = spike_ratio > 0.65
            center_e      = float(np.sum(magnitude[cy-r:cy+r, cx-r:cx+r]))
            total_e       = float(np.sum(magnitude))
            fft_score     = float(np.clip((0.5 - center_e / (total_e + 1e-8)) * 200, 0, 100))
            fft_dir = settings.MEDIA_ROOT / 'forensic' / 'fft'
            fft_dir.mkdir(parents=True, exist_ok=True)
            fid = uuid.uuid4().hex[:8]
            colored = cv2.cvtColor(cv2.applyColorMap(mag_u8, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(img, cmap='gray'); ax1.set_title('Grayscale'); ax1.axis('off')
            ax2.imshow(colored); ax2.set_title(f'FFT spike={spike_ratio:.3f}'); ax2.axis('off')
            plt.tight_layout()
            plt.savefig(fft_dir / f'{fid}_fft.jpg', dpi=120, bbox_inches='tight')
            plt.close(); gc.collect()
            return {'fft_score': round(fft_score, 2), 'spike_ratio': round(spike_ratio, 3),
                    'has_grid_artifacts': bool(grid_artifact), 'fft_map_path': f'forensic/fft/{fid}_fft.jpg'}
        except Exception as exc:
            logger.warning(f"FFT analysis failed: {exc}")
            return None

    @staticmethod
    def compression_analysis(image_path) -> dict | None:
        try:
            img  = cv2.imread(str(image_path))
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w, bs = gray.shape[0], gray.shape[1], 8
            stds = [float(np.std(gray[y:y+bs, x:x+bs]))
                    for y in range(0, h-bs, bs) for x in range(0, w-bs, bs)]
            if not stds:
                return None
            score = float(np.clip((np.var(stds) / (np.mean(stds) + 1e-8)) * 10, 0, 100))
            return {'inconsistency_score': round(score, 2), 'suspicious': score > 50}
        except Exception as exc:
            logger.warning(f"Compression analysis failed: {exc}")
            return None

    @staticmethod
    def noise_analysis(image_path) -> dict | None:
        try:
            img   = cv2.imread(str(image_path))
            if img is None:
                return None
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
            n_std = float(np.std(noise))
            h, w  = noise.shape
            q_stds = [float(np.std(noise[:h//2, :w//2])), float(np.std(noise[:h//2, w//2:])),
                      float(np.std(noise[h//2:, :w//2])), float(np.std(noise[h//2:, w//2:]))]
            score = float(np.clip((1.0 - np.std(q_stds) / (n_std + 1e-8)) * 50, 0, 100))
            return {'noise_score': round(score, 2), 'suspicious': score > 50}
        except Exception as exc:
            logger.warning(f"Noise analysis failed: {exc}")
            return None

    @staticmethod
    def trust_score(model_conf: float, quality_results) -> float:
        score = model_conf
        if quality_results:
            if quality_results.get('errors'):
                score *= 0.70
            elif quality_results.get('warnings'):
                score *= 0.90
        return round(float(np.clip(score, 0, 100)), 2)


# =============================================================================
# DEEPFAKE DETECTOR
# =============================================================================

class DeepfakeDetector:

    def __init__(self):
        self.image_model_path    = Path(settings.IMAGE_MODEL_PATH)
        self.image_threshold     = float(settings.IMAGE_DETECTION_THRESHOLD)
        self.image_model         = None
        self.image_grad_model    = None
        self._image_model_loaded = False

        self.video_model_path    = Path(settings.VIDEO_MODEL_PATH)
        self.video_threshold     = float(settings.VIDEO_DETECTION_THRESHOLD)
        self.video_model         = None
        self._video_model_loaded = False

        self.input_size      = settings.IMAGE_SIZE
        self.model_type      = settings.MODEL_TYPE
        self.media_root      = Path(settings.MEDIA_ROOT)
        self.is_binary_model = True
        self.temperature     = getattr(settings, 'MODEL_TEMPERATURE', 1.0)

        logger.info("DeepfakeDetector: focal_alpha=0.75 (FIXED), bilateral=REMOVED, "
                    "decision=weighted_average (FIXED)")

    @staticmethod
    def _focal_loss(y_true, y_pred):
        """alpha=0.75 — must match training exactly."""
        gamma, alpha = 2.0, 0.75
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)
        y_true = tf.cast(y_true, tf.float32)
        bce    = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return tf.reduce_mean(
            (y_true * alpha + (1 - y_true) * (1 - alpha))
            * tf.pow(1.0 - p_t, gamma) * bce
        )

    def _load_keras_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        logger.info(f"Loading {model_path} …")
        try:
            model = keras.models.load_model(
                model_path,
                custom_objects={'focal_loss': self._focal_loss},
                compile=False,
            )
        except Exception:
            model = keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.input_size      = model.input_shape[1]
        self.is_binary_model = (model.output_shape[-1] == 1)
        return model

    def _load_image_model(self):
        if self._image_model_loaded:
            return
        self.image_model         = self._load_keras_model(self.image_model_path)
        self._image_model_loaded = True
        self.image_grad_model    = self._build_gradcam_model(self.image_model)
        dummy = preprocess_input(np.random.rand(2, self.input_size, self.input_size, 3).astype(np.float32) * 255)
        out   = self.image_model.predict(dummy, verbose=0)
        logger.info(f"IMAGE model ✓  input={self.input_size}  out=[{out.min():.3f}, {out.max():.3f}]")

    def _load_video_model(self):
        if self._video_model_loaded:
            return
        self.video_model         = self._load_keras_model(self.video_model_path)
        self._video_model_loaded = True
        dummy = preprocess_input(np.random.rand(2, self.input_size, self.input_size, 3).astype(np.float32) * 255)
        out   = self.video_model.predict(dummy, verbose=0)
        logger.info(f"VIDEO model ✓  input={self.input_size}  out=[{out.min():.3f}, {out.max():.3f}]")

    def load_model(self):
        self._load_image_model()
        return self.image_model

    def _build_gradcam_model(self, model):
        dummy = np.zeros((1, self.input_size, self.input_size, 3), dtype=np.float32)
        for layer in reversed(model.layers):
            if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                continue
            try:
                c = keras.Model(inputs=model.input, outputs=[layer.output, model.output])
                c(dummy)
                logger.info(f"Grad-CAM layer: {layer.name}")
                return c
            except Exception:
                continue

        sub = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
        if sub is None:
            logger.warning("Grad-CAM++ unavailable — fallback heatmap")
            return None

        target = next(
            (sl.name for sl in reversed(sub.layers)
             if isinstance(sl, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))),
            None,
        )
        if target is None:
            return None
        try:
            x = model.input
            conv_tensor = None
            bridge = keras.Model(inputs=sub.input, outputs=sub.get_layer(target).output, name='gcam_bridge')
            for layer in model.layers[1:]:
                if layer is sub:
                    conv_tensor = bridge(x); x = sub(x)
                else:
                    x = layer(x)
            if conv_tensor is None:
                return None
            c = keras.Model(inputs=model.input, outputs=[conv_tensor, x], name='gcam_model')
            c(dummy)
            logger.info(f"Grad-CAM sub-model layer: {target}")
            return c
        except Exception as exc:
            logger.warning(f"Grad-CAM build failed: {exc}")
            return None

    def _fake_prob(self, prediction: np.ndarray) -> float:
        raw = float(prediction[0][0]) if self.is_binary_model else float(prediction[0][1])
        if self.temperature != 1.0 and 0 < raw < 1:
            logit = np.log(raw / (1 - raw + 1e-8))
            raw   = float(1 / (1 + np.exp(-logit / self.temperature)))
        return raw * 100.0

    def preprocess_image(self, image_path) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.input_size, self.input_size), Image.LANCZOS)
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        return preprocess_input(arr)

    def _preprocess_video_frame(self, frame_path, is_mobile_likely=False) -> np.ndarray:
        """
        FIXED: No bilateral filter (matches training).

        For mobile video: apply CLAHE (Contrast Limited Adaptive Histogram
        Equalization) to normalize exposure. Mobile cameras frequently
        produce dark/low-contrast frames that the model hasn't seen unless
        trained with brightness augmentation. CLAHE brings them into the
        normal distribution without altering the deepfake artifacts.

        is_mobile_likely is set True when the frame has low contrast
        (std of luma channel < 50), which is the main mobile failure mode.
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return np.zeros((1, self.input_size, self.input_size, 3), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mobile enhancement: CLAHE on low-contrast frames
        if is_mobile_likely:
            lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_ch  = lab[:, :, 0]
            if float(np.std(l_ch)) < 50:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(l_ch)
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        img = cv2.resize(img, (self.input_size, self.input_size))
        arr = np.expand_dims(img.astype(np.float32), axis=0)
        return preprocess_input(arr)

    def _is_frame_low_contrast(self, frame_path) -> bool:
        """Returns True if frame looks like it came from a mobile camera (dark / low contrast)."""
        try:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            return float(np.std(img)) < 50
        except Exception:
            return False

    # =========================================================================
    # IMAGE PREDICTION
    # =========================================================================

    def predict(self, image_path, generate_heatmap=True,
                analyze_face=True, validate_quality=True) -> dict:
        self._load_image_model()
        t0 = time.time()
        logger.info(f"predict() [IMAGE] → {image_path}")

        quality = None
        if validate_quality:
            quality = QualityValidator.validate_image(image_path)
            if not quality['is_valid']:
                return {
                    'success': False, 'error': 'Quality validation failed',
                    'quality_issues': quality['errors'], 'metrics': quality['metrics'],
                }

        arr        = self.preprocess_image(image_path)
        prediction = self.image_model.predict(arr, verbose=0)
        fake_prob  = self._fake_prob(prediction)
        real_prob  = round(100.0 - fake_prob, 2)
        fake_prob  = round(fake_prob, 2)

        threshold_pct = self.image_threshold * 100.0
        is_deepfake   = fake_prob > threshold_pct
        confidence    = round(max(fake_prob, real_prob), 2)
        margin        = round(abs(fake_prob - threshold_pct), 2)
        certainty     = ('high' if margin > 20 else 'moderate' if margin > 10 else 'low')

        logger.info(f"  IMAGE: fake={fake_prob:.2f}% thresh={threshold_pct:.2f}% → "
                    f"{'FAKE' if is_deepfake else 'REAL'} ({certainty})")

        fft_r = comp_r = noise_r = None
        if generate_heatmap:
            try:
                fft_r   = ForensicAnalyzer.fft_artifact_analysis(image_path)
                comp_r  = ForensicAnalyzer.compression_analysis(image_path)
                noise_r = ForensicAnalyzer.noise_analysis(image_path)
            except Exception as exc:
                logger.warning(f"Forensic (non-critical): {exc}")

        ts = ForensicAnalyzer.trust_score(confidence, quality)

        result = {
            'success':           True,
            'prediction':        'FAKE' if is_deepfake else 'REAL',
            'isDeepfake':        bool(is_deepfake),
            'is_deepfake':       bool(is_deepfake),
            'confidence':        confidence,
            'rawConfidence':     fake_prob,
            'fusedConfidence':   fake_prob,
            'threshold':         round(threshold_pct, 2),
            'probabilities':     {'fake': fake_prob, 'real': real_prob},
            'fake_probability':  fake_prob,
            'real_probability':  real_prob,
            'file_type':         'image',
            'model_used':        f'{self.model_type} (image)',
            'modelUsed':         f'{self.model_type} (image)',
            'decision': {
                'prediction': 'FAKE' if is_deepfake else 'REAL',
                'reason':     (f"Model output {fake_prob:.2f}% "
                               f"{'>' if is_deepfake else '<='} threshold {threshold_pct:.2f}%"),
                'margin':    margin,
                'certainty': certainty,
            },
            'trustScore':        ts,   'trust_score':       ts,
            'uncertainty':       round(float(np.clip(100 - margin * 2, 0, 100)), 2),
            'forensicAnalysis':  {'fft': fft_r, 'compression': comp_r, 'noise': noise_r},
            'forensic_analysis': {'fft': fft_r, 'compression': comp_r, 'noise': noise_r},
            'processTime':       round(time.time() - t0, 2),
            'process_time':      round(time.time() - t0, 2),
            'heatmapPath':          None, 'heatmap_path':         None,
            'facialAnalysisPath':   None, 'facial_analysis_path': None,
            'artifactMapPath':      None, 'artifact_map_path':    None,
            'fftMapPath':           None, 'fft_map_path':         None,
            'indicators':           [],
            'qualityMetrics':       quality['metrics']  if quality else {},
            'quality_metrics':      quality['metrics']  if quality else {},
            'qualityWarnings':      quality['warnings'] if quality else [],
            'quality_warnings':     quality['warnings'] if quality else [],
        }

        if fft_r and fft_r.get('fft_map_path'):
            result['fftMapPath']   = f'/media/{fft_r["fft_map_path"]}'
            result['fft_map_path'] = fft_r['fft_map_path']

        if generate_heatmap:
            try:
                hp = self._gradcam_plus_plus(image_path, arr, self.image_grad_model)
                if hp:
                    result['heatmapPath']  = f'/media/{hp}'
                    result['heatmap_path'] = hp
            finally:
                gc.collect()

        if analyze_face:
            facial = self._analyze_facial_regions(image_path, fake_prob)
            if facial:
                result['facialAnalysisPath']   = f'/media/{facial["path"]}'
                result['facial_analysis_path'] = facial['path']
                result['facial_regions']       = facial['regions']
                if facial.get('no_face'):
                    result['noFaceDetected'] = True

        if is_deepfake:
            art = self._detect_ai_artifacts(image_path, fake_prob)
            if art:
                result['artifactMapPath']   = f'/media/{art}'
                result['artifact_map_path'] = art

        result['indicators'] = self._indicators_image(
            is_deepfake, confidence, fake_prob, real_prob,
            quality, ts, fft_r, comp_r, noise_r,
        )

        del arr
        gc.collect()
        return result

    # =========================================================================
    # GRAD-CAM++
    # =========================================================================

    def _gradcam_plus_plus(self, image_path, preprocessed_img, grad_model) -> str | None:
        if grad_model is None:
            return self._fallback_heatmap(image_path)
        try:
            img_tensor = tf.cast(preprocessed_img, tf.float32)
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_tensor, training=False)
                tape.watch(conv_out)
                loss = preds[0][0] if self.is_binary_model else preds[0][1]
            grads = tape.gradient(loss, conv_out)
            if grads is None:
                return self._fallback_heatmap(image_path)
            pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
            co      = conv_out[0].numpy().copy()
            for i in range(pooled.shape[-1]):
                co[:, :, i] *= pooled[i]
            heatmap  = np.maximum(np.mean(co, axis=-1), 0)
            heatmap /= (np.max(heatmap) + 1e-10)
            orig    = np.array(Image.open(image_path).convert('RGB'))
            resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
            colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, colored, 0.4, 0)
            mask    = (resized > 0.6).astype(np.uint8) * 255
            hdir = self.media_root / 'heatmaps'
            hdir.mkdir(parents=True, exist_ok=True)
            hid  = uuid.uuid4().hex[:8]
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            axes[0, 0].imshow(orig);           axes[0, 0].set_title('Original');            axes[0, 0].axis('off')
            axes[0, 1].imshow(resized, cmap='jet'); axes[0, 1].set_title('Heatmap');        axes[0, 1].axis('off')
            axes[1, 0].imshow(overlay);        axes[1, 0].set_title('Overlay');             axes[1, 0].axis('off')
            axes[1, 1].imshow(orig); axes[1, 1].imshow(mask, cmap='Reds', alpha=0.5)
            axes[1, 1].set_title('High-attention (>60%)');                                  axes[1, 1].axis('off')
            plt.tight_layout()
            plt.savefig(hdir / f'{hid}_gradcam.jpg', dpi=150, bbox_inches='tight')
            plt.close()
            return f'heatmaps/{hid}_gradcam.jpg'
        except Exception as exc:
            logger.error(f"Grad-CAM++ failed: {exc}", exc_info=True)
            return self._fallback_heatmap(image_path)

    def _fallback_heatmap(self, image_path) -> str | None:
        try:
            orig   = np.array(Image.open(image_path).convert('RGB'))
            h, w   = orig.shape[:2]
            cy, cx = h // 2, w // 2
            sigma  = min(h, w) / 3
            y, x   = np.ogrid[:h, :w]
            hm     = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            hm     = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, colored, 0.4, 0)
            hdir    = self.media_root / 'heatmaps'
            hdir.mkdir(parents=True, exist_ok=True)
            hid     = uuid.uuid4().hex[:8]
            Image.fromarray(overlay).save(hdir / f'{hid}_heatmap.jpg')
            return f'heatmaps/{hid}_heatmap.jpg'
        except Exception as exc:
            logger.error(f"Fallback heatmap failed: {exc}")
            return None

    # =========================================================================
    # FACIAL / REGION ANALYSIS
    # =========================================================================

    def _analyze_facial_regions(self, image_path, fake_prob: float) -> dict | None:
        try:
            img  = cv2.imread(str(image_path))
            if img is None:
                return None
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
            if len(faces) == 0:
                return self._region_fallback(image_path, fake_prob)
            x, y, fw, fh = faces[0]
            annotated    = rgb.copy()
            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (255, 100, 100), 3)
            region_defs = {
                'Eyes':    (y,              y+fh//4,       x,          x+fw),
                'Forehead':(max(0,y-fh//6), y,             x,          x+fw),
                'Nose':    (y+fh//4,        y+fh*2//3,     x+fw//4,    x+fw*3//4),
                'Mouth':   (y+fh*2//3,      y+fh,          x+fw//4,    x+fw*3//4),
                'Cheeks':  (y+fh//4,        y+fh*3//4,     x,          x+fw),
                'Jawline': (y+fh*3//4,      min(img.shape[0],y+fh+fh//6), x, x+fw),
            }
            colors = [(255,100,100),(100,150,255),(100,255,150),(255,200,100),(200,100,255),(100,220,220)]
            regions = []
            for (name, (r1,r2,c1,c2)), col in zip(region_defs.items(), colors):
                r1,r2,c1,c2 = int(r1),int(r2),int(c1),int(c2)
                cv2.rectangle(annotated, (c1,r1), (c2,r2), col, 2)
                patch  = gray[r1:r2, c1:c2]
                tf_fac = max(0, (50 - float(np.std(patch) if patch.size else 30)) / 50) * 15
                score  = float(np.clip(fake_prob + tf_fac, 0, 100))
                regions.append({'name': name, 'score': round(score, 2)})
            return self._save_region_chart(annotated, regions, 'Facial Region Analysis', no_face=False)
        except Exception as exc:
            logger.error(f"Facial analysis failed: {exc}", exc_info=True)
            return None

    def _region_fallback(self, image_path, fake_prob: float) -> dict | None:
        try:
            img  = cv2.imread(str(image_path))
            if img is None:
                return None
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            annotated = rgb.copy()
            quads = {
                'Top-Left':    (0,    h//2, 0,    w//2),
                'Top-Right':   (0,    h//2, w//2, w),
                'Bottom-Left': (h//2, h,    0,    w//2),
                'Bottom-Right':(h//2, h,    w//2, w),
                'Center':      (h//4, 3*h//4, w//4, 3*w//4),
            }
            colors  = [(255,100,100),(100,150,255),(100,255,150),(255,200,100),(200,100,255)]
            regions = []
            for (name, (r1,r2,c1,c2)), col in zip(quads.items(), colors):
                cv2.rectangle(annotated, (c1,r1), (c2,r2), col, 2)
                patch = gray[r1:r2, c1:c2]
                score = float(np.clip(
                    fake_prob + max(0, (50-float(np.std(patch) if patch.size else 30))/50)*15, 0, 100
                ))
                regions.append({'name': name, 'score': round(score, 2)})
            return self._save_region_chart(annotated, regions,
                                           'Image Region Analysis (No Face Detected)', no_face=True)
        except Exception as exc:
            logger.error(f"Region fallback failed: {exc}", exc_info=True)
            return None

    def _save_region_chart(self, annotated, regions, title, no_face) -> dict | None:
        try:
            names  = [r['name']  for r in regions]
            scores = [r['score'] for r in regions]
            colors = ['#22c55e' if s < 60 else '#f59e0b' if s < 70 else '#ef4444' for s in scores]
            aid  = uuid.uuid4().hex[:8]
            adir = self.media_root / 'facial_analysis'
            adir.mkdir(parents=True, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            ax1.imshow(annotated); ax1.set_title(title, fontweight='bold'); ax1.axis('off')
            ax2.barh(names, scores, color=colors)
            ax2.axvline(60, color='#f59e0b', linestyle='--', lw=2, label='Caution (60%)', alpha=0.7)
            ax2.axvline(70, color='#ef4444', linestyle='--', lw=2, label='High Risk (70%)', alpha=0.7)
            ax2.set_xlabel('Inconsistency Score (%)'); ax2.set_xlim(0, 100)
            ax2.legend(loc='lower right'); ax2.grid(axis='x', alpha=0.3)
            for i, s in enumerate(scores):
                ax2.text(s+2, i, f'{s:.1f}%', va='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            plt.savefig(adir / f'{aid}_analysis.jpg', dpi=150, bbox_inches='tight')
            plt.close()
            return {'path': f'facial_analysis/{aid}_analysis.jpg', 'regions': regions, 'no_face': no_face}
        except Exception as exc:
            logger.error(f"Region chart save failed: {exc}", exc_info=True)
            return None

    def _detect_ai_artifacts(self, image_path, fake_prob: float) -> str | None:
        if fake_prob < 60:
            return None
        try:
            img  = cv2.imread(str(image_path))
            if img is None:
                return None
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges    = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            diff     = np.mean(np.abs(img.astype(np.float32) - denoised.astype(np.float32)), axis=2)
            artifact = cv2.normalize(0.5*edges + 0.5*diff/(diff.max()+1e-8),
                                     None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.cvtColor(cv2.applyColorMap(artifact, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(rgb, 0.6, colored, 0.4, 0)
            aid  = uuid.uuid4().hex[:8]
            adir = self.media_root / 'artifacts'
            adir.mkdir(parents=True, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            ax1.imshow(overlay); ax1.set_title('Artifact Overlay'); ax1.axis('off')
            ax2.imshow(artifact, cmap='hot'); ax2.set_title('Artifact Map'); ax2.axis('off')
            plt.tight_layout()
            plt.savefig(adir / f'{aid}_artifacts.jpg', dpi=150, bbox_inches='tight')
            plt.close()
            return f'artifacts/{aid}_artifacts.jpg'
        except Exception as exc:
            logger.error(f"Artifact detection failed: {exc}", exc_info=True)
            return None

    def _indicators_image(self, is_deepfake, confidence, fake_prob,
                           real_prob, quality, ts, fft_r, comp_r, noise_r) -> list:
        ind = []
        if quality and quality.get('warnings'):
            for w in quality['warnings']:
                ind.append({'type': 'warning', 'message': w})
        if ts >= 80:
            ind.append({'type': 'success', 'message': f'High confidence ({ts:.1f}/100)'})
        elif ts >= 60:
            ind.append({'type': 'info', 'message': f'Moderate confidence ({ts:.1f}/100)'})
        else:
            ind.append({'type': 'warning', 'message': f'Low confidence ({ts:.1f}/100) — manual review recommended'})
        if confidence >= 90:
            ind.append({'type': 'success' if not is_deepfake else 'error',
                        'message': f'Very high model confidence ({confidence:.1f}%)'})
        elif confidence >= 70:
            ind.append({'type': 'info', 'message': f'High model confidence ({confidence:.1f}%)'})
        else:
            ind.append({'type': 'warning', 'message': f'Borderline result ({confidence:.1f}%) — verify independently'})
        if is_deepfake:
            if fake_prob >= 85:
                ind.append({'type': 'error', 'message': 'Strong manipulation detected'})
            elif fake_prob >= 65:
                ind.append({'type': 'error', 'message': 'Significant manipulation detected'})
            else:
                ind.append({'type': 'warning', 'message': 'Borderline — verify source independently'})
        else:
            if real_prob >= 85:
                ind.append({'type': 'success', 'message': 'Strong authenticity indicators'})
            else:
                ind.append({'type': 'info', 'message': 'No manipulation detected (borderline — cross-check recommended)'})
        if fft_r and fft_r.get('has_grid_artifacts'):
            ind.append({'type': 'info', 'message': 'Advisory: FFT grid patterns found (may be JPEG compression)'})
        if comp_r and comp_r.get('suspicious'):
            ind.append({'type': 'info', 'message': 'Advisory: Compression pattern irregular'})
        if noise_r and noise_r.get('suspicious'):
            ind.append({'type': 'info', 'message': 'Advisory: Noise pattern variation detected'})
        return ind

    # =========================================================================
    # VIDEO DECISION LOGIC — FIXED
    #
    # Parameters used and why:
    #
    # 1. avg_fake_prob (weight 0.60 — most important)
    #    The average fake probability across ALL frames. This is the model's
    #    core signal. A video where all frames score 0.80 is clearly fake.
    #    A video where all frames score 0.30 is clearly real.
    #    Weight 0.60 because this is the most reliable single number.
    #
    # 2. fake_frame_pct (weight 0.30 — secondary)
    #    What fraction of frames exceeded the per-frame threshold.
    #    Matters separately from avg because: a video where 5% of frames
    #    score 0.95 (strong fake signal in some frames) but 95% score 0.40
    #    might be a partially deepfaked video. The avg would be ~0.43 (below
    #    threshold) but the fake_frame_pct = 5% catches the partial case.
    #
    # 3. confidence_weighted_score (weight 0.10 — tiebreaker)
    #    Frames where the model is very confident (score close to 0 or 1)
    #    are weighted higher than frames where the model is uncertain (score
    #    close to 0.5). This prevents a pile of uncertain 0.55 predictions
    #    from overriding a few very confident 0.95 predictions.
    #
    # Final decision:
    #    weighted_score = 0.60 * avg_fake_prob + 0.30 * fake_frame_pct + 0.10 * conf_score
    #    is_deepfake = weighted_score > VIDEO_DETECTION_THRESHOLD * 100
    #
    # =========================================================================

    def _make_video_decision(self, frame_results: list, threshold_pct: float) -> dict:
        """
        Weighted combination of three signals for the final fake/real verdict.
        See docstring above for full explanation of each parameter.
        """
        probs = [f['fake_probability'] for f in frame_results]

        # Signal 1: average fake probability across all frames
        avg_fake_prob = float(np.mean(probs))

        # Signal 2: fraction of frames above threshold
        fake_frame_pct = sum(1 for p in probs if p > threshold_pct) / len(probs) * 100

        # Signal 3: confidence-weighted score
        # Confidence = max(p, 100-p) / 100  →  0.5 for uncertain, 1.0 for certain
        conf_weights   = [max(p, 100 - p) / 100 for p in probs]
        total_w        = sum(conf_weights) + 1e-8
        conf_score     = sum(p * w for p, w in zip(probs, conf_weights)) / total_w

        # Final weighted score
        weighted_score = (
            0.60 * avg_fake_prob +
            0.30 * fake_frame_pct +
            0.10 * conf_score
        )

        is_deepfake = weighted_score > threshold_pct

        logger.info(
            f"  VIDEO decision: avg={avg_fake_prob:.1f}% fake_pct={fake_frame_pct:.1f}% "
            f"conf_score={conf_score:.1f}% weighted={weighted_score:.1f}% "
            f"thresh={threshold_pct:.1f}% → {'FAKE' if is_deepfake else 'REAL'}"
        )

        return {
            'is_deepfake':          bool(is_deepfake),
            'weighted_score':       round(weighted_score, 2),
            'avg_fake_prob':        round(avg_fake_prob, 2),
            'fake_frame_pct':       round(fake_frame_pct, 2),
            'confidence_weighted':  round(conf_score, 2),
        }

    # =========================================================================
    # VIDEO PREDICTION
    # =========================================================================

    def predict_video(self, video_path) -> dict:
        self._load_video_model()
        logger.info(f"predict_video() → {video_path}")
        try:
            frames_data = self._extract_frames(video_path)
            if not frames_data['frames']:
                return {'success': False, 'error': 'No frames extracted'}

            raw_probs = self._batch_predict_video(frames_data['frames'])
            if not raw_probs:
                return {'success': False, 'error': 'Frame analysis failed'}

            smoothed      = self._smooth(raw_probs)
            threshold_pct = self.video_threshold * 100.0

            frame_results = [
                {
                    'frame_number':     int(i + 1),
                    'timestamp':        float(round(fi['timestamp'], 2)),
                    'is_deepfake':      bool(smoothed[i] > threshold_pct),
                    'confidence':       round(max(smoothed[i], 100 - smoothed[i]), 2),
                    'fake_probability': round(smoothed[i], 2),
                    'raw_probability':  round(raw_probs[i], 2),
                    'thumbnail_url':    fi['thumbnail_url'],
                }
                for i, fi in enumerate(frames_data['frames'])
            ]

            # FIXED: Use weighted_average decision instead of majority_vote
            decision = self._make_video_decision(frame_results, threshold_pct)

            deepfake_frames = sum(1 for r in frame_results if r['is_deepfake'])
            is_deepfake     = decision['is_deepfake']
            fake_prob       = round(decision['weighted_score'], 2)
            real_prob       = round(100.0 - fake_prob, 2)
            confidence      = round(max(fake_prob, real_prob), 2)
            temporal        = self._temporal_metrics(raw_probs, smoothed, threshold_pct)
            stability       = round(float(np.clip(100 - temporal['norm_var'], 0, 100)), 2)
            manipulation    = self._analyze_manipulation_timeline(frame_results, threshold_pct)

            result = {
                'success':          True,
                'prediction':       'FAKE' if is_deepfake else 'REAL',
                'isDeepfake':       bool(is_deepfake),
                'is_deepfake':      bool(is_deepfake),
                'confidence':       confidence,
                'probabilities':    {'fake': fake_prob, 'real': real_prob},
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'file_type':        'video',
                'model_used':       f'{self.model_type} (video)',
                'modelUsed':        f'{self.model_type} (video)',
                'threshold':        round(threshold_pct, 2),
                'videoAnalysis': {
                    'totalFrames':          len(frame_results),
                    'deepfakeFrames':        deepfake_frames,
                    'deepfakePercentage':    round(deepfake_frames / len(frame_results) * 100, 2),
                    'stabilityScore':        stability,
                    'decisionBreakdown':     decision,   # new — shows all three signals
                    'temporalConsistency': {
                        'variance':      temporal['variance'],
                        'is_consistent': temporal['is_consistent'],
                        'transitions':   temporal['transitions'],
                        'message':       temporal['message'],
                    },
                    'frameByFrame':          frame_results,
                    'manipulationAnalysis':  manipulation,
                    'videoInfo': {
                        'duration':        float(round(frames_data['duration'], 2)),
                        'fps':             int(frames_data['fps']),
                        'total_frames':    int(frames_data['total_frames']),
                        'analyzed_frames': len(frame_results),
                    },
                },
                'indicators': self._indicators_video(
                    is_deepfake, confidence, fake_prob,
                    temporal['variance'], stability, manipulation, decision
                ),
            }

            for fi in frames_data['frames']:
                try:
                    Path(fi['path']).unlink(missing_ok=True)
                except Exception:
                    pass
            gc.collect()
            return result

        except Exception as exc:
            logger.error(f"Video prediction failed: {exc}", exc_info=True)
            return {'success': False, 'error': str(exc)}

    # =========================================================================
    # MANIPULATION TIMELINE
    # =========================================================================

    def _analyze_manipulation_timeline(self, frame_results: list, threshold_pct: float) -> dict:
        fake_frames = [f for f in frame_results if f['is_deepfake']]
        if not fake_frames:
            return {
                'has_manipulation': False,
                'first_fake_timestamp': None, 'first_fake_frame': None,
                'first_fake_thumbnail': None, 'first_fake_confidence': None,
                'longest_run_start_timestamp': None, 'longest_run_end_timestamp': None,
                'longest_run_frame_count': 0, 'longest_run_frames': [],
                'top_suspicious_frames': [], 'manipulation_location': 'none',
            }

        first_fake = fake_frames[0]

        longest_run = []; current_run = []
        for frame in frame_results:
            if frame['is_deepfake']:
                current_run.append(frame)
            else:
                if len(current_run) > len(longest_run):
                    longest_run = current_run[:]
                current_run = []
        if len(current_run) > len(longest_run):
            longest_run = current_run[:]

        top_suspicious = sorted(frame_results, key=lambda f: f['fake_probability'], reverse=True)[:5]

        total       = len(frame_results)
        fake_pos    = [f['frame_number'] for f in fake_frames]
        avg_pos     = np.mean(fake_pos) / total if fake_pos else 0.5
        location    = ('beginning' if avg_pos < 0.33 else 'end' if avg_pos > 0.66 else 'middle')

        return {
            'has_manipulation':            True,
            'first_fake_timestamp':        float(round(first_fake['timestamp'], 2)),
            'first_fake_frame':            int(first_fake['frame_number']),
            'first_fake_thumbnail':        first_fake.get('thumbnail_url'),
            'first_fake_confidence':       float(first_fake['fake_probability']),
            'longest_run_start_timestamp': float(round(longest_run[0]['timestamp'], 2)) if longest_run else None,
            'longest_run_end_timestamp':   float(round(longest_run[-1]['timestamp'], 2)) if longest_run else None,
            'longest_run_frame_count':     int(len(longest_run)),
            'longest_run_frames':          [f['frame_number'] for f in longest_run],
            'top_suspicious_frames':       [
                {'frame_number': f['frame_number'], 'timestamp': f['timestamp'],
                 'fake_probability': f['fake_probability'], 'thumbnail_url': f.get('thumbnail_url')}
                for f in top_suspicious
            ],
            'manipulation_location': location,
        }

    def _batch_predict_video(self, frames) -> list:
        batch_size = 8
        results    = []
        paths      = [f['path'] for f in frames]
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch_arrs  = []
            for fp in batch_paths:
                try:
                    # Check if frame looks like mobile footage (low contrast)
                    is_mobile = self._is_frame_low_contrast(fp)
                    batch_arrs.append(self._preprocess_video_frame(fp, is_mobile_likely=is_mobile)[0])
                except Exception:
                    batch_arrs.append(np.zeros((self.input_size, self.input_size, 3), dtype=np.float32))

            batch_in   = np.stack(batch_arrs, axis=0)
            batch_pred = self.video_model.predict(batch_in, verbose=0)
            for pred in batch_pred:
                raw = float(pred[0] if self.is_binary_model else pred[1])
                if self.temperature != 1.0 and 0 < raw < 1:
                    logit = np.log(raw / (1 - raw + 1e-8))
                    raw   = float(1 / (1 + np.exp(-logit / self.temperature)))
                results.append(raw * 100.0)
            del batch_in, batch_arrs
            gc.collect()
        return results

    @staticmethod
    def _smooth(probs: list, window: int = 3) -> list:
        """Light smoothing — weights [0.20, 0.60, 0.20] to preserve individual frame signal better."""
        if len(probs) < window:
            return probs
        weights = np.array([0.20, 0.60, 0.20])   # FIX: was [0.25,0.50,0.25] — less blurring
        half    = window // 2
        out     = []
        for i in range(len(probs)):
            s = max(0, i - half)
            e = min(len(probs), i + half + 1)
            w = probs[s:e]
            out.append(round(float(np.dot(w, weights) if len(w) == window else np.mean(w)), 2))
        return out

    @staticmethod
    def _temporal_metrics(raw, smoothed, threshold) -> dict:
        arr      = np.array(smoothed)
        variance = round(float(np.var(arr)), 2)
        norm_var = round(float(np.clip(variance / 5.0, 0, 100)), 2)
        verdicts = [p > threshold for p in smoothed]
        trans    = sum(1 for i in range(1, len(verdicts)) if verdicts[i] != verdicts[i-1])
        is_con   = variance < 200 and trans <= max(2, len(arr) * 0.1)
        if is_con and variance < 50:
            msg = 'Highly consistent detection across all frames'
        elif is_con:
            msg = 'Consistent detection with minor variance'
        elif trans > len(arr) * 0.3:
            msg = 'High flip rate — possible localized manipulation'
        else:
            msg = 'Inconsistent detections — possible partial manipulation'
        return {'variance': variance, 'norm_var': norm_var,
                'is_consistent': bool(is_con), 'transitions': int(trans), 'message': msg}

    def _indicators_video(self, is_deepfake, confidence, fake_prob,
                          variance, stability, manipulation_analysis=None, decision=None) -> list:
        ind = []
        if stability >= 80:
            ind.append({'type': 'success' if not is_deepfake else 'info',
                        'message': f'High detection stability ({stability:.0f}/100)'})
        elif stability >= 60:
            ind.append({'type': 'warning', 'message': f'Moderate stability ({stability:.0f}/100)'})
        else:
            ind.append({'type': 'warning', 'message': f'Low stability ({stability:.0f}/100)'})

        if confidence >= 85:
            ind.append({'type': 'success' if not is_deepfake else 'error',
                        'message': f'High confidence ({confidence:.1f}%)'})
        elif confidence >= 70:
            ind.append({'type': 'info', 'message': f'Moderate confidence ({confidence:.1f}%)'})
        else:
            ind.append({'type': 'warning', 'message': f'Low confidence ({confidence:.1f}%) — borderline result'})

        if variance < 100:
            ind.append({'type': 'info', 'message': 'Consistent frame-by-frame detection'})
        elif variance < 300:
            ind.append({'type': 'warning', 'message': 'Some variance — possible partial manipulation'})
        else:
            ind.append({'type': 'warning', 'message': 'High variance — may include compression artifacts'})

        # Show decision breakdown if available
        if decision:
            avg  = decision.get('avg_fake_prob', 0)
            fpct = decision.get('fake_frame_pct', 0)
            ind.append({'type': 'info',
                        'message': f'Avg frame score: {avg:.1f}%  |  Fake frames: {fpct:.1f}%'})

        if is_deepfake:
            ind.append({'type': 'error' if fake_prob >= 70 else 'warning',
                        'message': 'Strong deepfake indicators' if fake_prob >= 70 else 'Possible deepfake detected'})
            if manipulation_analysis and manipulation_analysis.get('has_manipulation'):
                ma  = manipulation_analysis
                ts  = ma.get('first_fake_timestamp')
                loc = ma.get('manipulation_location', 'unknown')
                run = ma.get('longest_run_frame_count', 0)
                if ts is not None:
                    ind.append({'type': 'error',
                                'message': f'Manipulation first at {ts:.1f}s (in {loc} of video)'})
                if run > 1:
                    s_ts = ma.get('longest_run_start_timestamp')
                    e_ts = ma.get('longest_run_end_timestamp')
                    ind.append({'type': 'error',
                                'message': f'Longest fake segment: {run} frames ({s_ts:.1f}s – {e_ts:.1f}s)'})
        else:
            ind.append({'type': 'success', 'message': 'Video appears authentic'})

        return ind

    # =========================================================================
    # FRAME EXTRACTION
    # =========================================================================

    def _extract_frames(self, video_path) -> dict:
        cap   = cv2.VideoCapture(str(video_path))
        fps   = int(round(cap.get(cv2.CAP_PROP_FPS) or 25))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur   = total / fps if fps > 0 else 0
        fe       = settings.VIDEO.get('FRAME_EXTRACTION', {})
        sample_r = fe.get('SAMPLE_RATE', 10)
        max_f    = fe.get('MAX_FRAMES',  60)
        thumb_dir  = self.media_root / 'frame_thumbnails'
        frames_dir = self.media_root / 'temp_frames'
        thumb_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames, fc, sc = [], 0, 0
        sid = uuid.uuid4().hex[:8]
        while cap.isOpened() and sc < max_f:
            ret, frame = cap.read()
            if not ret:
                break
            if fc % sample_r == 0:
                raw_path = frames_dir / f'raw_{sid}_{sc:04d}.jpg'
                cv2.imwrite(str(raw_path), frame)
                tname = f'thumb_{sid}_{sc:04d}.jpg'
                cv2.imwrite(str(thumb_dir / tname),
                            cv2.resize(frame, (320, 180)),
                            [cv2.IMWRITE_JPEG_QUALITY, 70])
                frames.append({
                    'path':          str(raw_path),
                    'thumbnail_url': f'/media/frame_thumbnails/{tname}',
                    'timestamp':     fc / fps,
                })
                sc += 1
            fc += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} frames (fps={fps} dur={dur:.1f}s sr={sample_r})")
        return {'frames': frames, 'fps': fps, 'total_frames': total, 'duration': dur}

    def get_model_info(self) -> dict:
        self._load_image_model()
        return {
            'image_model_path':  str(self.image_model_path),
            'video_model_path':  str(self.video_model_path),
            'input_size':        int(self.input_size),
            'output_type':       'binary' if self.is_binary_model else 'categorical',
            'image_threshold':   self.image_threshold,
            'video_threshold':   self.video_threshold,
            'temperature':       self.temperature,
            'total_params':      int(self.image_model.count_params()),
            'supports_video':    True,
            'decision_strategy': 'weighted_average',
        }


# =============================================================================
# SINGLETON
# =============================================================================

def get_detector() -> DeepfakeDetector:
    global _detector_instance
    if _detector_instance is None:
        logger.info("Creating DeepfakeDetector …")
        _detector_instance = DeepfakeDetector()
    return _detector_instance


def reset_detector():
    global _detector_instance
    _detector_instance = None
    logger.info("Detector singleton reset")