"""
settings.py — Deepfake Detection Django Settings
Thresholds corrected from actual training metadata.
"""

from pathlib import Path
from django.contrib.messages import constants as messages

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY    = 'django-insecure-your-secret-key-here'
DEBUG         = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
    'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF     = 'deepfake_detection.urls'
WSGI_APPLICATION = 'deepfake_detection.wsgi.application'

TEMPLATES = [{
    'BACKEND':  'django.template.backends.django.DjangoTemplates',
    'DIRS':     [],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'db.sqlite3'}}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'UTC'
USE_I18N      = True
USE_TZ        = True

STATIC_URL       = 'static/'
STATIC_ROOT      = BASE_DIR / 'staticfiles'
_static_dir      = BASE_DIR / 'static'
STATICFILES_DIRS = [_static_dir] if _static_dir.exists() else []

MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_URL           = 'detector:login'
LOGIN_REDIRECT_URL  = 'detector:index'
LOGOUT_REDIRECT_URL = 'detector:index'

SESSION_COOKIE_AGE         = 1_209_600
SESSION_SAVE_EVERY_REQUEST = True
SESSION_COOKIE_HTTPONLY    = True
SESSION_COOKIE_SAMESITE    = 'Lax'
CSRF_COOKIE_HTTPONLY       = False
CSRF_COOKIE_SAMESITE       = 'Lax'

MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'
MESSAGE_TAGS = {
    messages.DEBUG: 'debug', messages.INFO: 'info', messages.SUCCESS: 'success',
    messages.WARNING: 'warning', messages.ERROR: 'error',
}

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_TYPE = 'EfficientNet-B1'
IMAGE_SIZE = 240

IMAGE_MODEL_PATH = BASE_DIR / 'detector' / 'ml_models' / 'deepfake_model_b1.h5'
VIDEO_MODEL_PATH = BASE_DIR / 'detector' / 'ml_models' / 'deepfake_video_model.h5'

# ── Detection thresholds ──────────────────────────────────────────────────────

IMAGE_DETECTION_THRESHOLD = 0.49

VIDEO_DETECTION_THRESHOLD = 0.60

# Temperature: 1.0 = no change. Models are not overconfident.
MODEL_TEMPERATURE = 1.0

# Legacy
MODEL_PATH          = IMAGE_MODEL_PATH
DETECTION_THRESHOLD = IMAGE_DETECTION_THRESHOLD
CLASS_INDICES       = {'real': 0, 'fake': 1}

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE         = 50  * 1024 * 1024
MAX_VIDEO_FILE_SIZE   = 200 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/jpg']

# ── Video configuration ───────────────────────────────────────────────────────
VIDEO = {
    'MAX_FILE_SIZE': 200 * 1024 * 1024,
    'ALLOWED_FORMATS': [
        'video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm',
        'video/quicktime', 'video/x-msvideo', 'video/x-matroska',
    ],
    'ALLOWED_EXTENSIONS': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    'FRAME_EXTRACTION': {'SAMPLE_RATE': 10, 'MIN_FRAMES': 5, 'MAX_FRAMES': 60,
                         'DETECTION_STRATEGY': 'majority_vote'},
    'ENABLE_TEMPORAL_ANALYSIS': True,
}

# ── Quality validation ────────────────────────────────────────────────────────
QUALITY_VALIDATION = {
    'ENABLE_BLUR_DETECTION': True, 'BLUR_THRESHOLD': 50.0,
    'MIN_RESOLUTION': (224, 224), 'WARN_RESOLUTION': (480, 480),
    'ENABLE_NOISE_DETECTION': True, 'MAX_NOISE_LEVEL': 50.0,
}

# ── Training reference values ─────────────────────────────────────────────────
TRAINING = {
    'BATCH_SIZE': 16, 'EPOCHS': 60,
    'LEARNING_RATE_PHASE1': 0.001, 'LEARNING_RATE_PHASE2': 0.00001,
    'VALIDATION_SPLIT': 0.2, 'EARLY_STOPPING_PATIENCE': 12,
    'PHASE1_EPOCHS': 25, 'PHASE2_EPOCHS': 40, 'UNFREEZE_LAYERS': 20,
}

DATASET = {
    'IMAGE_TRAIN': BASE_DIR / 'dataset' / 'images'       / 'train',
    'IMAGE_VAL':   BASE_DIR / 'dataset' / 'images'       / 'val',
    'IMAGE_TEST':  BASE_DIR / 'dataset' / 'images'       / 'test',
    'VIDEO_TRAIN': BASE_DIR / 'dataset' / 'video_frames' / 'train',
    'VIDEO_VAL':   BASE_DIR / 'dataset' / 'video_frames' / 'val',
    'VIDEO_TEST':  BASE_DIR / 'dataset' / 'video_frames' / 'test',
    'TRAIN': BASE_DIR / 'dataset' / 'train',
    'VAL':   BASE_DIR / 'dataset' / 'val',
    'TEST':  BASE_DIR / 'dataset' / 'test',
}

AUGMENTATION = {
    'rotation_range': 15, 'width_shift_range': 0.10, 'height_shift_range': 0.10,
    'zoom_range': 0.12, 'horizontal_flip': True, 'brightness_range': [0.80, 1.20],
    'shear_range': 0.08, 'fill_mode': 'nearest',
}

INFERENCE = {
    'BATCH_SIZE': 8, 'GENERATE_HEATMAP': True, 'HEATMAP_TYPE': 'gradcam++',
    'ENABLE_FACIAL_ANALYSIS': True, 'ENABLE_ARTIFACT_DETECTION': True,
}

# ── Logging ───────────────────────────────────────────────────────────────────
(BASE_DIR / 'logs').mkdir(exist_ok=True)

LOGGING = {
    'version': 1, 'disable_existing_loggers': False,
    'formatters': {'verbose': {'format': '{levelname} {asctime} {module} {message}', 'style': '{'}},
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'verbose'},
        'file':    {'class': 'logging.FileHandler',
                    'filename': BASE_DIR / 'logs' / 'deepfake.log', 'formatter': 'verbose'},
    },
    'root':    {'handlers': ['console', 'file'], 'level': 'INFO'},
    'loggers': {'detector': {'handlers': ['console', 'file'], 'level': 'INFO', 'propagate': False}},
}

TF_CONFIG = {'ENABLE_MIXED_PRECISION': True, 'ENABLE_XLA': False, 'MEMORY_GROWTH': True}