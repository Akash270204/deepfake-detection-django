from pathlib import Path

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-here'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
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

ROOT_URLCONF = 'deepfake_detection.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'deepfake_detection.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',  # Commented out since we don't have this directory
]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ============================================================================
# MODEL CONFIGURATION - Updated for Improved Training
# ============================================================================

MODEL_PATH = BASE_DIR / 'detector' / 'ml_models' / 'deepfake_model.h5'
MODEL_TYPE = 'EfficientNet-B0'  # ✅ FIXED: Changed from B1 to B0
IMAGE_SIZE = 224  # ✅ Correct for EfficientNetB0
DETECTION_THRESHOLD = 0.70  # For binary model: output < 0.7 = real, output >= 0.7 = fake

# Model Output Configuration
OUTPUT_TYPE = 'binary'  # 'binary' (1 output) or 'categorical' (2 outputs)
# NOTE: New model uses binary classification (single sigmoid output)


# ============================================================================
# TRAINING CONFIGURATION - Optimized for Small Dataset
# ============================================================================

TRAINING = {
    'BATCH_SIZE': 16,  # ✅ FIXED: Reduced from 24 to 8 (better for limited data)
    'EPOCHS': 30,  # ✅ FIXED: Increased from 40 to 50 (with early stopping)
    'LEARNING_RATE_PHASE1': 0.001,  # ✅ NEW: Phase 1 (head training)
    'LEARNING_RATE_PHASE2': 0.0001,  # ✅ NEW: Phase 2 (fine-tuning)
    'VALIDATION_SPLIT': 0.2,
    'EARLY_STOPPING_PATIENCE': 10,  # ✅ FIXED: Increased from 7 to 10
    
    # Progressive training settings
    'PHASE1_EPOCHS': 15,  # Train head only
    'PHASE2_EPOCHS': 15,  # Fine-tune with unfrozen layers
    'UNFREEZE_LAYERS': 20,  # Number of layers to unfreeze in phase 2
}


# ============================================================================
# DATASET PATHS
# ============================================================================

DATASET = {
    'TRAIN': BASE_DIR / 'dataset' / 'train',
    'VAL': BASE_DIR / 'dataset' / 'val',
    'TEST': BASE_DIR / 'dataset' / 'test'
}


# ============================================================================
# UPLOAD CONFIGURATION
# ============================================================================

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/jpg']


# ============================================================================
# CLASS CONFIGURATION
# ============================================================================

CLASS_INDICES = {
    'real': 0,  # Real images → 0
    'fake': 1   # Fake images → 1
}

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION = {
    'rotation_range': 15,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'deepfake.log',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'detector': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Create logs directory if it doesn't exist
(BASE_DIR / 'logs').mkdir(exist_ok=True)


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# TensorFlow configuration
TF_CONFIG = {
    'ENABLE_MIXED_PRECISION': True,  # Use float16 for faster training
    'ENABLE_XLA': True,  # XLA compilation for optimization
    'MEMORY_GROWTH': True,  # Dynamic GPU memory allocation
}


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE = {
    'BATCH_SIZE': 16,  # Batch size for batch predictions
    'GENERATE_HEATMAP': True,  # Generate Grad-CAM heatmaps by default
    'CONFIDENCE_THRESHOLD': 0.7,  # Minimum confidence for "high confidence" results
}


# ============================================================================
# DATASET REQUIREMENTS (for validation)
# ============================================================================

DATASET_REQUIREMENTS = {
    'MIN_IMAGES_PER_CLASS': 100,  # Minimum images per class
    'RECOMMENDED_IMAGES_PER_CLASS': 1000,  # Recommended for good results
    'IDEAL_IMAGES_PER_CLASS': 5000,  # Ideal for production
}


# ============================================================================
# MODEL METADATA (for UI display)
# ============================================================================

MODEL_INFO = {
    'name': 'EfficientNet-B0 Deepfake Detector',
    'version': '2.0',
    'architecture': 'EfficientNetB0 + Binary Classification',
    'input_size': f'{IMAGE_SIZE}x{IMAGE_SIZE}',
    'output_type': OUTPUT_TYPE,
    'training_date': None,  # Will be set after training
    'accuracy': None,  # Will be set after training
}