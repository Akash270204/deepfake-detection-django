"""
views.py — Deepfake Detector Django Views

FIXES:
  1. dashboard: all_users query now correctly imports Count before use
     and uses annotation properly.
  2. predict: removed redundant file_type override logic that could
     cause confusion when extension and frontend disagree.
  3. _save_history: safer field access — no crash if videoAnalysis missing.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .forms import LoginForm, SignUpForm
from .ml_utils import get_detector

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}


# =============================================================================
# HELPERS
# =============================================================================

def _to_python(obj):
    """Recursively convert numpy / Path objects to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _get_file_type(filename: str) -> str:
    """Return 'video' or 'image' based on file extension."""
    ext = Path(filename).suffix.lower().strip('.')
    return 'video' if ext in VIDEO_EXTENSIONS else 'image'


# =============================================================================
# AUTHENTICATION VIEWS
# =============================================================================

def signup_view(request):
    if request.user.is_authenticated:
        return redirect('detector:index')

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            came_for_video = request.GET.get('video') or request.POST.get('video')
            messages.success(
                request,
                f'Welcome {user.username}! '
                + ('You can now analyse videos.' if came_for_video
                   else 'Account created successfully.'),
            )
            return redirect('detector:index')
    else:
        form = SignUpForm()

    return render(request, 'detector/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('detector:index')

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(
                    request,
                    f'Welcome back, {username}!'
                    + (' You can now analyse videos.'
                       if request.GET.get('video') else ''),
                )
                return redirect(request.GET.get('next', 'detector:index'))
    else:
        form = LoginForm()

    return render(request, 'detector/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('detector:index')


# =============================================================================
# PAGE VIEWS
# =============================================================================

def index(request):
    return render(request, 'detector/index.html', {
        'video_support':     True,
        'max_video_size_mb': settings.VIDEO['MAX_FILE_SIZE'] // (1024 * 1024),
        'supported_formats': settings.VIDEO['ALLOWED_EXTENSIONS'],
    })


@login_required(login_url='detector:login')
def history(request):
    try:
        from .models import DetectionHistory
        items = list(DetectionHistory.objects
                     .filter(user=request.user)
                     .order_by('-created_at')[:50])
        fake_count = sum(1 for i in items if i.is_deepfake)
        real_count = len(items) - fake_count
    except Exception as exc:
        logger.error(f"History query failed: {exc}")
        items = []
        fake_count = 0
        real_count = 0
    return render(request, 'detector/history.html', {
        'history_items': items,
        'fake_count':    fake_count,
        'real_count':    real_count,
    })


@login_required
def dashboard(request):
    if not (request.user.is_staff or request.user.is_superuser):
        messages.error(request, 'Admin privileges required.')
        return redirect('detector:index')

    from .models import DetectionHistory
    from django.contrib.auth.models import User
    from django.db.models import Avg, Count
    from django.utils import timezone
    from datetime import timedelta
    import json as _json

    now          = timezone.now()
    last_7_days  = now - timedelta(days=7)
    last_30_days = now - timedelta(days=30)

    total_users      = User.objects.count()
    active_users     = User.objects.filter(last_login__gte=last_30_days).count()
    total_detections = DetectionHistory.objects.count()
    fake_detections  = DetectionHistory.objects.filter(is_deepfake=True).count()
    real_detections  = DetectionHistory.objects.filter(is_deepfake=False).count()
    image_detections = DetectionHistory.objects.filter(file_type='image').count()
    video_detections = DetectionHistory.objects.filter(file_type='video').count()
    avg_confidence   = DetectionHistory.objects.aggregate(avg=Avg('confidence'))['avg'] or 0
    recent_7d        = DetectionHistory.objects.filter(created_at__gte=last_7_days).count()

    daily_labels, daily_total, daily_fake, daily_real = [], [], [], []
    for i in range(13, -1, -1):
        day = now - timedelta(days=i)
        d_s = day.replace(hour=0,  minute=0,  second=0,  microsecond=0)
        d_e = day.replace(hour=23, minute=59, second=59, microsecond=999999)
        d_t = DetectionHistory.objects.filter(created_at__range=(d_s, d_e)).count()
        d_f = DetectionHistory.objects.filter(created_at__range=(d_s, d_e), is_deepfake=True).count()
        daily_labels.append(day.strftime('%b %d'))
        daily_total.append(d_t)
        daily_fake.append(d_f)
        daily_real.append(d_t - d_f)

    top_users = (DetectionHistory.objects
                 .values('user__username')
                 .annotate(count=Count('id'))
                 .order_by('-count')[:10])

    conf_buckets = {
        '50-60%':  DetectionHistory.objects.filter(confidence__gte=50,  confidence__lt=60).count(),
        '60-70%':  DetectionHistory.objects.filter(confidence__gte=60,  confidence__lt=70).count(),
        '70-80%':  DetectionHistory.objects.filter(confidence__gte=70,  confidence__lt=80).count(),
        '80-90%':  DetectionHistory.objects.filter(confidence__gte=80,  confidence__lt=90).count(),
        '90-100%': DetectionHistory.objects.filter(confidence__gte=90, confidence__lte=100).count(),
    }

    # FIX: Count is imported above — annotation works correctly now.
    # Using related_name='detections' from DetectionHistory.user FK.
    all_users = (User.objects
                 .annotate(detection_count=Count('detections'))
                 .order_by('-date_joined'))

    context = {
        'total_users':      total_users,
        'active_users':     active_users,
        'total_detections': total_detections,
        'fake_detections':  fake_detections,
        'real_detections':  real_detections,
        'image_detections': image_detections,
        'video_detections': video_detections,
        'avg_confidence':   round(avg_confidence, 1),
        'recent_7d':        recent_7d,
        'fake_pct': round(fake_detections / total_detections * 100
                          if total_detections else 0, 1),
        'real_pct': round(real_detections / total_detections * 100
                          if total_detections else 0, 1),
        'daily_labels_json':     _json.dumps(daily_labels),
        'daily_total_json':      _json.dumps(daily_total),
        'daily_fake_json':       _json.dumps(daily_fake),
        'daily_real_json':       _json.dumps(daily_real),
        'top_users_labels_json': _json.dumps([u['user__username'] or 'Anonymous'
                                              for u in top_users]),
        'top_users_counts_json': _json.dumps([u['count'] for u in top_users]),
        'conf_labels_json':      _json.dumps(list(conf_buckets.keys())),
        'conf_values_json':      _json.dumps(list(conf_buckets.values())),
        'recent_detections': (DetectionHistory.objects
                              .select_related('user')
                              .order_by('-created_at')[:20]),
        'all_users': all_users,
    }
    return render(request, 'detector/dashboard.html', context)


# =============================================================================
# API — UPLOAD
# =============================================================================

@csrf_exempt
def upload(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    try:
        if 'image' in request.FILES:
            field     = 'image'
            file_type = 'image'
        elif 'video' in request.FILES:
            field     = 'video'
            file_type = 'video'
        else:
            return JsonResponse({'success': False, 'error': 'No file provided'})

        if file_type == 'video' and not request.user.is_authenticated:
            return JsonResponse({
                'success':       False,
                'error':         'Login required for video uploads.',
                'auth_required': True,
                'redirect_url':  '/signup/?video=true',
            })

        uploaded = request.FILES[field]

        if file_type == 'image':
            allowed    = settings.ALLOWED_IMAGE_FORMATS
            max_size   = settings.MAX_FILE_SIZE
            upload_dir = 'uploads'
        else:
            allowed    = settings.VIDEO['ALLOWED_FORMATS']
            max_size   = settings.VIDEO['MAX_FILE_SIZE']
            upload_dir = 'video_uploads'

        if uploaded.content_type not in allowed:
            return JsonResponse({
                'success': False,
                'error':   f'Invalid file type: {uploaded.content_type}',
            })

        if uploaded.size > max_size:
            return JsonResponse({
                'success': False,
                'error':   f'File too large. Max {max_size // (1024*1024)} MB.',
            })

        fs       = FileSystemStorage(location=settings.MEDIA_ROOT / upload_dir)
        filename = fs.save(uploaded.name, uploaded)

        logger.info(f"Upload OK: {upload_dir}/{filename}  ({uploaded.size // 1024} KB)")
        return JsonResponse({'success': True, 'filename': filename, 'file_type': file_type})

    except Exception as exc:
        logger.error(f"Upload error: {exc}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(exc)})


# =============================================================================
# API — PREDICT
# =============================================================================

@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    try:
        data      = json.loads(request.body)
        filename  = data.get('filename')
        file_type = data.get('file_type', 'image')

        if not filename:
            return JsonResponse({'success': False, 'error': 'No filename provided'})

        # Always derive file type from extension — the extension is authoritative.
        # This prevents issues where the frontend sends the wrong type.
        file_type = _get_file_type(filename)

        if file_type == 'video' and not request.user.is_authenticated:
            return JsonResponse({
                'success':       False,
                'error':         'Login required for video analysis.',
                'auth_required': True,
                'redirect_url':  '/signup/?video=true',
            })

        upload_dir = 'uploads' if file_type == 'image' else 'video_uploads'
        file_path  = settings.MEDIA_ROOT / upload_dir / filename

        if not file_path.exists():
            return JsonResponse({'success': False, 'error': 'File not found'})

        t0       = time.time()
        detector = get_detector()

        if file_type == 'video':
            raw = detector.predict_video(str(file_path))
            logger.info(f"[predict] Routed to VIDEO model for: {filename}")
        else:
            raw = detector.predict(
                str(file_path),
                generate_heatmap=True,
                analyze_face=True,
                validate_quality=True,
            )
            logger.info(f"[predict] Routed to IMAGE model for: {filename}")

        elapsed = time.time() - t0

        if not raw.get('success', True):
            return JsonResponse({
                'success':        False,
                'error':          raw.get('error', 'Prediction failed'),
                'quality_issues': raw.get('quality_issues', []),
                'metrics':        raw.get('metrics', {}),
            })

        is_deepfake      = bool(raw.get('is_deepfake', False))
        confidence       = round(float(raw.get('confidence',       0.0)), 2)
        fake_probability = round(float(raw.get('fake_probability', 0.0)), 2)
        real_probability = round(float(raw.get('real_probability', 0.0)), 2)
        prediction       = 'FAKE' if is_deepfake else 'REAL'
        model_used       = raw.get('model_used', 'EfficientNet-B1')
        threshold_used   = round(float(raw.get('threshold', 0.0)), 2)

        resp = {
            'file_type':       file_type,
            'isDeepfake':      is_deepfake,
            'confidence':      confidence,
            'probabilities':   {'fake': fake_probability, 'real': real_probability},
            'threshold':       threshold_used,
            'rawConfidence':   round(float(raw.get('rawConfidence',   fake_probability)), 2),
            'fusedConfidence': round(float(raw.get('fusedConfidence', fake_probability)), 2),
            'decision':        raw.get('decision', {}),
            'modelUsed':       model_used,
            'processTime':     round(elapsed, 2),
            'indicators':      raw.get('indicators', []),
            'qualityMetrics':  raw.get('quality_metrics',  {}),
            'qualityWarnings': raw.get('quality_warnings', []),
            'trustScore':      round(float(raw.get('trust_score', 0.0)), 2),
            'uncertainty':     round(float(raw.get('uncertainty',  0.0)), 2),
            'forensicAnalysis': raw.get('forensic_analysis', {}),
        }

        if file_type == 'image':
            if raw.get('heatmap_path'):
                resp['heatmapPath'] = f"/media/{raw['heatmap_path']}"
            if raw.get('facial_analysis_path'):
                resp['facialAnalysisPath'] = f"/media/{raw['facial_analysis_path']}"
            if raw.get('facial_regions'):
                resp['facialRegions'] = raw['facial_regions']
            if raw.get('artifact_map_path'):
                resp['artifactMapPath'] = f"/media/{raw['artifact_map_path']}"
            if raw.get('fft_map_path'):
                resp['fftMapPath'] = f"/media/{raw['fft_map_path']}"
            if raw.get('noFaceDetected'):
                resp['noFaceDetected'] = True
        else:
            va = raw.get('videoAnalysis', {})
            resp['videoAnalysis'] = {
                'totalFrames':         va.get('totalFrames',         0),
                'deepfakeFrames':      va.get('deepfakeFrames',      0),
                'deepfakePercentage':  va.get('deepfakePercentage',  0),
                'stabilityScore':      va.get('stabilityScore',      0),
                'temporalConsistency': va.get('temporalConsistency', {}),
                'frameByFrame':        va.get('frameByFrame',        []),
                'videoInfo':           va.get('videoInfo',           {}),
            }

        _save_history(
            request, filename, file_type, prediction,
            confidence, fake_probability, real_probability,
            is_deepfake, elapsed, model_used, raw,
        )

        try:
            file_path.unlink()
        except Exception:
            pass

        logger.info(
            f"[predict] {file_type.upper()} → {prediction}  "
            f"conf={confidence:.1f}%  fake={fake_probability:.1f}%  "
            f"threshold={threshold_used:.1f}%  elapsed={elapsed:.2f}s"
        )

        return JsonResponse({'success': True, 'result': _to_python(resp)})

    except Exception as exc:
        logger.error(f"Predict error: {exc}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(exc)})


def _save_history(request, filename, file_type, prediction,
                  confidence, fake_prob, real_prob,
                  is_deepfake, elapsed, model_used, raw):
    """Save detection result to database (silently on failure)."""
    try:
        from .models import DetectionHistory
        va = raw.get('videoAnalysis', {}) or {}
        tc = va.get('temporalConsistency', {}) or {}
        DetectionHistory.objects.create(
            user                 = request.user if request.user.is_authenticated else None,
            filename             = filename,
            file_type            = file_type,
            prediction           = prediction,
            confidence           = confidence,
            fake_probability     = fake_prob,
            real_probability     = real_prob,
            is_deepfake          = is_deepfake,
            process_time         = elapsed,
            heatmap_path         = raw.get('heatmap_path', '')         or '',
            facial_analysis_path = raw.get('facial_analysis_path', '') or '',
            artifact_map_path    = raw.get('artifact_map_path', '')    or '',
            model_used           = model_used,
            total_frames         = int(va.get('totalFrames',    0)),
            deepfake_frames      = int(va.get('deepfakeFrames', 0)),
            temporal_variance    = float(tc.get('variance',     0.0)),
        )
    except Exception as exc:
        logger.error(f"History save failed: {exc}", exc_info=True)


# =============================================================================
# API — QUALITY CHECK
# =============================================================================

@csrf_exempt
def check_quality(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    try:
        data     = json.loads(request.body)
        filename = data.get('filename')
        if not filename:
            return JsonResponse({'success': False, 'error': 'No filename provided'})

        fp = settings.MEDIA_ROOT / 'uploads' / filename
        if not fp.exists():
            return JsonResponse({'success': False, 'error': 'File not found'})

        from .ml_utils import QualityValidator
        qr = QualityValidator.validate_image(fp)
        return JsonResponse({
            'success':  True,
            'is_valid': qr['is_valid'],
            'warnings': qr['warnings'],
            'errors':   qr['errors'],
            'metrics':  qr['metrics'],
        })

    except Exception as exc:
        logger.error(f"Quality check error: {exc}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(exc)})