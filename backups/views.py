from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
import os
import json

from .models import DetectionHistory
from .ml_utils import detector

def index(request):
    """Home page - Image detection"""
    context = {
        'model_type': settings.MODEL_TYPE,
        'image_size': settings.IMAGE_SIZE,
        'max_file_size_mb': settings.MAX_FILE_SIZE / (1024 * 1024),
    }
    return render(request, 'detector/index.html', context)

def history(request):
    """View detection history"""
    history_entries = DetectionHistory.objects.all()[:50]
    
    context = {
        'history': history_entries,
        'total_detections': DetectionHistory.objects.count(),
        'total_deepfakes': DetectionHistory.objects.filter(is_deepfake=True).count(),
        'total_authentic': DetectionHistory.objects.filter(is_deepfake=False).count(),
    }
    return render(request, 'detector/history.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def upload_image(request):
    """Handle image upload"""
    try:
        if not request.FILES.get('image'):
            return JsonResponse({
                'success': False,
                'error': 'No image provided'
            }, status=400)
        
        image = request.FILES['image']
        
        # Validate file type
        if image.content_type not in settings.ALLOWED_IMAGE_FORMATS:
            return JsonResponse({
                'success': False,
                'error': 'Invalid file type. Only JPG and PNG allowed.'
            }, status=400)
        
        # Validate file size
        if image.size > settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            return JsonResponse({
                'success': False,
                'error': f'File too large. Maximum size is {max_mb:.0f}MB.'
            }, status=400)
        
        # Save file
        filename = default_storage.save(f'uploads/{image.name}', image)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        return JsonResponse({
            'success': True,
            'filename': filename,
            'path': file_path,
            'size': image.size,
            'type': image.content_type
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """Make prediction on uploaded image"""
    try:
        data = json.loads(request.body)
        filename = data.get('filename')
        
        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)
        
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        if not os.path.exists(file_path):
            return JsonResponse({
                'success': False,
                'error': 'Image file not found'
            }, status=404)
        
        # Make prediction
        print(f"\n{'='*70}")
        print(f"🔍 PREDICTION REQUEST")
        print(f"{'='*70}")
        print(f"File: {filename}")
        print(f"Path: {file_path}")
        
        result = detector.predict(file_path, generate_heatmap=True)
        
        print(f"\n📊 RESULT:")
        print(f"   Deepfake: {result['is_deepfake']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Real: {result['real_probability']:.2f}%")
        print(f"   Fake: {result['fake_probability']:.2f}%")
        print(f"{'='*70}\n")
        
        # Save to history
        history_entry = DetectionHistory.objects.create(
            image=filename,
            is_deepfake=result['is_deepfake'],
            confidence=result['confidence'],
            real_probability=result['real_probability'],
            fake_probability=result['fake_probability'],
            process_time=result['process_time'],
            heatmap_path=result.get('heatmap_path', ''),
            facial_analysis_path=result.get('facial_analysis_path', ''),
            model_used=settings.MODEL_TYPE
        )
        
        # Delete uploaded file after processing
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Cleaned up uploaded file: {filename}")
        except Exception as e:
            print(f"⚠️  Could not delete file: {e}")
        
        return JsonResponse({
            'success': True,
            'result': {
                'isDeepfake': result['is_deepfake'],
                'confidence': f"{result['confidence']:.2f}",
                'probabilities': {
                    'real': f"{result['real_probability']:.2f}",
                    'fake': f"{result['fake_probability']:.2f}"
                },
                'processTime': f"{result['process_time']:.2f}",
                'modelUsed': settings.MODEL_TYPE,
                'imageSize': settings.IMAGE_SIZE,
                'indicators': result['indicators'],
                'heatmapPath': result.get('heatmap_path', ''),
                'facialAnalysisPath': result.get('facial_analysis_path', ''),
                'historyId': history_entry.id
            }
        })
        
    except Exception as e:
        print(f"\n❌ PREDICTION ERROR:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def stats(request):
    """Get detection statistics"""
    try:
        total = DetectionHistory.objects.count()
        deepfakes = DetectionHistory.objects.filter(is_deepfake=True).count()
        authentic = DetectionHistory.objects.filter(is_deepfake=False).count()
        
        avg_confidence = 0
        if total > 0:
            from django.db.models import Avg
            avg_confidence = DetectionHistory.objects.aggregate(
                Avg('confidence')
            )['confidence__avg'] or 0
        
        return JsonResponse({
            'success': True,
            'stats': {
                'total': total,
                'deepfakes': deepfakes,
                'authentic': authentic,
                'avgConfidence': f"{avg_confidence:.2f}",
                'modelType': settings.MODEL_TYPE,
                'imageSize': settings.IMAGE_SIZE
            }
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)