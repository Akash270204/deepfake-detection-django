import os
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from pathlib import Path
import json
import time

from .ml_utils import get_detector

logger = logging.getLogger(__name__)


def index(request):
    """Main page"""
    return render(request, 'detector/index.html')
    

def history(request):
    """Display detection history"""
    try:
        from .models import DetectionHistory
        history_items = DetectionHistory.objects.all().order_by('-created_at')[:20]
        return render(request, 'detector/history.html', {
            'history_items': history_items
        })
    except Exception as e:
        logger.error(f"History error: {e}")
        return render(request, 'detector/history.html', {
            'history_items': []
        })


@csrf_exempt
def upload(request):
    """Handle file upload"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST allowed'})
    
    try:
        logger.info("\n" + "="*70)
        logger.info("📤 UPLOAD REQUEST")
        logger.info("="*70)
        
        # Get uploaded file
        if 'image' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No image provided'})
        
        uploaded_file = request.FILES['image']
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({
                'success': False,
                'error': 'Invalid file type. Only JPG and PNG allowed.'
            })
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if uploaded_file.size > max_size:
            return JsonResponse({
                'success': False,
                'error': 'File too large. Maximum size is 10MB.'
            })
        
        logger.info(f"📁 File: {uploaded_file.name}")
        logger.info(f"📊 Size: {uploaded_file.size / 1024:.2f} KB")
        logger.info(f"🎨 Type: {uploaded_file.content_type}")
        
        # Save file
        fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'uploads')
        filename = fs.save(uploaded_file.name, uploaded_file)
        
        logger.info(f"✅ File saved: uploads/{filename}")
        logger.info("="*70 + "\n")
        
        return JsonResponse({
            'success': True,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
def predict(request):
    """Handle prediction request"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST allowed'})
    
    try:
        logger.info("\n" + "="*70)
        logger.info("🔍 PREDICTION REQUEST")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Get filename from request
        data = json.loads(request.body)
        filename = data.get('filename')
        
        if not filename:
            return JsonResponse({'success': False, 'error': 'No filename provided'})
        
        # Build file path
        file_path = settings.MEDIA_ROOT / 'uploads' / filename
        
        logger.info(f"📁 File: uploads/{filename}")
        logger.info(f"📂 Path: {file_path}")
        
        if not file_path.exists():
            return JsonResponse({'success': False, 'error': 'File not found'})
        
        # Run prediction
        logger.info("🤖 Running prediction...")
        detector = get_detector()
        result = detector.predict(str(file_path), generate_heatmap=True, analyze_face=True)
        
        process_time = time.time() - start_time
        
        # Format result for frontend
        response_data = {
            'isDeepfake': result['is_deepfake'],
            'confidence': result['confidence'],
            'probabilities': {
                'fake': result['fake_probability'],
                'real': result['real_probability']
            },
            'threshold': result['threshold'],
            'modelUsed': result['model_used'],
            'processTime': round(process_time, 2),
            'heatmapPath': None,
            'facialAnalysisPath': None,
            'indicators': result.get('indicators', [])
        }
        
        # Add heatmap path if available
        if result.get('heatmap_path'):
            response_data['heatmapPath'] = f"/media/{result['heatmap_path']}"
        
        # Add facial analysis if available
        if result.get('facial_analysis_path'):
            response_data['facialAnalysisPath'] = f"/media/{result['facial_analysis_path']}"
        
        if result.get('facial_regions'):
            response_data['facialRegions'] = result['facial_regions']
        
        # Log summary
        logger.info(f"\n   Result: {'🚨 FAKE' if result['is_deepfake'] else '✅ REAL'}")
        logger.info(f"   Confidence: {result['confidence']:.2f}%")
        logger.info(f"   Real: {result['real_probability']:.2f}% | Fake: {result['fake_probability']:.2f}%")
        logger.info(f"   Process time: {process_time:.2f}s")
        
        if result.get('heatmap_path'):
            logger.info(f"   ✅ Heatmap: {result['heatmap_path']}")
        
        if result.get('facial_analysis_path'):
            logger.info(f"   ✅ Facial analysis: {result['facial_analysis_path']}")
        
        # Try to save to history
        try:
            from .models import DetectionHistory
            history_entry = DetectionHistory.objects.create(
                filename=filename,
                prediction=result['prediction'],
                confidence=result['confidence'],
                fake_probability=result['fake_probability'],
                real_probability=result['real_probability'],
                is_deepfake=result['is_deepfake'],
                process_time=process_time,
                heatmap_path=result.get('heatmap_path', ''),
                facial_analysis_path=result.get('facial_analysis_path', ''),
                model_used=result['model_used']
            )
            logger.info(f"   ✅ History saved: {history_entry.id}")
        except Exception as history_error:
            logger.warning(f"   ⚠️  History save failed: {history_error}")
        
        # Clean up uploaded file
        try:
            file_path.unlink()
            logger.info(f"🗑️  Cleaned up: uploads/{filename}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️  Cleanup failed: {cleanup_error}")
        
        logger.info("="*70 + "\n")
        
        return JsonResponse({
            'success': True,
            'result': response_data
        })
        
    except Exception as e:
        logger.error(f"❌ PREDICTION ERROR: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        })
