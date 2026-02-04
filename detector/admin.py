from django.contrib import admin
from .models import DetectionHistory


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    """Admin interface for DetectionHistory"""
    
    list_display = [
        'id',
        'filename',
        'prediction',
        'confidence',
        'is_deepfake',
        'model_used',
        'created_at'
    ]
    
    list_filter = [
        'is_deepfake',
        'prediction',
        'model_used',
        'created_at'
    ]
    
    search_fields = [
        'filename',
        'prediction'
    ]
    
    readonly_fields = [
        'filename',  # ✅ FIXED: Changed from 'image' to 'filename'
        'prediction',
        'is_deepfake',
        'confidence',
        'real_probability',
        'fake_probability',
        'process_time',
        'heatmap_path',
        'facial_analysis_path',
        'model_used',
        'created_at'
    ]
    
    ordering = ['-created_at']
    
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        """Disable manual creation of history records"""
        return False
    
    def has_change_permission(self, request, obj=None):
        """Make records read-only"""
        return False