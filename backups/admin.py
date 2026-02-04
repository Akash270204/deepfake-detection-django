from django.contrib import admin
from .models import DetectionHistory

@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    """Admin interface for detection history"""
    
    list_display = [
        'id',
        'get_result_display',
        'confidence',
        'model_used',
        'created_at'
    ]
    
    list_filter = [
        'is_deepfake',
        'model_used',
        'created_at'
    ]
    
    search_fields = [
        'image',
    ]
    
    readonly_fields = [
        'image',
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
    
    list_per_page = 25
    
    fieldsets = (
        ('Detection Result', {
            'fields': (
                'is_deepfake',
                'confidence',
                'real_probability',
                'fake_probability',
            )
        }),
        ('Image', {
            'fields': (
                'image',
            )
        }),
        ('Analysis', {
            'fields': (
                'heatmap_path',
                'facial_analysis_path',
            )
        }),
        ('Metadata', {
            'fields': (
                'model_used',
                'process_time',
                'created_at',
            )
        }),
    )
    
    def has_add_permission(self, request):
        """Disable manual addition"""
        return False
    
    def has_change_permission(self, request, obj=None):
        """Disable editing"""
        return False