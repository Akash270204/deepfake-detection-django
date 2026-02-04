from django.db import models
from django.utils import timezone


class DetectionHistory(models.Model):
    """Store image detection history"""
    
    # File information
    filename = models.CharField(max_length=255, default='unknown.jpg')
    
    # Detection results
    prediction = models.CharField(max_length=10, default='UNKNOWN')  # 'REAL' or 'FAKE'
    is_deepfake = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    real_probability = models.FloatField(default=0.0)
    fake_probability = models.FloatField(default=0.0)
    process_time = models.FloatField(default=0.0)
    
    # Visualization paths
    heatmap_path = models.CharField(max_length=500, blank=True, null=True, default='')
    facial_analysis_path = models.CharField(max_length=500, blank=True, null=True, default='')
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    model_used = models.CharField(max_length=50, default='EfficientNet-B0')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Detection Histories'
    
    def __str__(self):
        result = 'FAKE' if self.is_deepfake else 'REAL'
        return f"{result} - {self.confidence:.2f}% - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_result_display(self):
        """Human-readable result"""
        if self.is_deepfake:
            return f"🚨 DEEPFAKE ({self.confidence:.1f}%)"
        else:
            return f"✅ AUTHENTIC ({self.confidence:.1f}%)"