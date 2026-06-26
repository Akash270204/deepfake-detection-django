from django.db import models
from django.contrib.auth.models import User


class DetectionHistory(models.Model):
    """Store deepfake detection results with user association"""

    # User association (nullable for backward compatibility)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='detections'
    )

    # File information
    filename  = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10, default='image')  # 'image' or 'video'

    # Detection results
    prediction       = models.CharField(max_length=10)   # 'REAL' or 'FAKE'
    confidence       = models.FloatField()
    fake_probability = models.FloatField()
    real_probability = models.FloatField()
    is_deepfake      = models.BooleanField(default=False)

    # Processing metadata
    process_time = models.FloatField(default=0.0)
    # model_used now tracks which of the two models was used:
    # 'EfficientNet-B1 (image)' or 'EfficientNet-B1 (video)'
    model_used   = models.CharField(max_length=50, default='EfficientNet-B1')

    # Visualization paths (images only)
    heatmap_path         = models.CharField(max_length=500, blank=True, null=True)
    facial_analysis_path = models.CharField(max_length=500, blank=True, null=True)
    artifact_map_path    = models.CharField(max_length=500, blank=True, null=True)

    # Video-specific fields
    total_frames      = models.IntegerField(default=0)
    deepfake_frames   = models.IntegerField(default=0)
    temporal_variance = models.FloatField(default=0.0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name        = 'Detection History'
        verbose_name_plural = 'Detection Histories'
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['file_type', '-created_at']),  # added for dashboard queries
        ]

    def __str__(self):
        user_str = f"{self.user.username}'s" if self.user else "Anonymous"
        return (
            f"{user_str} {self.file_type} detection "
            f"— {self.prediction} ({self.confidence:.1f}%)"
        )

    @property
    def is_video(self):
        return self.file_type == 'video'

    @property
    def is_image(self):
        return self.file_type == 'image'

    @property
    def detection_percentage(self):
        """Return deepfake frame percentage for videos."""
        if self.total_frames > 0:
            return round((self.deepfake_frames / self.total_frames) * 100, 1)
        return 0.0

    @property
    def model_label(self):
        """Human-readable model label for templates."""
        if 'video' in self.model_used.lower():
            return 'Video Model'
        return 'Image Model'