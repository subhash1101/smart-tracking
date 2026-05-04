from django.contrib import admin
from .models import UserProfile, MoodEntry, PredictionResult


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'occupation', 'age', 'created_at']
    search_fields = ['user__username']


@admin.register(MoodEntry)
class MoodEntryAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'entry_date', 'mood_score', 'sleep_hours', 'work_hours',
        'self_stress_level', 'anxiety_level', 'energy_level',
        'depression_indicator_count', 'is_edited',
    ]
    list_filter  = ['entry_date', 'social_interaction', 'appetite_level',
                    'physical_activity', 'feeling_hopeless']
    search_fields = ['user__username']
    ordering     = ['-entry_date']
    readonly_fields = ['entry_date', 'created_at', 'updated_at']


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'stress_category', 'burnout_risk',
        'depression_risk', 'depression_score',
        'confidence_score', 'predicted_at',
    ]
    list_filter  = ['stress_category', 'burnout_risk', 'depression_risk', 'predicted_at']
    search_fields = ['user__username']
    readonly_fields = ['predicted_at', 'updated_at']
