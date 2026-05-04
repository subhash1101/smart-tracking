"""Database models for user profiles, daily check-ins, and predictions."""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


class UserProfile(models.Model):
    """Extended profile for each registered user."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.PositiveIntegerField(null=True, blank=True)
    occupation = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Profile of {self.user.username}"


class MoodEntry(models.Model):
    """One daily mental health check-in for a user."""

    # Choices used by the form and admin.
    SOCIAL_CHOICES = [
        ('Low',    'Low — I mostly stayed alone'),
        ('Medium', 'Medium — Some interaction'),
        ('High',   'High — Very social today'),
    ]

    APPETITE_CHOICES = [
        ('Low',    'Low — I barely ate'),
        ('Normal', 'Normal — Ate as usual'),
        ('High',   'High — Overate / stress eating'),
    ]

    # Owner of the entry.
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mood_entries')

    # Basic health fields.
    mood_score = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        help_text="Overall mood today (1 = very bad, 10 = excellent)"
    )
    sleep_hours = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(24)],
        help_text="Hours of sleep last night"
    )
    work_hours = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(24)],
        help_text="Hours spent on work or study today"
    )
    self_stress_level = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        help_text="Self-reported stress level (1 = calm, 10 = extreme)"
    )

    # Mental state fields.
    anxiety_level = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        default=5,
        help_text="How anxious did you feel today? (1 = not at all, 10 = severely)"
    )
    energy_level = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        default=5,
        help_text="Overall energy level today (1 = exhausted, 10 = very energetic)"
    )
    social_interaction = models.CharField(
        max_length=10,
        choices=SOCIAL_CHOICES,
        default='Medium',
        help_text="Level of social interaction today"
    )
    appetite_level = models.CharField(
        max_length=10,
        choices=APPETITE_CHOICES,
        default='Normal',
        help_text="Appetite level today"
    )
    concentration_level = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        default=5,
        help_text="Ability to concentrate today (1 = very poor, 10 = excellent)"
    )
    motivation_level = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        default=5,
        help_text="Motivation to do tasks today (1 = none, 10 = highly motivated)"
    )
    screen_time = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(24)],
        default=0.0,
        help_text="Total recreational screen time in hours (phone, TV, social media)"
    )
    physical_activity = models.BooleanField(
        default=False,
        help_text="Did you do any physical exercise or activity today?"
    )

    # Depression signal fields.
    feeling_hopeless = models.BooleanField(
        default=False,
        help_text="Did you feel hopeless or worthless today?"
    )
    loss_of_interest = models.BooleanField(
        default=False,
        help_text="Did you lose interest in activities you usually enjoy?"
    )
    feeling_tired = models.BooleanField(
        default=False,
        help_text="Did you feel tired or fatigued for most of the day?"
    )
    trouble_sleeping = models.BooleanField(
        default=False,
        help_text="Did you have trouble falling or staying asleep?"
    )

    # Optional notes from the user.
    notes = models.TextField(blank=True, help_text="Any additional thoughts about today")

    # Dates used for history, edits, and one-entry-per-day checks.
    entry_date = models.DateField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-entry_date']
        unique_together = ['user', 'entry_date']

    def __str__(self):
        return f"{self.user.username} | {self.entry_date} | Mood:{self.mood_score}"

    # Small helpers used by templates and admin.
    @property
    def depression_indicator_count(self):
        """Count how many depression signals were checked."""
        return sum([
            self.feeling_hopeless,
            self.loss_of_interest,
            self.feeling_tired,
            self.trouble_sleeping,
        ])

    @property
    def is_edited(self):
        """Return True when the entry was edited after creation."""
        if self.created_at and self.updated_at:
            diff = (self.updated_at - self.created_at).total_seconds()
            return diff > 5
        return False


class PredictionResult(models.Model):
    """Prediction output linked to a daily mood entry."""

    LEVEL_CHOICES = [
        ('Low',    'Low'),
        ('Medium', 'Medium'),
        ('High',   'High'),
    ]

    mood_entry = models.OneToOneField(
        MoodEntry, on_delete=models.CASCADE, related_name='prediction'
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')

    # Values shown in reports and charts.
    stress_category  = models.CharField(max_length=10, choices=LEVEL_CHOICES)
    burnout_risk     = models.CharField(max_length=10, choices=LEVEL_CHOICES)
    depression_score = models.IntegerField(default=0, help_text="0–4 depression indicator count")
    depression_risk  = models.CharField(max_length=10, choices=LEVEL_CHOICES, default='Low')
    confidence_score = models.FloatField(default=0.0)

    # Recommendations are stored as a small pipe-separated list.
    recommendations = models.TextField(blank=True)

    predicted_at = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-predicted_at']

    def get_recommendations_list(self):
        if self.recommendations:
            return [r.strip() for r in self.recommendations.split('|') if r.strip()]
        return []

    def __str__(self):
        return (
            f"{self.user.username} | Stress:{self.stress_category} | "
            f"Burnout:{self.burnout_risk} | {self.predicted_at.date()}"
        )
