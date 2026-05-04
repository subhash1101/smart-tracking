"""
forms.py — Enhanced MoodEntryForm with 15 questions in 3 sections.
Supports both CREATE (new entry) and UPDATE (edit today's entry).
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import MoodEntry, UserProfile


# ─────────────────────────────────────────────────────────────
# AUTH FORMS (unchanged)
# ─────────────────────────────────────────────────────────────

class UserRegistrationForm(UserCreationForm):
    email      = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=False)
    last_name  = forms.CharField(max_length=30, required=False)

    class Meta:
        model  = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            UserProfile.objects.get_or_create(user=user)
        return user


class LoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )


# ─────────────────────────────────────────────────────────────
# SLIDER WIDGET HELPER
# ─────────────────────────────────────────────────────────────

def slider(id_name, min_val=1, max_val=10):
    """Returns a range-input widget with Bootstrap + custom id."""
    return forms.NumberInput(attrs={
        'type':  'range',
        'class': 'form-range',
        'min':   min_val,
        'max':   max_val,
        'step':  1,
        'id':    id_name,
    })


# ─────────────────────────────────────────────────────────────
# MAIN MOOD ENTRY FORM — 15 FIELDS
# ─────────────────────────────────────────────────────────────

class MoodEntryForm(forms.ModelForm):
    """
    Comprehensive daily mental health form.
    Works for both CREATE and UPDATE (pass instance= for edit mode).

    Sections:
        Section 1 — Basic Health       (4 fields)
        Section 2 — Mental State       (8 fields)
        Section 3 — Depression Signals (4 Yes/No fields)
    """

    class Meta:
        model  = MoodEntry
        fields = [
            # Section 1
            'mood_score', 'sleep_hours', 'work_hours', 'self_stress_level',
            # Section 2
            'anxiety_level', 'energy_level', 'social_interaction',
            'appetite_level', 'concentration_level', 'motivation_level',
            'screen_time', 'physical_activity',
            # Section 3
            'feeling_hopeless', 'loss_of_interest', 'feeling_tired', 'trouble_sleeping',
            # Extra
            'notes',
        ]

        widgets = {
            # ── Section 1 ────────────────────────────────────────────────
            'mood_score': slider('mood_slider'),
            'sleep_hours': forms.NumberInput(attrs={
                'class': 'form-control', 'min': 0, 'max': 24,
                'step': 0.5, 'placeholder': 'e.g. 7.5',
            }),
            'work_hours': forms.NumberInput(attrs={
                'class': 'form-control', 'min': 0, 'max': 24,
                'step': 0.5, 'placeholder': 'e.g. 8',
            }),
            'self_stress_level': slider('stress_slider'),

            # ── Section 2 ────────────────────────────────────────────────
            'anxiety_level':      slider('anxiety_slider'),
            'energy_level':       slider('energy_slider'),
            'social_interaction': forms.Select(attrs={'class': 'form-select'}),
            'appetite_level':     forms.Select(attrs={'class': 'form-select'}),
            'concentration_level': slider('concentration_slider'),
            'motivation_level':    slider('motivation_slider'),
            'screen_time': forms.NumberInput(attrs={
                'class': 'form-control', 'min': 0, 'max': 24,
                'step': 0.5, 'placeholder': 'e.g. 3',
            }),
            'physical_activity': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'physical_activity'}),

            # ── Section 3 ────────────────────────────────────────────────
            'feeling_hopeless':  forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'feeling_hopeless'}),
            'loss_of_interest':  forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'loss_of_interest'}),
            'feeling_tired':     forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'feeling_tired'}),
            'trouble_sleeping':  forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'trouble_sleeping'}),

            # ── Notes ────────────────────────────────────────────────────
            'notes': forms.Textarea(attrs={
                'class': 'form-control', 'rows': 3,
                'placeholder': 'Any additional thoughts about today… (optional)',
            }),
        }

        labels = {
            # Section 1
            'mood_score':       '😊 Overall Mood (1 = very bad, 10 = excellent)',
            'sleep_hours':      '😴 Sleep Hours Last Night',
            'work_hours':       '💼 Work / Study Hours Today',
            'self_stress_level':'🌡️ Self-Reported Stress (1 = calm, 10 = extreme)',
            # Section 2
            'anxiety_level':        '😰 Anxiety Level (1 = none, 10 = severe)',
            'energy_level':         '⚡ Energy Level (1 = exhausted, 10 = very energetic)',
            'social_interaction':   '🤝 Social Interaction Level Today',
            'appetite_level':       '🍽️ Appetite Level Today',
            'concentration_level':  '🎯 Concentration Level (1 = very poor, 10 = excellent)',
            'motivation_level':     '🚀 Motivation Level (1 = none, 10 = highly motivated)',
            'screen_time':          '📱 Recreational Screen Time (hours)',
            'physical_activity':    '🏃 Did you exercise or do physical activity today?',
            # Section 3
            'feeling_hopeless': '😔 I felt hopeless or worthless today',
            'loss_of_interest': '💔 I lost interest in activities I usually enjoy',
            'feeling_tired':    '😓 I felt tired or fatigued for most of the day',
            'trouble_sleeping': '🌙 I had trouble falling or staying asleep',
            # Notes
            'notes': '📝 Notes (optional)',
        }

    # ── Validation ───────────────────────────────────────────────────────

    def clean_mood_score(self):
        v = self.cleaned_data.get('mood_score')
        if v is not None and not (1 <= v <= 10):
            raise forms.ValidationError("Must be between 1 and 10.")
        return v

    def clean_sleep_hours(self):
        v = self.cleaned_data.get('sleep_hours')
        if v is not None and not (0 <= v <= 24):
            raise forms.ValidationError("Must be between 0 and 24.")
        return v

    def clean_work_hours(self):
        v = self.cleaned_data.get('work_hours')
        if v is not None and not (0 <= v <= 24):
            raise forms.ValidationError("Must be between 0 and 24.")
        return v

    def clean_anxiety_level(self):
        v = self.cleaned_data.get('anxiety_level')
        if v is not None and not (1 <= v <= 10):
            raise forms.ValidationError("Must be between 1 and 10.")
        return v

    def clean_energy_level(self):
        v = self.cleaned_data.get('energy_level')
        if v is not None and not (1 <= v <= 10):
            raise forms.ValidationError("Must be between 1 and 10.")
        return v

    def clean_screen_time(self):
        v = self.cleaned_data.get('screen_time')
        if v is not None and not (0 <= v <= 24):
            raise forms.ValidationError("Screen time must be between 0 and 24 hours.")
        return v

    def clean(self):
        cleaned = super().clean()
        sleep  = cleaned.get('sleep_hours') or 0
        work   = cleaned.get('work_hours') or 0
        screen = cleaned.get('screen_time') or 0

        if sleep + work > 24:
            raise forms.ValidationError(
                "Sleep hours + work/study hours cannot exceed 24 hours in a day."
            )
        if sleep + work + screen > 24:
            raise forms.ValidationError(
                "Sleep + work + screen time cannot exceed 24 hours in a day."
            )
        return cleaned
