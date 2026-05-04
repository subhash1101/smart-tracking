"""Views for the public page, auth, mood entries, and reports."""

import json
from datetime import timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg
from django.utils import timezone

from .models import MoodEntry, PredictionResult, UserProfile
from .forms import UserRegistrationForm, LoginForm, MoodEntryForm
from .ml.predictor import predict_stress


def home_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'tracker/home.html')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Welcome, {user.first_name or user.username}! Your account is ready.")
            return redirect('dashboard')
        messages.error(request, "Please fix the errors below.")
    else:
        form = UserRegistrationForm()

    return render(request, 'tracker/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(
                request,
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password'],
            )
            if user:
                login(request, user)
                messages.success(request, f"Welcome back, {user.first_name or user.username}!")
                return redirect(request.GET.get('next', 'dashboard'))
            messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()

    return render(request, 'tracker/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')


@login_required
def mood_entry_view(request):
    """Create today's entry, or update it if the user already filled it in."""
    today = timezone.localdate()
    existing_entry = MoodEntry.objects.filter(user=request.user, entry_date=today).first()
    is_edit = existing_entry is not None

    if request.method == 'POST':
        if is_edit:
            # Update today's existing entry.
            form = MoodEntryForm(request.POST, instance=existing_entry)
        else:
            # Start a fresh entry for today.
            form = MoodEntryForm(request.POST)

        if form.is_valid():
            entry = form.save(commit=False)
            entry.user = request.user
            entry.save()

            # Re-run prediction from the saved answers.
            pred_kwargs = _entry_to_pred_kwargs(entry)
            result = predict_stress(**pred_kwargs)

            if is_edit:
                # Keep one prediction row per entry.
                pred_obj, _ = PredictionResult.objects.get_or_create(
                    mood_entry=entry,
                    defaults={'user': request.user}
                )
                pred_obj.user             = request.user
                pred_obj.stress_category  = result['stress_category']
                pred_obj.burnout_risk     = result['burnout_risk']
                pred_obj.depression_score = result['depression_score']
                pred_obj.depression_risk  = result['depression_risk']
                pred_obj.confidence_score = result['confidence']
                pred_obj.recommendations  = '|'.join(result['recommendations'])
                pred_obj.save()
                messages.success(request, "✅ Today's entry has been updated! Your new report is ready.")
            else:
                # Store the first prediction for this entry.
                pred_obj = PredictionResult.objects.create(
                    mood_entry=entry,
                    user=request.user,
                    stress_category  = result['stress_category'],
                    burnout_risk     = result['burnout_risk'],
                    depression_score = result['depression_score'],
                    depression_risk  = result['depression_risk'],
                    confidence_score = result['confidence'],
                    recommendations  = '|'.join(result['recommendations']),
                )
                messages.success(request, "✅ Today's entry saved! Your mental health report is ready.")

            return redirect('result', pk=pred_obj.pk)

        messages.error(request, "Please fix the errors highlighted below.")

    else:
        # Show the right form state on page load.
        if is_edit:
            form = MoodEntryForm(instance=existing_entry)
        else:
            form = MoodEntryForm()

    return render(request, 'tracker/mood_entry.html', {
        'form':          form,
        'today':         today,
        'is_edit':       is_edit,
        'existing_entry': existing_entry,
    })


def _entry_to_pred_kwargs(entry) -> dict:
    """Convert a saved entry into predictor keyword arguments."""
    return dict(
        mood               = entry.mood_score,
        sleep              = entry.sleep_hours,
        work               = entry.work_hours,
        self_stress        = entry.self_stress_level,
        anxiety            = entry.anxiety_level,
        energy             = entry.energy_level,
        social             = entry.social_interaction,
        appetite           = entry.appetite_level,
        concentration      = entry.concentration_level,
        motivation         = entry.motivation_level,
        screen_time        = entry.screen_time,
        physical_activity  = entry.physical_activity,
        feeling_hopeless   = entry.feeling_hopeless,
        loss_of_interest   = entry.loss_of_interest,
        feeling_tired      = entry.feeling_tired,
        trouble_sleeping   = entry.trouble_sleeping,
    )


@login_required
def result_view(request, pk):
    """Show the prediction result for one saved entry."""
    prediction = get_object_or_404(PredictionResult, pk=pk, user=request.user)
    return render(request, 'tracker/result.html', {
        'prediction':    prediction,
        'entry':         prediction.mood_entry,
        'recommendations': prediction.get_recommendations_list(),
    })


@login_required
def dashboard_view(request):
    today = timezone.localdate()
    user  = request.user

    today_entry = (
        MoodEntry.objects
        .filter(user=user, entry_date=today)
        .select_related('prediction')
        .first()
    )

    # Build the week-long line chart data.
    last_7 = []
    for i in range(6, -1, -1):
        day   = today - timedelta(days=i)
        entry = MoodEntry.objects.filter(user=user, entry_date=day).first()
        last_7.append({
            'date':    day.strftime('%b %d'),
            'mood':    entry.mood_score       if entry else None,
            'stress':  entry.self_stress_level if entry else None,
            'sleep':   float(entry.sleep_hours) if entry else None,
            'anxiety': entry.anxiety_level    if entry else None,
            'energy':  entry.energy_level     if entry else None,
        })

    chart_labels  = json.dumps([d['date']    for d in last_7])
    mood_data     = json.dumps([d['mood']    for d in last_7])
    stress_data   = json.dumps([d['stress']  for d in last_7])
    sleep_data    = json.dumps([d['sleep']   for d in last_7])
    anxiety_data  = json.dumps([d['anxiety'] for d in last_7])
    energy_data   = json.dumps([d['energy']  for d in last_7])

    # Count stress levels for the last 30 days.
    last_30 = PredictionResult.objects.filter(
        user=user, predicted_at__date__gte=today - timedelta(days=30)
    )
    stress_counts = {
        'Low':    last_30.filter(stress_category='Low').count(),
        'Medium': last_30.filter(stress_category='Medium').count(),
        'High':   last_30.filter(stress_category='High').count(),
    }
    stress_pie_data = json.dumps(list(stress_counts.values()))

    # Count depression risk levels for the last 30 days.
    dep_counts = {
        'Low':    last_30.filter(depression_risk='Low').count(),
        'Medium': last_30.filter(depression_risk='Medium').count(),
        'High':   last_30.filter(depression_risk='High').count(),
    }
    dep_pie_data = json.dumps(list(dep_counts.values()))

    recent_entries = MoodEntry.objects.filter(user=user).select_related('prediction')[:5]

    avg_data = MoodEntry.objects.filter(
        user=user, entry_date__gte=today - timedelta(days=7)
    ).aggregate(
        avg_mood=Avg('mood_score'),
        avg_sleep=Avg('sleep_hours'),
        avg_work=Avg('work_hours'),
        avg_anxiety=Avg('anxiety_level'),
        avg_energy=Avg('energy_level'),
    )

    return render(request, 'tracker/dashboard.html', {
        'today_entry':    today_entry,
        'chart_labels':   chart_labels,
        'mood_data':      mood_data,
        'stress_data':    stress_data,
        'sleep_data':     sleep_data,
        'anxiety_data':   anxiety_data,
        'energy_data':    energy_data,
        'stress_pie_data': stress_pie_data,
        'dep_pie_data':   dep_pie_data,
        'stress_counts':  stress_counts,
        'dep_counts':     dep_counts,
        'recent_entries': recent_entries,
        'avg_mood':    round(avg_data['avg_mood']    or 0, 1),
        'avg_sleep':   round(avg_data['avg_sleep']   or 0, 1),
        'avg_work':    round(avg_data['avg_work']    or 0, 1),
        'avg_anxiety': round(avg_data['avg_anxiety'] or 0, 1),
        'avg_energy':  round(avg_data['avg_energy']  or 0, 1),
        'today': today,
    })


@login_required
def history_view(request):
    entries = MoodEntry.objects.filter(user=request.user).select_related('prediction')
    return render(request, 'tracker/history.html', {'entries': entries})


@login_required
def weekly_summary_view(request):
    today      = timezone.localdate()
    week_start = today - timedelta(days=today.weekday())
    week_end   = week_start + timedelta(days=6)

    entries = MoodEntry.objects.filter(
        user=request.user,
        entry_date__range=[week_start, week_end]
    ).select_related('prediction')

    if not entries.exists():
        messages.info(request, "No entries found for this week yet.")

    avg_data = entries.aggregate(
        avg_mood=Avg('mood_score'),
        avg_sleep=Avg('sleep_hours'),
        avg_work=Avg('work_hours'),
        avg_stress=Avg('self_stress_level'),
        avg_anxiety=Avg('anxiety_level'),
        avg_energy=Avg('energy_level'),
    )

    predictions = PredictionResult.objects.filter(
        user=request.user,
        predicted_at__date__range=[week_start, week_end]
    )

    high_stress_days  = predictions.filter(stress_category='High').count()
    burnout_days      = predictions.filter(burnout_risk='High').count()
    high_dep_days     = predictions.filter(depression_risk='High').count()

    avg_mood = avg_data['avg_mood'] or 0
    if avg_mood >= 7 and high_stress_days == 0:
        week_status, status_color = 'Excellent', 'success'
    elif avg_mood >= 5 and high_stress_days <= 2:
        week_status, status_color = 'Moderate', 'warning'
    else:
        week_status, status_color = 'Needs Attention', 'danger'

    return render(request, 'tracker/weekly_summary.html', {
        'entries':          entries,
        'week_start':       week_start,
        'week_end':         week_end,
        'avg_data':         avg_data,
        'high_stress_days': high_stress_days,
        'burnout_days':     burnout_days,
        'high_dep_days':    high_dep_days,
        'week_status':      week_status,
        'status_color':     status_color,
        'entry_count':      entries.count(),
    })
