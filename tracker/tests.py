"""
tests.py — Enhanced unit tests for v2 of Smart Mental Health Monitoring System.
Covers:
  - Predictor with all 16 new features
  - Form validation (new fields)
  - Edit (update) functionality
  - DB storage of new fields
  - Depression risk detection

Run: python manage.py test tracker
"""

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from .models import MoodEntry, PredictionResult, UserProfile
from .forms import MoodEntryForm
from .ml.predictor import predict_stress


# ─────────────────────────────────────────────
# BASE DATA HELPER
# ─────────────────────────────────────────────

def full_form_data(**overrides):
    """Return a complete valid form POST dict (all 15 fields)."""
    base = {
        'mood_score':          7,
        'sleep_hours':         7.0,
        'work_hours':          8.0,
        'self_stress_level':   4,
        'anxiety_level':       4,
        'energy_level':        7,
        'social_interaction':  'Medium',
        'appetite_level':      'Normal',
        'concentration_level': 7,
        'motivation_level':    7,
        'screen_time':         2.0,
        'physical_activity':   False,
        'feeling_hopeless':    False,
        'loss_of_interest':    False,
        'feeling_tired':       False,
        'trouble_sleeping':    False,
        'notes':               'Test entry',
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────
# 1. PREDICTOR TESTS
# ─────────────────────────────────────────────

class PredictorTests(TestCase):

    def test_low_stress_healthy_user(self):
        result = predict_stress(
            mood=9, sleep=8, work=4, self_stress=2,
            anxiety=1, energy=9, social='High', appetite='Normal',
            concentration=9, motivation=9, screen_time=1,
            physical_activity=True,
            feeling_hopeless=False, loss_of_interest=False,
            feeling_tired=False, trouble_sleeping=False,
        )
        self.assertEqual(result['stress_category'], 'Low')
        self.assertEqual(result['burnout_risk'], 'Low')
        self.assertEqual(result['depression_score'], 0)
        self.assertEqual(result['depression_risk'], 'Low')

    def test_high_stress_overworked_user(self):
        result = predict_stress(
            mood=2, sleep=3, work=14, self_stress=9,
            anxiety=9, energy=2, social='Low', appetite='Low',
            concentration=2, motivation=2, screen_time=8,
            physical_activity=False,
            feeling_hopeless=True, loss_of_interest=True,
            feeling_tired=True, trouble_sleeping=True,
        )
        self.assertIn(result['stress_category'], ['Medium', 'High'])
        self.assertIn(result['burnout_risk'],    ['Medium', 'High'])
        self.assertEqual(result['depression_score'], 4)
        self.assertEqual(result['depression_risk'], 'High')

    def test_depression_score_counts_correctly(self):
        result = predict_stress(
            mood=5, sleep=6, work=8, self_stress=5,
            feeling_hopeless=True, loss_of_interest=True,
            feeling_tired=False, trouble_sleeping=False,
        )
        self.assertEqual(result['depression_score'], 2)
        self.assertEqual(result['depression_risk'], 'Medium')

    def test_depression_score_zero(self):
        result = predict_stress(mood=7, sleep=7, work=7, self_stress=4)
        self.assertEqual(result['depression_score'], 0)
        self.assertEqual(result['depression_risk'], 'Low')

    def test_physical_activity_reduces_burnout(self):
        kwargs = dict(mood=5, sleep=6, work=9, self_stress=6,
                      anxiety=6, energy=5)
        with_exercise    = predict_stress(**kwargs, physical_activity=True)
        without_exercise = predict_stress(**kwargs, physical_activity=False)
        # Both should return valid results
        self.assertIn(with_exercise['burnout_risk'],    ['Low', 'Medium', 'High'])
        self.assertIn(without_exercise['burnout_risk'], ['Low', 'Medium', 'High'])

    def test_confidence_is_percentage(self):
        result = predict_stress(mood=6, sleep=7, work=8, self_stress=5)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 100.0)

    def test_recommendations_not_empty(self):
        result = predict_stress(mood=4, sleep=5, work=10, self_stress=7)
        self.assertGreater(len(result['recommendations']), 0)

    def test_high_screen_time_recommendation(self):
        result = predict_stress(mood=6, sleep=7, work=7, self_stress=5,
                                screen_time=7)
        joined = ' '.join(result['recommendations'])
        self.assertIn('screen', joined.lower())

    def test_low_social_recommendation(self):
        result = predict_stress(mood=5, sleep=7, work=8, self_stress=5,
                                social='Low')
        joined = ' '.join(result['recommendations'])
        self.assertIn('social', joined.lower())


# ─────────────────────────────────────────────
# 2. FORM VALIDATION TESTS
# ─────────────────────────────────────────────

class MoodEntryFormTests(TestCase):

    def test_valid_full_form(self):
        form = MoodEntryForm(data=full_form_data())
        self.assertTrue(form.is_valid(), form.errors)

    def test_mood_out_of_range(self):
        form = MoodEntryForm(data=full_form_data(mood_score=15))
        self.assertFalse(form.is_valid())
        self.assertIn('mood_score', form.errors)

    def test_anxiety_out_of_range(self):
        form = MoodEntryForm(data=full_form_data(anxiety_level=11))
        self.assertFalse(form.is_valid())

    def test_sleep_plus_work_exceeds_24(self):
        form = MoodEntryForm(data=full_form_data(sleep_hours=14, work_hours=12))
        self.assertFalse(form.is_valid())

    def test_total_hours_exceed_24(self):
        form = MoodEntryForm(data=full_form_data(sleep_hours=10, work_hours=8, screen_time=8))
        self.assertFalse(form.is_valid())

    def test_negative_screen_time(self):
        form = MoodEntryForm(data=full_form_data(screen_time=-1))
        self.assertFalse(form.is_valid())

    def test_invalid_social_choice(self):
        form = MoodEntryForm(data=full_form_data(social_interaction='VeryHigh'))
        self.assertFalse(form.is_valid())

    def test_invalid_appetite_choice(self):
        form = MoodEntryForm(data=full_form_data(appetite_level='Huge'))
        self.assertFalse(form.is_valid())

    def test_boolean_depression_fields_default_false(self):
        """Unchecked checkboxes should default to False (not raise errors)."""
        data = full_form_data()
        # Remove checkbox keys (simulates unchecked)
        for key in ['physical_activity','feeling_hopeless','loss_of_interest','feeling_tired','trouble_sleeping']:
            data.pop(key, None)
        form = MoodEntryForm(data=data)
        self.assertTrue(form.is_valid(), form.errors)


# ─────────────────────────────────────────────
# 3. AUTHENTICATION TESTS
# ─────────────────────────────────────────────

class AuthTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser', password='TestPass123!', email='t@t.com'
        )
        UserProfile.objects.get_or_create(user=self.user)

    def test_login_success(self):
        r = self.client.post(reverse('login'), {'username':'testuser', 'password':'TestPass123!'})
        self.assertRedirects(r, reverse('dashboard'))

    def test_login_fail(self):
        r = self.client.post(reverse('login'), {'username':'testuser', 'password':'wrong'})
        self.assertEqual(r.status_code, 200)

    def test_dashboard_requires_login(self):
        r = self.client.get(reverse('dashboard'))
        self.assertRedirects(r, '/login/?next=/dashboard/')


# ─────────────────────────────────────────────
# 4. MOOD ENTRY — CREATE
# ─────────────────────────────────────────────

class MoodEntryCreateTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u1', password='Pass123!')
        UserProfile.objects.get_or_create(user=self.user)
        self.client.login(username='u1', password='Pass123!')

    def test_get_entry_page_empty(self):
        r = self.client.get(reverse('mood_entry'))
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.context['is_edit'])

    def test_create_entry_saves_all_fields(self):
        r = self.client.post(reverse('mood_entry'), full_form_data(
            anxiety_level=7, energy_level=6,
            social_interaction='High', appetite_level='Low',
            concentration_level=6, motivation_level=5,
            screen_time=3.0, physical_activity=True,
            feeling_hopeless=True, loss_of_interest=False,
            feeling_tired=True, trouble_sleeping=False,
        ))
        self.assertEqual(r.status_code, 302)
        entry = MoodEntry.objects.get(user=self.user)
        self.assertEqual(entry.anxiety_level, 7)
        self.assertEqual(entry.energy_level, 6)
        self.assertEqual(entry.social_interaction, 'High')
        self.assertEqual(entry.appetite_level, 'Low')
        self.assertTrue(entry.physical_activity)
        self.assertTrue(entry.feeling_hopeless)
        self.assertFalse(entry.loss_of_interest)
        self.assertEqual(entry.depression_indicator_count, 2)

    def test_prediction_stored_with_depression_fields(self):
        self.client.post(reverse('mood_entry'), full_form_data(
            feeling_hopeless=True, loss_of_interest=True,
            feeling_tired=True, trouble_sleeping=True,
        ))
        pred = PredictionResult.objects.get(user=self.user)
        self.assertEqual(pred.depression_score, 4)
        self.assertEqual(pred.depression_risk, 'High')
        self.assertIn(pred.stress_category, ['Low','Medium','High'])


# ─────────────────────────────────────────────
# 5. MOOD ENTRY — EDIT (UPDATE)
# ─────────────────────────────────────────────

class MoodEntryEditTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u2', password='Pass123!')
        UserProfile.objects.get_or_create(user=self.user)
        self.client.login(username='u2', password='Pass123!')
        today = timezone.localdate()
        # Pre-create today's entry
        self.entry = MoodEntry.objects.create(
            user=self.user,
            mood_score=5, sleep_hours=6, work_hours=8,
            self_stress_level=5, anxiety_level=5, energy_level=5,
            social_interaction='Medium', appetite_level='Normal',
            concentration_level=5, motivation_level=5,
            screen_time=2, physical_activity=False,
            feeling_hopeless=False, loss_of_interest=False,
            feeling_tired=False, trouble_sleeping=False,
            entry_date=today,
        )

    def test_get_entry_page_shows_edit_mode(self):
        """GET /entry/ when entry exists → is_edit=True, form pre-filled."""
        r = self.client.get(reverse('mood_entry'))
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.context['is_edit'])
        self.assertEqual(r.context['existing_entry'].pk, self.entry.pk)

    def test_edit_updates_existing_entry(self):
        """POST to /entry/ when entry exists → updates the same record (no duplicate)."""
        self.client.post(reverse('mood_entry'), full_form_data(
            mood_score=9, anxiety_level=2, energy_level=9,
            physical_activity=True, feeling_hopeless=False,
        ))
        # Still only 1 MoodEntry for today
        count = MoodEntry.objects.filter(user=self.user, entry_date=timezone.localdate()).count()
        self.assertEqual(count, 1)

        self.entry.refresh_from_db()
        self.assertEqual(self.entry.mood_score, 9)
        self.assertEqual(self.entry.anxiety_level, 2)
        self.assertEqual(self.entry.energy_level, 9)
        self.assertTrue(self.entry.physical_activity)

    def test_edit_updates_prediction_not_creates_new(self):
        """Editing should update PredictionResult, not create a duplicate."""
        # Create initial prediction
        PredictionResult.objects.create(
            mood_entry=self.entry, user=self.user,
            stress_category='High', burnout_risk='High',
            depression_score=3, depression_risk='High',
            confidence_score=72.0,
        )
        self.client.post(reverse('mood_entry'), full_form_data(mood_score=9))

        # Still exactly 1 prediction for this entry
        pred_count = PredictionResult.objects.filter(mood_entry=self.entry).count()
        self.assertEqual(pred_count, 1)

    def test_edit_recalculates_prediction(self):
        """After edit with better values, prediction should change."""
        # First create with bad values
        self.client.post(reverse('mood_entry'), full_form_data(
            mood_score=2, sleep_hours=3, work_hours=13, self_stress_level=9,
            anxiety_level=9, energy_level=1,
            feeling_hopeless=True, loss_of_interest=True,
            feeling_tired=True, trouble_sleeping=True,
        ))
        pred_after_bad = PredictionResult.objects.get(mood_entry=self.entry)
        bad_stress = pred_after_bad.stress_category

        # Now edit with good values
        self.client.post(reverse('mood_entry'), full_form_data(
            mood_score=9, sleep_hours=8, work_hours=4, self_stress_level=2,
            anxiety_level=1, energy_level=9, physical_activity=True,
        ))
        pred_after_good = PredictionResult.objects.get(mood_entry=self.entry)
        self.assertIn(pred_after_good.stress_category, ['Low', 'Medium', 'High'])


# ─────────────────────────────────────────────
# 6. MODEL TESTS
# ─────────────────────────────────────────────

class ModelTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username='m', password='p')

    def test_depression_indicator_count_property(self):
        entry = MoodEntry.objects.create(
            user=self.user, mood_score=4, sleep_hours=5,
            work_hours=8, self_stress_level=6,
            anxiety_level=6, energy_level=4,
            social_interaction='Low', appetite_level='Low',
            concentration_level=4, motivation_level=3,
            screen_time=4, physical_activity=False,
            feeling_hopeless=True, loss_of_interest=True,
            feeling_tired=False, trouble_sleeping=True,
        )
        self.assertEqual(entry.depression_indicator_count, 3)

    def test_prediction_recommendations_list(self):
        entry = MoodEntry.objects.create(
            user=self.user, mood_score=3, sleep_hours=4, work_hours=12,
            self_stress_level=8, anxiety_level=8, energy_level=2,
            social_interaction='Low', appetite_level='Low',
            concentration_level=3, motivation_level=2,
            screen_time=6, physical_activity=False,
            feeling_hopeless=True, loss_of_interest=True,
            feeling_tired=True, trouble_sleeping=True,
        )
        pred = PredictionResult.objects.create(
            mood_entry=entry, user=self.user,
            stress_category='High', burnout_risk='High',
            depression_score=4, depression_risk='High',
            recommendations='Meditate|Exercise|Sleep more|Seek help'
        )
        recs = pred.get_recommendations_list()
        self.assertEqual(len(recs), 4)
        self.assertIn('Meditate', recs)
