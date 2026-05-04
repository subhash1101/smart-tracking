"""
predictor.py — Enhanced prediction module for Smart Mental Health Monitoring System.

Accepts all 16 form fields, builds the feature vector, runs Random Forest,
computes burnout risk + depression risk, and returns rich recommendations.

Usage:
    from tracker.ml.predictor import predict_stress

    result = predict_stress(
        mood=6, sleep=5.5, work=10, self_stress=7,
        anxiety=7, energy=4, social='Low', appetite='Low',
        concentration=4, motivation=3, screen_time=6,
        physical_activity=False,
        feeling_hopeless=True, loss_of_interest=True,
        feeling_tired=True, trouble_sleeping=False,
    )
"""

import pickle
import os
import numpy as np

ML_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(ML_DIR, 'stress_model.pkl')
ENCODER_PATH= os.path.join(ML_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(ML_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(ML_DIR, 'feature_columns.pkl')

_model    = None
_encoder  = None
_scaler   = None
_features = None


def _load():
    global _model, _encoder, _scaler, _features
    if _model is None:
        with open(MODEL_PATH,   'rb') as f: _model   = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f: _encoder = pickle.load(f)
        with open(SCALER_PATH,  'rb') as f: _scaler  = pickle.load(f)
        with open(FEATURES_PATH,'rb') as f: _features= pickle.load(f)


# ── Categorical encoders ──────────────────────────────────────────────────────
_SOCIAL_MAP   = {'Low': 0, 'Medium': 1, 'High': 2}
_APPETITE_MAP = {'Low': 0, 'Normal': 1, 'High': 2}


def predict_stress(
    mood: int,
    sleep: float,
    work: float,
    self_stress: int,
    anxiety: int       = 5,
    energy: int        = 5,
    social: str        = 'Medium',
    appetite: str      = 'Normal',
    concentration: int = 5,
    motivation: int    = 5,
    screen_time: float = 0.0,
    physical_activity: bool = False,
    feeling_hopeless: bool  = False,
    loss_of_interest: bool  = False,
    feeling_tired: bool     = False,
    trouble_sleeping: bool  = False,
) -> dict:
    """
    Predict stress category + burnout risk + depression risk.

    Returns:
        stress_category  : 'Low' | 'Medium' | 'High'
        burnout_risk     : 'Low' | 'Medium' | 'High'
        depression_risk  : 'Low' | 'Medium' | 'High'
        depression_score : int 0–4
        confidence       : float (percentage, e.g. 82.5)
        recommendations  : list[str]
    """
    _load()

    # ── Build feature vector (order must match FEATURE_COLS in train_model.py)
    social_enc   = _SOCIAL_MAP.get(social, 1)
    appetite_enc = _APPETITE_MAP.get(appetite, 1)

    features = np.array([[
        mood, sleep, work, self_stress,
        anxiety, energy, social_enc, appetite_enc,
        concentration, motivation, screen_time,
        int(physical_activity),
        int(feeling_hopeless), int(loss_of_interest),
        int(feeling_tired), int(trouble_sleeping),
    ]])

    # Predict
    pred_enc      = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]
    confidence    = float(probabilities.max()) * 100

    stress_category = _encoder.inverse_transform([pred_enc])[0]

    # ── Depression risk (0–4 indicator count) ───────────────────────────
    depression_score = sum([
        int(feeling_hopeless), int(loss_of_interest),
        int(feeling_tired),    int(trouble_sleeping),
    ])

    if depression_score >= 3:
        depression_risk = 'High'
    elif depression_score >= 1:
        depression_risk = 'Medium'
    else:
        depression_risk = 'Low'

    # ── Burnout risk score ───────────────────────────────────────────────
    burnout_score = 0

    if stress_category == 'High':   burnout_score += 3
    elif stress_category == 'Medium': burnout_score += 1

    if sleep < 5:    burnout_score += 2
    elif sleep < 6.5: burnout_score += 1

    if work > 10:    burnout_score += 2
    elif work > 8:   burnout_score += 1

    if mood <= 3:    burnout_score += 2
    elif mood <= 5:  burnout_score += 1

    if energy <= 3:  burnout_score += 2
    elif energy <= 5: burnout_score += 1

    if motivation <= 3: burnout_score += 1

    if depression_score >= 3: burnout_score += 2
    elif depression_score >= 1: burnout_score += 1

    if physical_activity: burnout_score -= 1  # exercise is protective

    if burnout_score >= 7:   burnout_risk = 'High'
    elif burnout_score >= 4: burnout_risk = 'Medium'
    else:                    burnout_risk = 'Low'

    # ── Smart Recommendation Engine ──────────────────────────────────────
    recs = []

    # — Stress-based —
    if stress_category == 'High':
        recs.append("🧘 Practice 10 minutes of deep breathing or box breathing right now.")
        recs.append("📵 Take a 30-minute digital detox — close all apps and sit in silence.")
        recs.append("🚶 Go for a 15-minute walk outdoors to lower your cortisol levels.")
    elif stress_category == 'Medium':
        recs.append("🎵 Listen to calming music or nature sounds for 15 minutes.")
        recs.append("📓 Write down 3 things you're grateful for today.")
        recs.append("🤸 Do a light 10-minute stretching or yoga routine.")
    else:
        recs.append("🌟 Excellent! You're managing stress well today. Keep your habits.")
        recs.append("📚 Use this clarity for your most creative or challenging tasks.")

    # — Anxiety-based —
    if anxiety >= 8:
        recs.append("😤 Try the 5-4-3-2-1 grounding technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.")
    elif anxiety >= 6:
        recs.append("💆 Progressive muscle relaxation for 10 minutes can reduce anxiety significantly.")

    # — Sleep-based —
    if sleep < 5:
        recs.append("😴 Critical sleep deprivation detected. Aim for 7–9 hours tonight — no exceptions.")
        recs.append("🌙 Avoid screens for 1 hour before bed and keep a consistent wake-up time.")
    elif sleep < 6.5:
        recs.append("💤 You need more sleep. Try to get at least 7 hours tonight for full recovery.")
    if trouble_sleeping:
        recs.append("🛌 For better sleep: keep your room cool and dark, avoid caffeine after 2 PM, and try 4-7-8 breathing to fall asleep.")

    # — Work-based —
    if work > 10:
        recs.append("⏰ You've been working excessively. Mandatory breaks every 90 minutes are essential to prevent burnout.")
        recs.append("🛑 Set a hard stop time for work today. Your brain needs recovery time.")
    elif work > 8:
        recs.append("☕ Use the Pomodoro technique: 25 minutes focused work, 5-minute break.")

    # — Energy-based —
    if energy <= 3:
        recs.append("⚡ Your energy is critically low. Rest should be your top priority today.")
    elif energy <= 5:
        recs.append("🥗 Low energy? Try a healthy snack, a short walk, or 10 minutes of sunlight.")

    # — Social interaction —
    if social == 'Low':
        recs.append("🤝 Social isolation worsens mental health. Try to connect with at least one person today — even a short text message helps.")

    # — Screen time —
    if screen_time >= 6:
        recs.append("📵 High screen time is linked to increased anxiety and poor sleep. Try a 1-hour screen break this evening.")
    elif screen_time >= 4:
        recs.append("📱 Consider reducing your screen time. Set app limits or use grayscale mode to reduce engagement.")

    # — Physical activity —
    if not physical_activity:
        recs.append("🏃 No exercise today? Even a 10-minute walk can boost mood by 20% through endorphin release.")

    # — Mood-based —
    if mood <= 3:
        recs.append("💙 Your mood is very low. Please reach out to a trusted friend, family member, or counselor today.")
        recs.append("🌿 Spend time in nature or do one small thing you enjoy — even for 15 minutes.")
    elif mood <= 5:
        recs.append("😊 Boost your mood with light exercise, listening to uplifting music, or calling a friend.")

    # — Motivation-based —
    if motivation <= 3:
        recs.append("🎯 Break your tasks into very small steps (5 minutes each) to rebuild momentum gradually.")

    # — Concentration-based —
    if concentration <= 4:
        recs.append("🧠 Poor concentration? Try a 5-minute mindfulness session before your next task, and eliminate distractions.")

    # — Appetite-based —
    if appetite == 'Low':
        recs.append("🍽️ Low appetite may indicate stress or depression. Try eating small, nutritious meals every 3–4 hours.")
    elif appetite == 'High':
        recs.append("🥦 Stress eating is common. Try replacing snacks with fruit, nuts, or water to manage cravings.")

    # — Depression indicators —
    if depression_risk == 'High':
        recs.append("🩺 You're showing multiple signs of depression today. Please speak with a mental health professional or a counselor — this is important.")
        recs.append("📞 VANDREVALA Foundation Helpline (India): 1860-2662-345 (24/7, free)")
    elif depression_risk == 'Medium':
        recs.append("💬 Some depression signals detected. Talk to someone you trust today and consider speaking with a professional if this persists.")

    if feeling_hopeless:
        recs.append("🌈 Hopeless feelings are often temporary distortions. Write down one small thing that went well today, however small.")

    if loss_of_interest:
        recs.append("🎨 Loss of interest is a key depression symptom. Try 'behavioral activation' — schedule one enjoyable activity for tomorrow.")

    # — Burnout —
    if burnout_risk == 'High':
        recs.append("🚨 High burnout risk! Schedule at least one full rest day this week without work or obligations.")
        recs.append("🩺 If this pattern continues for more than 2 weeks, consult a doctor or mental health professional.")

    return {
        'stress_category':  stress_category,
        'burnout_risk':     burnout_risk,
        'depression_risk':  depression_risk,
        'depression_score': depression_score,
        'confidence':       round(confidence, 1),
        'recommendations':  recs,
    }
