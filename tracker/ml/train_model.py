"""
train_model.py — Enhanced ML Training Script
Features: 14 inputs (basic + psychological + depression indicators)
Models:   Logistic Regression | Decision Tree | Random Forest (best)
Target:   stress_category (Low / Medium / High)

Run once before starting Django:
    python tracker/ml/train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET (1500 samples)
# ─────────────────────────────────────────────

np.random.seed(42)
N = 1500

def generate_dataset(n):
    rows = []
    for _ in range(n):
        # Basic
        mood          = np.random.randint(1, 11)
        sleep         = round(np.random.uniform(2, 10), 1)
        work          = round(np.random.uniform(2, 14), 1)
        self_stress   = np.random.randint(1, 11)

        # Mental state
        anxiety       = np.random.randint(1, 11)
        energy        = np.random.randint(1, 11)
        social        = np.random.choice([0, 1, 2])   # Low=0 Med=1 High=2
        appetite      = np.random.choice([0, 1, 2])   # Low=0 Normal=1 High=2
        concentration = np.random.randint(1, 11)
        motivation    = np.random.randint(1, 11)
        screen        = round(np.random.uniform(0, 10), 1)
        physical      = np.random.randint(0, 2)       # 0/1 boolean

        # Depression indicators (correlated with low mood/high stress)
        base_dep_prob = max(0, (11 - mood) / 10) * 0.6 + (self_stress / 10) * 0.4
        hopeless      = int(np.random.random() < base_dep_prob * 0.7)
        loss_interest = int(np.random.random() < base_dep_prob * 0.75)
        tired         = int(np.random.random() < base_dep_prob * 0.8)
        trouble_sleep = int(np.random.random() < base_dep_prob * 0.65)

        depression_count = hopeless + loss_interest + tired + trouble_sleep

        # ── Stress score formula ─────────────────────────────────────────
        score = (
            (11 - mood)       * 1.8 +   # low mood → high stress
            (8  - sleep)      * 1.4 +   # low sleep → high stress
            (work - 6)        * 0.9 +   # overwork → high stress
            self_stress       * 1.6 +   # self report
            anxiety           * 1.5 +   # anxiety adds stress
            (11 - energy)     * 0.8 +   # low energy
            (2  - social)     * 0.5 +   # low social
            (11 - concentration) * 0.4 +
            (11 - motivation) * 0.5 +
            depression_count  * 2.5 +   # depression indicators spike stress
            screen            * 0.3 -   # screen time adds a little
            physical          * 2.0     # exercise reduces stress
        )
        score += np.random.normal(0, 3)  # noise

        # Label thresholds (tuned so ~35% Low, ~40% Medium, ~25% High)
        if score < 28:
            label = 'Low'
        elif score < 46:
            label = 'Medium'
        else:
            label = 'High'

        rows.append({
            'mood_score':          mood,
            'sleep_hours':         sleep,
            'work_hours':          work,
            'self_stress_level':   self_stress,
            'anxiety_level':       anxiety,
            'energy_level':        energy,
            'social_interaction':  social,
            'appetite_level':      appetite,
            'concentration_level': concentration,
            'motivation_level':    motivation,
            'screen_time':         screen,
            'physical_activity':   physical,
            'feeling_hopeless':    hopeless,
            'loss_of_interest':    loss_interest,
            'feeling_tired':       tired,
            'trouble_sleeping':    trouble_sleep,
            'stress_category':     label,
        })

    return pd.DataFrame(rows)


df = generate_dataset(N)
print("=" * 55)
print("DATASET SUMMARY")
print("=" * 55)
print(f"Shape: {df.shape}")
print("\nClass distribution:")
print(df['stress_category'].value_counts())
print(f"\nClass %: {(df['stress_category'].value_counts(normalize=True)*100).round(1).to_dict()}")

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'mood_score', 'sleep_hours', 'work_hours', 'self_stress_level',
    'anxiety_level', 'energy_level', 'social_interaction',
    'appetite_level', 'concentration_level', 'motivation_level',
    'screen_time', 'physical_activity',
    'feeling_hopeless', 'loss_of_interest', 'feeling_tired', 'trouble_sleeping',
]

X = df[FEATURE_COLS].values
y = df['stress_category'].values

# Encode labels: High=0, Low=1, Medium=2 (alphabetical by default)
le = LabelEncoder()
le.fit(['Low', 'Medium', 'High'])
y_enc = le.transform(y)

print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. TRAIN & COMPARE 3 MODELS
# ─────────────────────────────────────────────

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=3, class_weight='balanced', random_state=42),
}

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

results = {}
for name, clf in models.items():
    if name == 'Logistic Regression':
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(
        clf,
        X_train_scaled if name == 'Logistic Regression' else X_train,
        y_train,
        cv=5
    ).mean()

    results[name] = {'acc': acc, 'cv': cv, 'model': clf, 'pred': y_pred}
    print(f"\n{name}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  CV Accuracy   : {cv*100:.2f}%")

# ─────────────────────────────────────────────
# 4. BEST MODEL = RANDOM FOREST
# ─────────────────────────────────────────────

best_name  = 'Random Forest'
best_model = results[best_name]['model']
best_pred  = results[best_name]['pred']

print("\n" + "=" * 55)
print(f"BEST MODEL: {best_name}")
print("=" * 55)
print(classification_report(y_test, best_pred, target_names=le.classes_))

# Feature importance
importances = best_model.feature_importances_
print("\nTop Feature Importances:")
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:8]:
    bar = '█' * int(imp * 60)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# 5. SAVE MODEL, ENCODER, SCALER, FEATURE LIST
# ─────────────────────────────────────────────

ML_DIR = os.path.dirname(os.path.abspath(__file__))

# Save dataset
df.to_csv(os.path.join(ML_DIR, 'sample_dataset.csv'), index=False)

paths = {
    'stress_model.pkl':    best_model,
    'label_encoder.pkl':   le,
    'scaler.pkl':          scaler,
    'feature_columns.pkl': FEATURE_COLS,
}

for filename, obj in paths.items():
    path = os.path.join(ML_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Saved → {path}")

print("\n✅ Training complete. You can now run the Django server.")

# ─────────────────────────────────────────────
# 6. SAMPLE PREDICTIONS (sanity check)
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("SAMPLE PREDICTIONS")
print("=" * 55)

samples = [
    # mood,slp,wrk,str,anx,eng,soc,apt,con,mot,scr,phy,hop,loi,ftd,trs
    [9, 8.0, 4.0, 2,  1, 9, 2, 1, 9, 9, 1.0, 1, 0, 0, 0, 0],  # Expected: Low
    [5, 6.0, 8.0, 5,  5, 5, 1, 1, 5, 5, 4.0, 0, 1, 0, 1, 0],  # Expected: Medium
    [2, 3.5, 13, 9,  9, 2, 0, 0, 2, 2, 8.0, 0, 1, 1, 1, 1],  # Expected: High
]

for i, s in enumerate(samples):
    pred = le.inverse_transform(best_model.predict([s]))[0]
    prob = best_model.predict_proba([s]).max()
    print(f"  Sample {i+1}: Predicted = {pred} (confidence {prob*100:.1f}%)")
