import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# CONFIGURATION
# =========================================================
PREDICTION_YEAR = 11
LAST_TRAIN_YEAR = 10

# Visual Setup
sns.set_style("whitegrid")
output_dir = "results/predictions"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print(f"========================================================")
print(f"COACH FIRING PREDICTION (HOT SEAT MODEL)")
print(f"========================================================")

# =========================================================
# 1. LOAD DATA
# =========================================================
coaches_path = "data/cleaned/coaches_cleaned.csv"
rankings_path = f"results/predictions/season_{PREDICTION_YEAR}_rankings.csv"

if not os.path.exists(coaches_path):
    raise FileNotFoundError(f"File not found: {coaches_path}")

print("--> Loading Coaches Data...")
df = pd.read_csv(coaches_path)

# Load predicted rankings (from the previous script)
if os.path.exists(rankings_path):
    print("--> Loading Season 11 Predicted Standings...")
    rankings_df = pd.read_csv(rankings_path)
    
    # Create map: {TeamID: Predicted_Win_Pct}
    # Ensure IDs are strings
    rankings_df['tmID'] = rankings_df['tmID'].astype(str)
    pred_map = dict(zip(rankings_df.tmID, rankings_df.predicted_win_pct))
    has_predictions = True
else:
    print("WARNING: Rankings file not found. Predictions will be inaccurate (assuming 0 wins).")
    has_predictions = False
    pred_map = {}

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

# Ensure correct sorting
df = df.sort_values(['tmID', 'year', 'stint'])

# Filter: We only want the last coach of each year per team
# (Usually the one who starts the next season, or is evaluated at the end of the season)
df_yearly = df.drop_duplicates(subset=['year', 'tmID'], keep='last').copy()

# --- 2.1 INJECT SEASON 11 PREDICTIONS ---
# Replace win_pct (which is 0.0) with the prediction from rankings.py
if has_predictions:
    mask_s11 = df_yearly['year'] == PREDICTION_YEAR
    # Map predictions
    predicted_values = df_yearly.loc[mask_s11, 'tmID'].map(pred_map)
    # Fill where prediction found (if not found, keep existing)
    df_yearly.loc[mask_s11, 'win_pct'] = predicted_values.fillna(df_yearly.loc[mask_s11, 'win_pct'])
    print("--> Updated Season 11 win_pct with predicted values.")

# --- 2.2 DEFINE TARGET (Coach Change) ---
# If next year's coachID is different, a change occurred.
df_yearly['next_coachID'] = df_yearly.groupby('tmID')['coachID'].shift(-1)
df_yearly['coach_change'] = (df_yearly['coachID'] != df_yearly['next_coachID']).astype(int)

# Note: In the last row for each team (S11), next_coachID is NaN. 
# This is fine, as S11 is the TEST set (we want to predict, not train on it).

# --- 2.3 ADVANCED FEATURES ---

# Trend: Current performance vs Previous year
df_yearly['prev_win_pct'] = df_yearly.groupby('tmID')['win_pct'].shift(1).fillna(0.5)
df_yearly['trend'] = df_yearly['win_pct'] - df_yearly['prev_win_pct']

# Lifetime Win % (Accumulated history of the coach)
# Calculate the expanding mean of win_pct up to that year for that coach
df_yearly['lifetime_win_pct'] = df_yearly.groupby('coachID')['win_pct'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.5)

# Postseason Success (Recent success?)
# Rolling mean of playoff wins in the last 2 years
df_yearly['recent_playoff_success'] = df_yearly.groupby('coachID')['post_wins'].transform(lambda x: x.rolling(2, min_periods=1).mean()).fillna(0)

# =========================================================
# 3. MODEL TRAINING
# =========================================================
features = [
    'win_pct', 
    'trend', 
    'tenure_years', 
    'lifetime_win_pct', 
    'recent_playoff_success',
    'is_new_coach' # Do first-year coaches have a "honeymoon" period?
]
target = 'coach_change'

# Training Set: Years 1 to 10
# Important: Remove rows where we don't know the target (NaN)
train_df = df_yearly[df_yearly['year'] < PREDICTION_YEAR].dropna(subset=['next_coachID'])

# Prediction Set: Year 11
predict_df = df_yearly[df_yearly['year'] == PREDICTION_YEAR].copy()

print(f"Training on {len(train_df)} historical seasons.")
print(f"Predicting for {len(predict_df)} active coaches in Season {PREDICTION_YEAR}.")

# Random Forest
rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42, class_weight='balanced')
rf.fit(train_df[features], train_df[target])

# =========================================================
# 4. PREDICTION AND RESULTS
# =========================================================

# Calculate probabilities
probs = rf.predict_proba(predict_df[features])[:, 1]
predict_df['firing_prob'] = probs

# Define Risk Categories
predict_df['Risk Status'] = pd.cut(
    predict_df['firing_prob'], 
    bins=[-0.1, 0.35, 0.65, 1.1], 
    labels=['Safe', 'In Danger', 'Hot Seat']
)

# Select final columns
results = predict_df[['tmID', 'coachID', 'win_pct', 'tenure_years', 'trend', 'firing_prob', 'Risk Status']]
results = results.sort_values('firing_prob', ascending=False)

# Formatting for display
results_display = results.copy()
results_display['win_pct'] = (results_display['win_pct'] * 100).round(1).astype(str) + '%'
results_display['trend'] = (results_display['trend'] * 100).round(1).astype(str) + '%'
results_display['firing_prob'] = (results_display['firing_prob'] * 100).round(1).astype(str) + '%'

print("\n========================================================")
print("COACH HOT SEAT REPORT (Season 11 Forecast)")
print("========================================================")
print(results_display.to_string(index=False))

# Save CSV
csv_out = f"{output_dir}/season_{PREDICTION_YEAR}_coach_changes.csv"
results.to_csv(csv_out, index=False)
print(f"\n--> Detailed predictions saved to: {csv_out}")

# =========================================================
# 5. VISUALIZATION
# =========================================================
plt.figure(figsize=(10, 6))
colors = {'Safe': 'green', 'In Danger': 'orange', 'Hot Seat': 'red'}

sns.barplot(
    x='firing_prob', 
    y='coachID', 
    data=results, 
    hue='Risk Status', 
    palette=colors, 
    dodge=False
)

plt.axvline(0.5, color='black', linestyle='--', alpha=0.3)
plt.title(f'Likelihood of Coaching Change After Season {PREDICTION_YEAR}\n(Based on Predicted Team Performance)')
plt.xlabel('Probability of Change (0-1)')
plt.ylabel('Head Coach')
plt.xlim(0, 1)
plt.legend(title='Risk Level', loc='lower right')
plt.tight_layout()

plot_path = f"{plot_dir}/coach_hot_seat_s{PREDICTION_YEAR}.png"
plt.savefig(plot_path)
print(f"--> Chart saved to: {plot_path}")

# Feature Importance
print("\nModel Insights - What gets coaches fired?")
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(imp.to_string())