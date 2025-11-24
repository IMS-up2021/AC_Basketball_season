# ==============================================
# Phase 3: Regular Season Team Rankings by Conference
# ==============================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# =========================================================
# STEP 3.1: LOAD DATA
# =========================================================
data_path = "data/cleaned/full_dataset_prepared.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# =========================================================
# STEP 3.2: TEAM-LEVEL AGGREGATION
# =========================================================
if 'lgID_x' in df.columns:
    df.rename(columns={'lgID_x': 'conference'}, inplace=True)
elif 'lgID_y' in df.columns:
    df.rename(columns={'lgID_y': 'conference'}, inplace=True)
elif 'lgID' in df.columns:
    df.rename(columns={'lgID': 'conference'}, inplace=True)

team_df = df.groupby(['year', 'tmID', 'conference'], as_index=False).agg({
    'points': 'sum',
    'rebounds': 'sum',
    'assists': 'sum',
    'steals': 'sum',
    'blocks': 'sum',
    'turnovers': 'sum',
    'PF': 'sum',
    'fgMade': 'sum',
    'fgAttempted': 'sum',
    'win_pct':'max'
})

# =========================================================
# STEP 3.3: FEATURE ENGINEERING
# =========================================================
team_df['shooting_efficiency'] = team_df['fgMade'] / team_df['fgAttempted'].replace(0,1)
team_df['turnover_ratio'] = team_df['assists'] / team_df['turnovers'].replace(0,1)
team_df['def_impact'] = team_df['steals'] + team_df['blocks']
team_df['foul_discipline'] = team_df['PF']

feature_cols = [
    'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'shooting_efficiency', 'turnover_ratio', 'def_impact', 'win_pct'
]

# =========================================================
# STEP 3.4: LAG GENERATION
# =========================================================
print("\n Generating Lag Features (Year N-1 -> Year N)...")

team_df = team_df.sort_values(['tmID', 'year'])

for col in feature_cols:
    team_df[f'prev_{col}'] = team_df.groupby('tmID')[col].shift(1)

model_df = team_df.dropna(subset=[f'prev_{col}' for col in feature_cols])

print(f"Dataset reduced from {len(team_df)} to {len(model_df)} rows after lag creation")

# =========================================================
# STEP 3.4: MODEL TRAINING (USANDO ANOS HISTÓRICOS)
# =========================================================

X = model_df[[f'prev_{col}' for col in feature_cols]]
y = model_df['win_pct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance: MAE={mae:.4f}, RMSE={r2:.4f}")

# =========================================================
# STEP 3.5: FUTURE SEASON PREDICTION (YEAR 11)
# =========================================================

max_year = team_df['year'].max()
latest_stats = team_df[team_df['year'] == max_year].copy()

pred_features = {}
for col in feature_cols:
    pred_features[f'prev_{col}'] = latest_stats[col]

X_future = pd.DataFrame(pred_features)

latest_stats['predicted_win_pct'] = model.predict(X_future)

latest_stats['predicted_wins'] = (latest_stats['predicted_win_pct'] * 34).round(0)

latest_stats['rank'] = latest_stats.groupby('conference')['predicted_wins'].rank(ascending=False, method='dense')

# =========================================================
# STEP 3.7: SAVE RESULTS
# =========================================================
os.makedirs("results", exist_ok=True)

output_cols = ['conference', 'tmID', 'predicted_wins', 'predicted_win_pct', 'rank']
future_rankings = latest_stats[output_cols].sort_values(['conference', 'rank'])

future_path = "results/predicted_season_11_rankings.csv"
future_rankings.to_csv(future_path, index=False)

print(f"\n Saved Season 11 Predictions to {future_path}")
# =========================================================
# STEP 3.8: DISPLAY RESULTS
# =========================================================
print("\n🚀 === Predicted Rankings for Next Season (Year 11) ===")
for conf in future_rankings['conference'].dropna().unique():
    print(f"\n🏆 Conference: {conf}")
    print(future_rankings[future_rankings['conference'] == conf][['rank', 'tmID', 'predicted_wins']].to_string(index=False))
print("\n✅ Ranking generation complete.")

print("\nFeature Importance:")
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importances.head(5))