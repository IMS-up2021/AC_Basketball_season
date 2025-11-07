# ==============================================
# Phase 3: Regular Season Team Rankings by Conference
# ==============================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
group_cols = ['year', 'tmID']
if 'lgID_x' in df.columns:
    group_cols.append('lgID_x')
elif 'lgID_y' in df.columns:
    group_cols.append('lgID_y')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['playerID', 'coachID']]

team_df = df.groupby(group_cols, as_index=False)[numeric_cols].mean()

team_df['conference'] = (
    team_df['lgID_x'] if 'lgID_x' in team_df.columns else team_df['lgID_y']
)

# =========================================================
# STEP 3.3: FEATURE ENGINEERING
# =========================================================
team_df['off_efficiency'] = (team_df['points'] + team_df['assists']) / team_df['turnovers'].replace(0, np.nan)
team_df['def_efficiency'] = team_df['steals'] + team_df['blocks'] - team_df['PF']
team_df['net_efficiency'] = team_df['off_efficiency'] - team_df['PF']
team_df['rebound_strength'] = team_df['rebounds'] + team_df['dRebounds'] + team_df['oRebounds']
team_df['shooting_efficiency'] = (
    (team_df['fgMade'] + team_df['ftMade'] + team_df['threeMade'])
    / (team_df['fgAttempted'] + team_df['ftAttempted'] + team_df['threeAttempted']).replace(0, np.nan)
)

if 'win_pct' in df.columns:
    team_win_pct = df.groupby(group_cols, as_index=False)['win_pct'].mean()
    team_df = pd.merge(team_df, team_win_pct, on=group_cols, how='left')

team_df['wins'] = (
    team_df['off_efficiency'].rank(pct=True) * 15
    + team_df['def_efficiency'].rank(pct=True) * 10
    + team_df['shooting_efficiency'].rank(pct=True) * 5
)
team_df['wins'] = team_df['wins'].fillna(team_df['wins'].mean())

# =========================================================
# STEP 3.4: MODEL TRAINING (USANDO ANOS HISTÓRICOS)
# =========================================================
features = [
    'off_efficiency', 'def_efficiency', 'net_efficiency',
    'rebound_strength', 'shooting_efficiency'
]
if 'win_pct' in team_df.columns:
    features.append('win_pct')

max_year = team_df['year'].max()
train_df = team_df[team_df['year'] < max_year]

X_train = train_df[features].fillna(0)
y_train = train_df['wins']

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Validação (histórica)
X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = sqrt(mean_squared_error(y_val, y_pred))
print(f"\n📊 Model Performance: MAE={mae:.3f}, RMSE={rmse:.3f}")

# =========================================================
# STEP 3.5: HISTORICAL RANKINGS
# =========================================================
train_df['predicted_wins'] = model.predict(X_train)

rankings = (
    train_df[['year', 'conference', 'tmID', 'predicted_wins']]
    .sort_values(by=['year', 'conference', 'predicted_wins'], ascending=[True, True, False])
    .drop_duplicates(subset=['year', 'tmID', 'conference'])
    .reset_index(drop=True)
)

rankings['rank'] = rankings.groupby(['year', 'conference'])['predicted_wins'].rank(
    ascending=False, method='dense'
)
rankings = rankings.sort_values(by=['year', 'conference', 'rank']).reset_index(drop=True)

# =========================================================
# STEP 3.6: FUTURE SEASON PREDICTION (YEAR 11)
# =========================================================
future_df = (
    team_df[team_df['year'] == max_year]
    .copy()
    .drop(columns=['wins'], errors='ignore')
)
future_df['year'] = max_year + 1

X_future = future_df[features].fillna(0)
future_df['predicted_wins'] = model.predict(X_future)

future_df['rank'] = future_df.groupby('conference')['predicted_wins'].rank(
    ascending=False, method='dense'
)

# =========================================================
# STEP 3.7: SAVE RESULTS
# =========================================================
os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("results", exist_ok=True)

historical_path = "data/cleaned/regular_season_team_rankings.csv"
future_path = "results/predicted_season_11_rankings.csv"

rankings['predicted_wins'] = rankings['predicted_wins'].round(0)
future_df['predicted_wins'] = future_df['predicted_wins'].round(0)

rankings_sorted = rankings[['year', 'conference', 'tmID', 'predicted_wins', 'rank']].sort_values(
    by=['year', 'conference', 'rank'], ascending=[True, True, True]
)
future_df_sorted = future_df[['year', 'conference', 'tmID', 'predicted_wins', 'rank']].sort_values(
    by=['conference', 'rank'], ascending=[True, True]
)

rankings_sorted.to_csv(historical_path, index=False)
future_df_sorted.to_csv(future_path, index=False)

# =========================================================
# STEP 3.8: DISPLAY RESULTS
# =========================================================
print("\n🏀 === Regular Season Team Rankings by Conference (Historical) ===")
for conf in rankings['conference'].dropna().unique():
    latest = rankings[rankings['year'] == max_year]
    subset = latest[latest['conference'] == conf]
    print(f"\n🏆 Conference: {conf}")
    print("Top 5 Teams (last season):")
    print(subset[['tmID', 'predicted_wins']].head(5).to_string(index=False))

print("\n🚀 === Predicted Rankings for Next Season (Year 11) ===")
for conf in future_df['conference'].dropna().unique():
    subf = future_df[future_df['conference'] == conf].sort_values(by='rank')
    print(f"\n🏆 Conference: {conf}")
    print(subf[['tmID', 'predicted_wins', 'rank']].head(10).to_string(index=False))

print("\n✅ Ranking generation complete.")
