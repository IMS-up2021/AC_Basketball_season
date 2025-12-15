import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# =========================================================
# CONFIGURATION
# =========================================================
PREDICTION_YEAR = 11      # The season we want to predict
LAST_TRAIN_YEAR = 10      # The last season with real data for training

# Folder setup
sns.set_style("whitegrid")
output_dir = "results/predictions"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print(f"========================================================")
print(f"PREDICTING SEASON {PREDICTION_YEAR} STANDINGS")
print(f"Training model on seasons 1-{LAST_TRAIN_YEAR}")
print(f"========================================================")

# =========================================================
# 1. LOAD DATA
# =========================================================
data_path = "data/cleaned/full_dataset_prepared.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError("Run data_prepare.py first!")

df = pd.read_csv(data_path)

# Normalize conference names and IDs
if 'confID' not in df.columns:
    col_map = {'lgID_x': 'conference', 'lgID': 'conference'}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
else:
    df.rename(columns={'confID': 'conference'}, inplace=True)

df['conference'] = df['conference'].fillna('Unknown')
df['tmID'] = df['tmID'].astype(str)

# Fill missing metrics for Season 11 with 0 (to prevent errors during calculation)
cols_to_fill = ['points', 'efficiency', 'win_pct']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

print("--> Building Features: Lag Stats & Roster Continuity...")

# --- 2.1 Lag Stats (Performance from the previous year) ---
team_stats = df.groupby(['tmID', 'year']).agg({
    'points': 'sum', 
    'win_pct': 'max'
}).reset_index()

team_stats['next_year'] = team_stats['year'] + 1

# FIX: Drop 'year' before renaming 'next_year' to avoid duplicate columns
lag_merge = team_stats.drop(columns=['year']).rename(columns={
    'next_year': 'year', 
    'win_pct': 'prev_win_pct', 
    'points': 'prev_points'
})

# --- 2.2 Roster Continuity (Who stayed on the team?) ---
if 'efficiency' not in df.columns:
    df['efficiency'] = (df['points'] + df['rebounds'].fillna(0) + 
                        df['assists'].fillna(0) + df['steals'].fillna(0) - 
                        df['turnovers'].fillna(0))

# Aux Table: Player performance in the PREVIOUS year
player_prev = df[['playerID', 'year', 'tmID', 'efficiency']].copy()
player_prev['next_year'] = player_prev['year'] + 1
player_prev = player_prev.rename(columns={
    'efficiency': 'prev_efficiency',
    'tmID': 'prev_tmID',
    'year': 'orig_year',
    'next_year': 'year'
})

# Merge: Current roster (df) vs past performance (player_prev)
roster_merge = df[['year', 'tmID', 'playerID']].merge(
    player_prev,
    on=['playerID', 'year'],
    how='left'
)

# Filter: Keep only players who STAYED on the same team
staying_players = roster_merge[roster_merge['tmID'] == roster_merge['prev_tmID']]

# Sum the efficiency returning to the team
continuity_sum = staying_players.groupby(['year', 'tmID'])['prev_efficiency'].sum().reset_index(name='returning_efficiency')

# Calculate TOTAL team efficiency in the previous year (for ratio calculation)
team_total_prev = df.groupby(['year', 'tmID'])['efficiency'].sum().reset_index()
team_total_prev['next_year'] = team_total_prev['year'] + 1

# FIX: Drop 'year' before renaming
team_total_prev = team_total_prev.drop(columns=['year']).rename(columns={
    'next_year': 'year',
    'efficiency': 'total_prev_eff'
})

# --- 2.3 Merge everything into Final Dataset ---
final_features = df[['year', 'tmID', 'conference']].drop_duplicates()

# Merge Lag Stats
final_features = final_features.merge(lag_merge, on=['year', 'tmID'], how='left')

# Merge Continuity
final_features = final_features.merge(continuity_sum, on=['year', 'tmID'], how='left')
final_features = final_features.merge(team_total_prev, on=['year', 'tmID'], how='left')

# Calculate Continuity Score (% of team efficiency that returned)
final_features['continuity_score'] = final_features['returning_efficiency'] / final_features['total_prev_eff']
final_features['continuity_score'] = final_features['continuity_score'].fillna(0).clip(0, 1.1)

# Fill nulls in lag stats (new teams or first season)
final_features['prev_win_pct'] = final_features['prev_win_pct'].fillna(0.5)
final_features['prev_points'] = final_features['prev_points'].fillna(final_features['prev_points'].mean())

# Add Actual Target (Win Pct) for training
actual_wins = df.groupby(['year', 'tmID'])['win_pct'].max().reset_index()
model_df = final_features.merge(actual_wins, on=['year', 'tmID'], how='left')

# =========================================================
# 3. TRAINING AND PREDICTION (XGBoost)
# =========================================================

features = ['prev_win_pct', 'prev_points', 'continuity_score']
target = 'win_pct'

# Split Train (Years 1-10) and Prediction (Year 11)
train_df = model_df[model_df['year'] <= LAST_TRAIN_YEAR].dropna(subset=[target])
predict_df = model_df[model_df['year'] == PREDICTION_YEAR].copy()

print(f"--> Data split: {len(train_df)} training rows, {len(predict_df)} teams to predict.")

if len(predict_df) == 0:
    print("ERROR: No teams found for Season 11. Check data_prepare.py.")
    exit()

# Train Model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(train_df[features], train_df[target])

# Predict Season 11
predict_df['predicted_win_pct'] = model.predict(predict_df[features])

# Calculate Wins (assuming 34 games)
predict_df['predicted_wins'] = (predict_df['predicted_win_pct'] * 34).round(0).astype(int)

# =========================================================
# 4. RESULTS AND EXPORT
# =========================================================

# Sort Standings
standings = predict_df[['conference', 'tmID', 'predicted_wins', 'predicted_win_pct', 'prev_win_pct', 'continuity_score']]
standings = standings.sort_values(by=['conference', 'predicted_wins'], ascending=[True, False])

print("\n========================================================")
print(f"PREDICTED STANDINGS FOR SEASON {PREDICTION_YEAR}")
print("========================================================")
print(standings.to_string(index=False))

# Save CSV
csv_path = f"{output_dir}/season_{PREDICTION_YEAR}_rankings.csv"
standings.to_csv(csv_path, index=False)
print(f"\n--> Saved rankings to: {csv_path}")

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x='predicted_wins', 
    y='tmID', 
    data=standings, 
    hue='conference', 
    palette='viridis', 
    dodge=False
)
plt.title(f'Predicted Wins - Season {PREDICTION_YEAR} (Based on Continuity & Past Performance)')
plt.xlabel('Projected Wins')
plt.ylabel('Team')
plt.legend(title='Conference', loc='lower right')
plt.tight_layout()
plt.savefig(f"{plot_dir}/season_{PREDICTION_YEAR}_forecast.png")
print(f"--> Saved plot to: {plot_dir}/season_{PREDICTION_YEAR}_forecast.png")