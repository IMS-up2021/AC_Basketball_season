# ==============================================
# Phase 3: Regular Season Team Rankings by Conference
# ==============================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

#Setup
sns.set_style("whitegrid")
output_dir = "results/predictions"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# =========================================================
# STEP 3.1: LOAD DATA
# =========================================================
data_path = "data/cleaned/full_dataset_prepared.csv"
teams_path = "data/teams.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
teams_df_raw = pd.read_csv(teams_path)

print(f"Loaded player data: {df.shape}")
print(f"Loaded teams dats: {teams_df_raw.shape}")

# Ensure we handle the conference column name correctly
if 'confID' not in df.columns:
    if 'lgID_x' in df.columns:
        df.rename(columns={'lgID_x': 'conference'}, inplace=True)
    elif 'lgID' in df.columns:
        df.rename(columns={'lgID': 'conference'}, inplace=True)
else:
    df.rename(columns={'confID': 'conference'}, inplace=True)

df['conference'] = df['conference'].fillna('Unknown')

# =========================================================
# STEP 3.2: FEATURE ENGINNERING - PYTHAGOREAN EXPECTATION
# =========================================================
print("Calculating Pythagorean Expectation...")

teams_df_raw['pythag_win_pct'] = (teams_df_raw['o_pts']**13.91) / (teams_df_raw['o_pts']**13.91 + teams_df_raw['d_pts']**13.91)

teams_df_raw['point_diff_pg'] = (teams_df_raw['o_pts'] - teams_df_raw['d_pts']) / teams_df_raw['GP']

teams_df_raw['next_year'] = teams_df_raw['year'] + 1
pythag_features = teams_df_raw[['tmID', 'next_year', 'pythag_win_pct', 'point_diff_pg']].rename(columns={'next_year': 'year', 'pythag_win_pct': 'prev_pythag_expectation', 'point_diff_pg': 'prev_point_diff'})

# =========================================================
# STEP 3.3: WEIGHTED ROSTER CONTINUITY
# =========================================================
print("Calculating Weighted Roster Continuity (Efficiency Returning)...")

player_stats_cols = ['playerID', 'year', 'tmID', 'minutes', 'points', 'win_pct', 'efficiency']
if 'efficiency' not in df.columns:
    df['efficiency'] = df['points'] + df['rebounds'] + df['assists'] + df['steals'] + df['blocks'] - df['turnovers']

team_total_eff = df.groupby(['tmID', 'year'])['efficiency'].sum().reset_index(name='team_total_efficiency')

player_history = df[player_stats_cols].copy()
player_history['next_year'] = player_history['year'] + 1
player_history = player_history.rename(columns={
    'minutes': 'prev_min',
    'points': 'prev_pts',
    'win_pct': 'prev_player_win_pct',
    'efficiency': 'prev_player_eff',
    'year': 'prev_year',
    'tmID': 'prev_tmID'
})

df_roster = df[['year', 'tmID', 'playerID']].merge(
    player_history,
    left_on=['playerID', 'year'],
    right_on=['playerID', 'next_year'],
    how='left'
)

df_roster = df_roster[df_roster['tmID'] == df_roster['prev_tmID']]

roster_continuity_agg = df_roster.groupby(['year', 'tmID'])['prev_player_eff'].sum().reset_index(name='returning_efficiency_total')

team_total_eff['next_year'] = team_total_eff['year'] + 1
prev_team_totals = team_total_eff[['tmID', 'next_year', 'team_total_efficiency']].rename(columns={'next_year': 'year', 'team_total_efficiency': 'prev_year_team_total_eff'})

roster_features = roster_continuity_agg.merge(prev_team_totals, on=['year', 'tmID'], how='left')

roster_features['weighted_continuity'] = roster_features['returning_efficiency_total'] / roster_features['prev_year_team_total_eff']

roster_features['weighted_continuity'] = roster_features['weighted_continuity'].fillna(0).clip(0, 1.1)

print("Weighted continuity calculated")

# =========================================================
# STEP 3.4: TEAM-LEVEL AGGREGATION & FINAL MERGE
# =========================================================
team_df = df.groupby(['year', 'tmID', 'conference'], as_index=False).agg({
    'points': 'sum',
    'fgMade': 'sum',
    'fgAttempted': 'sum',
    'steals': 'sum',
    'blocks': 'sum',
    'assists': 'sum',
    'turnovers': 'sum',
    'win_pct': 'max',
})

team_df['shooting_efficiency'] = team_df['fgMade'] / team_df['fgAttempted'].replace(0,1)
team_df['turnover_ratio'] = team_df['assists'] / team_df['turnovers'].replace(0,1)
team_df['def_impact'] = team_df['steals'] + team_df['blocks']

conf_strength = team_df.groupby(['year', 'conference'])['win_pct'].mean().reset_index()
conf_strength['year'] = conf_strength['year'] + 1
conf_strength.rename(columns={'win_pct': 'prev_conf_strength'}, inplace=True)

team_df = team_df.merge(conf_strength, on=['year', 'conference'], how='left')
team_df['prev_conf_strength'] = team_df['prev_conf_strength'].fillna(0.5)

lag_cols = ['shooting_efficiency', 'def_impact', 'win_pct']
for col in lag_cols:
    team_df[f'team_lag_{col}'] = team_df.groupby('tmID')[col].shift(1)

model_df = team_df.merge(pythag_features, on=['tmID', 'year'], how='left')
model_df = model_df.merge(roster_features[['year', 'tmID', 'weighted_continuity']], on=['year', 'tmID'], how='left')

model_df = model_df.dropna(subset=[f'team_lag_{col}' for col in lag_cols])

model_df['prev_pythag_expectation'] = model_df['prev_pythag_expectation'].fillna(model_df['team_lag_win_pct'])
model_df['weighted_continuity'] = model_df['weighted_continuity'].fillna(0.6)

# =========================================================
# STEP 3.5: MODEL TRAINING (XGBOOST)
# =========================================================
features = [
    'prev_pythag_expectation',
    'prev_point_diff',
    'team_lag_def_impact',
    'team_lag_shooting_efficiency',
    'weighted_continuity',
    'prev_conf_strength'
]

target = 'win_pct'

print(f"\nFinal Dataset: {len(model_df)} rows. Features selected:")
print(features)

train_df = model_df[model_df['year'] < 10]
test_df = model_df[model_df['year'] == 10].copy()

print(f"Training XGBoost Model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate = 0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1
)

model.fit(train_df[features], train_df[target])

test_df['raw_pred_win_pct'] = model.predict(test_df[features])

for conf in test_df['conference'].unique():
    conf_mask = test_df['conference'] == conf
    conf_mean = test_df.loc[conf_mask, 'raw_pred_win_pct'].mean()
    test_df.loc[conf_mask, 'predicted_win_pct'] = (test_df.loc[conf_mask, 'raw_pred_win_pct'] - conf_mean + 0.5).clip(0,1)

# Metrics
mae = mean_absolute_error(test_df['win_pct'], test_df['predicted_win_pct'])
r2 = r2_score(test_df['win_pct'], test_df['predicted_win_pct'])

print(f"\n--- Season 10 Prediction Results ---")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# =========================================================
# STEP 3.6: OUTPUTS 
# =========================================================
test_df['pred_rank'] = test_df.groupby('conference')['predicted_win_pct'].rank(ascending=False, method='dense')
test_df['actual_rank'] = test_df.groupby('conference')['win_pct'].rank(ascending=False, method='dense')
test_df['predicted_wins'] = (test_df['predicted_win_pct'] * 34).round(0).astype(int)

output = test_df[['conference', 'tmID', 'predicted_wins', 'win_pct', 'pred_rank', 'actual_rank', 'prev_pythag_expectation', 'weighted_continuity']]
output = output.sort_values(['conference', 'pred_rank'])

print("\nPredicted Season 10 Rankings:")
print(output)

output.to_csv(f"{output_dir}/season_10_rankings.csv", index=False)
print(f"\nSaved rankings to {output_dir}/season_10_rankings.csv")

# =========================================================
# STEP 3.7: PLOTS
# =========================================================

# 1. Accuracy Plot
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=test_df['win_pct'], 
    y=test_df['predicted_win_pct'], 
    hue=test_df['conference'], 
    s=120, palette='viridis'
)
plt.plot([0,1], [0,1], 'r--', label='Perfect Fit')
plt.xlabel('Actual Win % (Year 10)')
plt.ylabel('Predicted Win % (Year 10)')
plt.title(f'Prediction Accuracy (MAE: {mae:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/season10_pred_vs_actual.png")
plt.close()
print(f" Saved accuracy plot to {plot_dir}/season10_pred_vs_actual.png")

# 2. Feature Importance
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='magma')
plt.title('Drivers of Team Success (XGBoost)')
plt.tight_layout()
plt.savefig(f"{plot_dir}/ranking_feature_importance.png")
plt.close()
print(f" Saved feature importance plot to {plot_dir}/ranking_feature_importance.png")

# 3. Plot: Predicted Rank vs Actual Rank
plt.figure(figsize=(12, 7))

# Ordenar por ranking real (para visualização clara)
bar_df = test_df.sort_values("actual_rank")

# Largura para deslocar as barras
x = np.arange(len(bar_df))
width = 0.35

plt.bar(x - width/2, bar_df["actual_rank"], width, label="Actual Rank")
plt.bar(x + width/2, bar_df["pred_rank"], width, label="Predicted Rank (XGB)")

plt.xticks(x, bar_df["tmID"], rotation=45, ha='right')
plt.xlabel("Teams")
plt.ylabel("Ranking")
plt.title("Rank Comparison: Actual vs XGBoost Trediction (Year 10)")
plt.gca().invert_yaxis()  # ranking 1 como melhor (topo)
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/season10_rank_barplot.png")
plt.close()

print(f" Saved rank comparison barplot to {plot_dir}/season10_rank_barplot.png")

print("\n Completed: Team Rankings with Roster Stability & Momentum.")