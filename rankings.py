# ==============================================
# Phase 3: Regular Season Team Rankings by Conference
# ==============================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"Loaded dataset with shape: {df.shape}")

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
# STEP 3.2: PLAYER-LEVEL ROSTER COMPOSITION
# =========================================================
# Logic: % of minutes played in Year N-1 by players who returned in Year N.
print("Calculating Roster Composition metrics...")

player_stats_cols = ['playerID', 'year', 'minutes', 'points', 'win_pct', 'efficiency']
if 'efficiency' not in df.columns:
    df['efficiency'] = df['points'] + df['rebounds'] + df['assists'] + df['steals'] + df['blocks'] - df['turnovers']

    player_history = df[player_stats_cols].copy()
    player_history['next_year'] = player_history['year'] + 1

    player_history = player_history.rename(columns={
        'minutes': 'prev_min',
        'points': 'prev_pts',
        'win_pct': 'prev_player_win_pct',
        'efficiency': 'prev_eff',
        'year': 'prev_year'
    })

    df_roster = df[['year', 'tmID', 'playerID']].merge(
        player_history,
        left_on=['playerID', 'year'],
        right_on=['playerID', 'next_year'],
        how='left'
    )

    df_roster = df_roster.fillna(0)

    roster_features = df_roster.groupby(['year', 'tmID']).agg({
        'prev_min': 'sum',
        'prev_pts': 'sum',
        'prev_eff': 'mean',
        'prev_player_win_pct': 'mean'
    }).reset_index()

    roster_features['roster_continuity'] = roster_features['prev_min'] / 6800
    roster_features['roster_continuity'] = roster_features['roster_continuity'].clip(0, 1)

    print("Roster features calculated")

# =========================================================
# STEP 3.3: TEAM-LEVEL AGGREGATION & SOS
# =========================================================
# Aggregating metrics. We include 'rolling_win_pct' here (Momentum)
# Note: 'rolling_win_pct' is likely same for all players on team, so we take max or mean
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

# =========================================================
# STEP 3.4: MERGE AND FINALIZE FEATURES
# =========================================================

# Features to LAG (History determines future)
lag_cols = ['points','shooting_efficiency', 'def_impact', 'win_pct']
for col in lag_cols:
    team_df[f'team_lag_{col}'] = team_df.groupby('tmID')[col].shift(1)

model_df = team_df.merge(roster_features, on=['year', 'tmID'], how='left')

# Drop rows missing history (Year 1)
model_df = model_df.dropna(subset=[f'team_lag_{col}' for col in lag_cols])

features = [
    'team_lag_win_pct', 'team_lag_points', 'team_lag_def_impact',
    'roster_continuity', 'prev_pts', 'prev_eff', 'prev_player_win_pct',
    'prev_conf_strength'
]

target = 'win_pct'

print(f"\nFinal Dataset: {len(model_df)} rows. Features selected:")
print(features)
# =========================================================
# STEP 3.5: MODEL TRAINING
# =========================================================
train_df = model_df[model_df['year'] < 10]
test_df = model_df[model_df['year'] == 10].copy()

print(f"Training Ranking Model with {len(features)} features...")

rf = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42)
rf.fit(train_df[features], train_df[target])

test_df['raw_pred_win_pct'] = rf.predict(test_df[features])

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
# STEP 3.6: OUTPUTS & VISUALIZATION
# =========================================================
test_df['pred_rank'] = test_df.groupby('conference')['predicted_win_pct'].rank(ascending=False, method='dense')
test_df['actual_rank'] = test_df.groupby('conference')['win_pct'].rank(ascending=False, method='dense')

test_df['predicted_wins'] = (test_df['predicted_win_pct'] * 34).round(0).astype(int)

output = test_df[['conference', 'tmID', 'predicted_wins', 'win_pct', 'pred_rank', 'actual_rank']]
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
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='magma')
plt.title('Drivers of Team Success (New Features Added)')
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

plt.bar(x - width/2, bar_df["actual_rank"], width, label="Ranking Real")
plt.bar(x + width/2, bar_df["pred_rank"], width, label="Ranking Previsto")

plt.xticks(x, bar_df["tmID"], rotation=45, ha='right')
plt.xlabel("Times")
plt.ylabel("Ranking")
plt.title("Comparação de Ranking — Previsto vs Real (Ano 10)")
plt.gca().invert_yaxis()  # ranking 1 como melhor (topo)
plt.legend()

plt.tight_layout()
plt.savefig(f"{plot_dir}/season10_rank_barplot.png")
plt.close()

print(f" Saved rank comparison barplot to {plot_dir}/season10_rank_barplot.png")

print("\n Completed: Team Rankings with Roster Stability & Momentum.")