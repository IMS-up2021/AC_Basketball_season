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
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# Ensure we handle the conference column name correctly
if 'lgID_x' in df.columns:
    df.rename(columns={'lgID_x': 'conference'}, inplace=True)
elif 'lgID_y' in df.columns:
    df.rename(columns={'lgID_y': 'conference'}, inplace=True)
elif 'lgID' in df.columns:
    df.rename(columns={'lgID': 'conference'}, inplace=True)

# =========================================================
# STEP 3.2: CALCULATE ROSTER STABILITY (New Feature)
# =========================================================
# Logic: % of minutes played in Year N-1 by players who returned in Year N.
print("Calculating Roster Stability (Returning Minutes %)...")

# 1. Get total minutes per player per year/team
player_minutes = df.groupby(['year', 'tmID', 'playerID'])['minutes'].sum().reset_index()

# 2. Identify the roster (set of players) for each team/year
rosters = df.groupby(['year', 'tmID'])['playerID'].apply(set).reset_index()

stability_data = []

# Loop through years to compare Current Year (N) vs Previous Year (N-1)
for i, row in rosters.iterrows():
    year = row['year']
    tmID = row['tmID']
    current_players = row['playerID']
    
    if year == 1:
        stability_data.append({'year': year, 'tmID': tmID, 'roster_stability': 0.0})
        continue
        
    # Get stats for this team from the PREVIOUS year
    prev_year_stats = player_minutes[
        (player_minutes['year'] == year - 1) & 
        (player_minutes['tmID'] == tmID)
    ]
    
    if prev_year_stats.empty:
        stability_data.append({'year': year, 'tmID': tmID, 'roster_stability': 0.0})
        continue
        
    total_prev_minutes = prev_year_stats['minutes'].sum()
    
    if total_prev_minutes == 0:
        val = 0.0
    else:
        # Sum minutes of players from Year N-1 who appear in the Year N roster set
        returning_mins = prev_year_stats[prev_year_stats['playerID'].isin(current_players)]['minutes'].sum()
        val = returning_mins / total_prev_minutes
        
    stability_data.append({'year': year, 'tmID': tmID, 'roster_stability': val})

stability_df = pd.DataFrame(stability_data)
print("Roster Stability calculation complete.")

# =========================================================
# STEP 3.3: TEAM-LEVEL AGGREGATION
# =========================================================
# Aggregating metrics. We include 'rolling_win_pct' here (Momentum)
# Note: 'rolling_win_pct' is likely same for all players on team, so we take max or mean
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
    'win_pct': 'max',
    'rolling_win_pct': 'max' # Momentum Feature
})

# Merge Stability Feature back in
team_df = team_df.merge(stability_df, on=['year', 'tmID'], how='left').fillna(0)

# =========================================================
# STEP 3.4: FEATURE ENGINEERING & LAGS
# =========================================================
team_df['shooting_efficiency'] = team_df['fgMade'] / team_df['fgAttempted'].replace(0,1)
team_df['turnover_ratio'] = team_df['assists'] / team_df['turnovers'].replace(0,1)
team_df['def_impact'] = team_df['steals'] + team_df['blocks']
team_df['foul_discipline'] = team_df['PF']

feature_cols = [
    'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
    'shooting_efficiency', 'turnover_ratio', 'def_impact', 'win_pct'
]

# Features to LAG (History determines future)
lag_cols = [
    'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
    'shooting_efficiency', 'turnover_ratio', 'def_impact', 
    'win_pct', 'rolling_win_pct' # Momentum lagged
]

print("\nGenerating Lag Features (Year N-1 -> Year N)...")
team_df = team_df.sort_values(['tmID', 'year'])

for col in lag_cols:
    team_df[f'prev_{col}'] = team_df.groupby('tmID')[col].shift(1)

# We do NOT lag 'roster_stability' because it describes the transition *into* the current year.
# (e.g., Year 10 stability is calculated based on who returned from Year 9)

# Drop rows missing history (Year 1)
model_df = team_df.dropna(subset=[f'prev_{col}' for col in lag_cols])

print(f"Dataset prepared for modeling: {len(model_df)} rows")

# =========================================================
# STEP 3.5: MODEL TRAINING & PREDICTION
# =========================================================
train_df = model_df[model_df['year'] < 10]
test_df = model_df[model_df['year'] == 10].copy()

# Feature List
input_features = [f'prev_{col}' for col in lag_cols] + ['roster_stability']
target = 'win_pct'

print(f"Training Ranking Model with {len(input_features)} features...")

rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(train_df[input_features], train_df[target])

print("Predicting Year 10...")
test_df['predicted_win_pct'] = rf.predict(test_df[input_features])

# Metrics
mae = mean_absolute_error(test_df['win_pct'], test_df['predicted_win_pct'])
r2 = r2_score(test_df['win_pct'], test_df['predicted_win_pct'])

print(f"\n--- Season 10 Prediction Results ---")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# =========================================================
# STEP 3.6: RANKING GENERATION & SAVING
# =========================================================
test_df['pred_rank'] = test_df.groupby('conference')['predicted_win_pct'].rank(ascending=False, method='dense')
test_df['actual_rank'] = test_df.groupby('conference')['win_pct'].rank(ascending=False, method='dense')

test_df['predicted_wins'] = (test_df['predicted_win_pct'] * 34).round(0).astype(int)
test_df['actual_wins'] = (test_df['win_pct'] * 34).round(0).astype(int)

output = test_df[['conference', 'tmID', 'predicted_wins', 'actual_wins', 'predicted_win_pct', 'win_pct', 'pred_rank', 'actual_rank']]
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
    'Feature': input_features,
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