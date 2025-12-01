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
print(f"Loaded dataset with shape: {df.shape}")

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
    'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
    'shooting_efficiency', 'turnover_ratio', 'def_impact', 'win_pct'
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
# STEP 3.5: MODEL TRAINING & PREDICTION
# =========================================================
train_df = model_df[model_df['year'] < 10]
test_df = model_df[model_df['year'] == 10].copy()

input_features = [f'prev_{col}' for col in feature_cols]
target = 'win_pct'

print(f"Training Ranking Model on {len(train_df)} seasons...")

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(train_df[input_features], train_df[target])

print("Predicting Year 10...")
test_df['predicted_win_pct'] = rf.predict(test_df[input_features])

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

# Prediction vs Actual Scatter Plot
plt.figure(figsize=(8,8))
sns.scatterplot(x=test_df['win_pct'], y=test_df['predicted_win_pct'], hue=test_df['conference'], s=120, palette='viridis')
plt.plot([0,1], [0,1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Win % (Year 10)')
plt.ylabel('Predicted Win % (Year 10)')
plt.title(f'Year 10 Prediction Accuracy (MAE: {mae:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/season10_pred_vs_actual.png")
plt.close()
print(f"Saved accuracy plot to {plot_dir}/season10_pred_vs_actual.png")

# Feature Importance
importances = pd.DataFrame({
    'Feature': input_features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importance: What drives next years success?')
plt.tight_layout()
plt.savefig(f"{plot_dir}/ranking_feature_importance.png")
plt.close()
print(f"Saved feature importance plot to {plot_dir}/ranking_feature_importance.png")

print("\nCompleted: Season 10 prediction + real comparison + plots generated.")