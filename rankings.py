# ==============================================
# Phase 3: Regular Season Team Rankings by Conference
# ==============================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

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
# STEP 3.5: MODEL TRAINING (USANDO ANOS HISTÓRICOS)
# =========================================================

X = model_df[[f'prev_{col}' for col in feature_cols]]
y = model_df['win_pct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance: MAE={mae:.4f}, RMSE={r2:.4f}")

# =========================================================
# STEP 3.6: MODEL PERFORMANCE
# =========================================================
plot_dir = "results/plots"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(8,8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')

plt.plot([0,1], [0,1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Win %')
plt.ylabel('Predicted Win %')
plt.title(f'Model Accuracy: Actual vs Predicted (MAE: {mae:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/ranking_model_performance.png")
print(f"Saved model performance plot to {plot_dir}/ranking_model_performance.png")
plt.close()

# =========================================================
# STEP 3.7: FEATURE IMPORTANCE
# =========================================================
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, hue='Feature', legend=False, palette='viridis')
plt.title('Feature Importance: Drivers of Next Season Success')
plt.tight_layout()
plt.savefig(f"{plot_dir}/ranking_feature_importance.png")
print(f"Saved feature importance plot to {plot_dir}/ranking_feature_importance.png")
plt.close()

# =========================================================
# STEP 3.8: PREDICT SEASON 11
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
future_rankings = latest_stats.sort_values(['conference', 'predicted_wins'], ascending=[True, False])

plt.figure(figsize=(12,6))
sns.barplot(x='tmID', y='predicted_wins', hue='conference', data=future_rankings, dodge=False)
plt.title('Predicted Wins for Season 11 by Team')
plt.ylabel('Predicted Wins (approx 34 games)')
plt.xlabel('Team')
plt.legend(title='Conference')
plt.tight_layout()
plt.savefig(f"{plot_dir}/season_11_forecast.png")
print(f"Saved Season 11 forecast plot to {plot_dir}/season_11_forecast.png")
plt.close()

# =========================================================
# STEP 3.7: SAVE RESULTS
# =========================================================
output_cols = ['conference', 'tmID', 'predicted_wins', 'predicted_win_pct', 'rank']
future_rankings[output_cols].to_csv("results/predicted_season11_rankings.csv", index=False)

print(f"\n Saved Season 11 Rankings saved to results/predicted_season11_rankings.csv")
# =========================================================
# STEP 3.8: DISPLAY RESULTS
# =========================================================
for conf in future_rankings['conference'].dropna().unique():
    print(f"\n🏆 Conference: {conf}")
    print(future_rankings[future_rankings['conference'] == conf][['rank', 'tmID', 'predicted_wins']].to_string(index=False))
print("\n✅ Ranking generation complete.")
