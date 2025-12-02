import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14,10)

print("="*80)
print("PHASE 5: INDIVIDUAL AWARDS PREDICTION (TEST SEASON)")
print("="*80)

# =========================================================
# 1. LOAD DATA
# =========================================================
clean_path = "data/cleaned/full_dataset_prepared.csv"
awards_path = "data/awards_players.csv"
output_dir = "results/predictions"
plot_dir = "results/plots"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

if not os.path.exists(clean_path):
    raise FileNotFoundError(f"File not found: {clean_path}")

df = pd.read_csv(clean_path)
awards_df = pd.read_csv(awards_path)

df['playerID'] = df['playerID'].astype(str).str.upper()
awards_df['playerID'] = awards_df['playerID'].astype(str).str.upper()

print(f"Data Loaded. PLayers/Teams: {df.shape}, Awards: {awards_df.shape}")

# =========================================================
# 2. FEATURE ENGINEERING: TEAM CONTEXT
# =========================================================
print("Calculating Team Context (Ranks)...")

if 'lgID' in df.columns:
    df.rename(columns={'lgID': 'conference'}, inplace=True)
elif 'lgID_x' in df.columns:
    df.rename(columns={'lgID': 'conference'}, inplace=True)
elif 'lgID_y' in df.columns:
    df.rename(columns={'lgID': 'conference'}, inplace=True)

if 'conference' not in df.columns:
    df['conference'] = 'League'

team_stats = df.groupby(['year', 'tmID', 'conference'])['win_pct'].mean().reset_index()
team_stats['conf_rank'] = team_stats.groupby(['year', 'conference'])['win_pct'].rank(ascending=False, method='min')

df = df.merge(team_stats[['year', 'tmID', 'conf_rank']], on=['year', 'tmID'], how='left')

# =========================================================
# 3. DATA AGGREGATION & ADVANCED METRICS
# =========================================================
metric_cols = ['GP', 'minutes', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'PF', 'fgMade', 'fgAttempted', 'ftMade', 'ftAttempted']

player_stats = df.groupby(['playerID', 'year'], as_index=False).agg({
    **{col: 'sum' for col in metric_cols},
    'win_pct': 'mean',
    'conf_rank': 'mean',
    'experience_years': 'max',
    'GS': 'sum'
})

player_stats['ts_pct'] = player_stats['points'] / (2 * (player_stats['fgAttempted'] + 0.44 * player_stats['ftAttempted']))
player_stats['ts_pct'] = player_stats['ts_pct'].fillna(0)

player_stats['efficiency'] = (
    player_stats['points'] + player_stats['rebounds'] + player_stats['assists'] + player_stats['steals'] + player_stats['blocks'] - player_stats['turnovers'] - (player_stats['fgAttempted'] - player_stats['fgMade']) - (player_stats['ftAttempted'] - player_stats['ftMade'])
)

player_stats['def_score'] = player_stats['steals'] * 2 + player_stats['blocks'] * 2 + player_stats['rebounds']

for col in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'efficiency']:
    player_stats[f'{col}_pg'] = player_stats[col] / player_stats['GP'].replace(0,1)

player_stats['starter_ratio'] = player_stats['GS'] / player_stats['GP'].replace(0,1)

z_cols = ['points_pg', 'efficiency_pg', 'rebounds_pg', 'assists_pg', 'steals_pg', 'blocks_pg', 'def_score']
print("Calculating Z-Scores...(Relative to League Average)")

for col in z_cols:
    player_stats[f'z_{col}'] = player_stats.groupby('year')[col].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)

player_stats.sort_values(['playerID', 'year'], inplace=True)
player_stats['prev_efficiency_pg'] = player_stats.groupby('playerID')['efficiency_pg'].shift(1).fillna(0)
player_stats['efficiency_delta'] = player_stats['efficiency_pg'] - player_stats['prev_efficiency_pg']

# =========================================================
# 4. TARGET MAPPING & NARRATIVE FEATURES
# =========================================================
target_awards = [
    'Most Valuable Player',
    'Rookie of the Year',
    'Defensive Player of the Year',
    'Sixth Woman of the Year',
    'Most Improved Player'
]

print("\n Mapping targets for awards...")

for award in target_awards:
    winners = awards_df[awards_df['award'] == award][['playerID', 'year']]
    winners['is_winner'] = 1

    col_name = f"target_{award.replace(' ', '_')}"
    player_stats = player_stats.merge(winners, on=['playerID', 'year'], how='left')
    player_stats.rename(columns={'is_winner': col_name}, inplace=True)
    player_stats[col_name] = player_stats[col_name].fillna(0)

    prev_col_name = f"prev_winner_{award.replace(' ', '_')}"
    player_stats[prev_col_name] = player_stats.groupby('playerID')[col_name].shift(1).fillna(0)

# =========================================================
# 5. MODEL TRAINING
# =========================================================
TEST_YEAR = 10
train_data = player_stats[player_stats['year'] < TEST_YEAR].fillna(0)
test_data = player_stats[player_stats['year'] == TEST_YEAR].fillna(0)

print(f"\n Training Data: {train_data['year'].nunique()} seasons ({len(train_data)} rows)")
print(f"Test Data: Season {TEST_YEAR} ({len(test_data)} rows)")

prediction_summary = []
race_data = {}

def predict_award(award_name, features, filter_func=None):
    target_col = f"target_{award_name.replace(' ', '_')}"

    if filter_func:
        train_set = train_data[filter_func(train_data)].copy()
        test_set = test_data[filter_func(test_data)].copy()
    else:
        train_set = train_data.copy()
        test_set = test_data.copy()
    
    if train_set[target_col].sum() == 0:
        return None

    X_train = train_set[features]
    y_train = train_set[target_col]
    X_test = test_set[features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')    
    clf.fit(X_train_scaled, y_train)

    if len(clf.classes_) > 1:
        probs = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = np.zeros(len(X_test))

    test_set['probability'] = probs

    top_candidates = test_set.sort_values('probability', ascending=False).head(5)
    race_data[award_name] = top_candidates[['playerID', 'probability', 'conf_rank',] if 'conf_rank' in features else ['playerID', 'probability']]

    winner_row = top_candidates.iloc[0]
    actual_row = test_set[test_set[target_col] == 1]
    actual_winner = actual_row.iloc[0]['playerID'] if not actual_row.empty else "None"

    prediction_summary.append({
        'Award': award_name,
        'Predicted Winner': winner_row['playerID'],
        'Prob': f"{winner_row['probability']:.2f}",
        'Actual Winner': actual_winner,
        'Correct': "YES" if winner_row['playerID'] == actual_winner else "NO"
    })

    race_data[award_name] = top_candidates[['playerID', 'probability']]

    print(f"Computed: {award_name}")

    return clf

# =========================================================
# 6. EXECUTE PREDICTIONS
# =========================================================
mvp_features = ['z_points_pg', 'z_efficiency_pg', 'ts_pct', 'conf_rank', 'win_pct', 'prev_winner_Most_Valuable_Player']
mvp_model = predict_award('Most Valuable Player', mvp_features)

roy_features = ['z_points_pg', 'z_rebounds_pg', 'z_efficiency_pg', 'minutes', 'GP']
predict_award('Rookie of the Year', roy_features, lambda x: x['experience_years'] == 0)

dpoy_features = ['z_steals_pg', 'z_blocks_pg', 'z_rebounds_pg', 'z_def_score']
predict_award('Defensive Player of the Year', dpoy_features)

six_features = ['z_points_pg', 'z_efficiency_pg', 'ts_pct', 'minutes']
predict_award('Sixth Woman of the Year', six_features, lambda x: x['starter_ratio'] < 0.5)

mip_features = ['efficiency_delta', 'z_points_pg', 'minutes']
predict_award('Most Improved Player', mip_features, lambda x: x['experience_years'] > 0)

# =========================================================
# 7. SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(prediction_summary)
csv_path = f"{output_dir}/season_10_award_predictions.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")
print(results_df)

# =========================================================
# 8. VISUALIZATIONS
# =========================================================
if mvp_model:
    plt.figure(figsize=(10,6))
    importances = pd.DataFrame({
        'Feature': mvp_features,
        'Importance': mvp_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=importances, palette='magma', hue='Feature', legend=False)
    plt.title('Drivers of MVP Prediction (Team Success vs Stats)')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mvp_factors.png")

num_awards = len(race_data)
fig, axes = plt.subplots(2,3, figsize=(18,10))
axes = axes.flatten()

for i, (award, data) in enumerate(race_data.items()):
    ax = axes[i]
    sns.barplot(x='probability', y='playerID', data=data, ax=ax, palette='viridis', hue='playerID', legend=False)
    ax.set_title(award, fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("")
    ax.set_xlim(0,1)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
viz_path = f"{plot_dir}/award_races_Season_11.png"
plt.savefig(viz_path)
print(f"Visualization saved to: {viz_path}")