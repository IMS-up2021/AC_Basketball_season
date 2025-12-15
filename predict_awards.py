import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRanker

# =========================================================
# CONFIGURATION
# =========================================================
PREDICTION_YEAR = 11
LAST_TRAIN_YEAR = 10

# Visual Setup (Seaborn Style)
sns.set_style("whitegrid")
output_dir = "results/predictions"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print(f"========================================================")
print(f"AWARD PREDICTIONS (FINAL FIX) - SEASON {PREDICTION_YEAR}")
print(f"========================================================")

# =========================================================
# 1. LOAD DATA
# =========================================================
data_path = "data/cleaned/full_dataset_prepared.csv"
rankings_path = f"results/predictions/season_{PREDICTION_YEAR}_rankings.csv"
awards_path = "data/awards_players.csv"
players_path = "data/players.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError("Run data_prepare.py first!")

df = pd.read_csv(data_path)
awards_df = pd.read_csv(awards_path)
players_df = pd.read_csv(players_path)

# Normalize Conference Names
if 'confID' not in df.columns:
    col_map = {'lgID_x': 'conference', 'lgID': 'conference', 'confID': 'conference'}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
else:
    df.rename(columns={'confID': 'conference'}, inplace=True)
df['conference'] = df['conference'].fillna('Unknown')

# Normalize IDs
df['playerID'] = df['playerID'].astype(str).str.upper()
awards_df['playerID'] = awards_df['playerID'].astype(str).str.upper()
if 'bioID' in players_df.columns:
    players_df.rename(columns={'bioID': 'playerID'}, inplace=True)
players_df['playerID'] = players_df['playerID'].astype(str).str.upper()

# Full Name Creation
if 'firstName' in players_df.columns and 'lastName' in players_df.columns:
    players_df['fullName'] = players_df['firstName'] + " " + players_df['lastName']
else:
    players_df['fullName'] = players_df['playerID']

df = df.merge(players_df[['playerID', 'fullName']], on='playerID', how='left')
df['fullName'] = df['fullName'].fillna(df['playerID'])

# Load Season 11 Rankings
pred_win_map = {}
pred_rank_map = {}
if os.path.exists(rankings_path):
    print("--> Loading predicted team standings...")
    rankings_df = pd.read_csv(rankings_path)
    rankings_df['tmID'] = rankings_df['tmID'].astype(str)
    pred_win_map = dict(zip(rankings_df.tmID, rankings_df.predicted_win_pct))
    
    if 'conference' not in rankings_df.columns and 'lgID' in rankings_df.columns:
        rankings_df.rename(columns={'lgID': 'conference'}, inplace=True)
        
    rankings_df['conf_rank'] = rankings_df.groupby('conference')['predicted_wins'].rank(ascending=False, method='min')
    pred_rank_map = dict(zip(rankings_df.tmID, rankings_df.conf_rank))

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

metrics = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']
if 'efficiency' not in df.columns:
    df['efficiency'] = df['points'].fillna(0) + df['rebounds'].fillna(0) + df['assists'].fillna(0) + df['steals'].fillna(0) + df['blocks'].fillna(0) - df['turnovers'].fillna(0)

for m in metrics + ['efficiency']:
    df[f'{m}_pg'] = df[m] / df['GP'].replace(0, 1).fillna(1)

if 'GS' in df.columns:
    df['starter_ratio'] = df['GS'].fillna(0) / df['GP'].replace(0, 1).fillna(1)
else:
    df['starter_ratio'] = 1.0

target_awards = ['Most Valuable Player', 'Rookie of the Year', 'Defensive Player of the Year', 'Sixth Woman of the Year', 'Most Improved Player']

for award in target_awards:
    winners = awards_df[awards_df['award'] == award][['playerID', 'year']]
    winners['is_winner'] = 1
    col_name = f"target_{award.replace(' ', '_')}"
    df = df.merge(winners, on=['playerID', 'year'], how='left')
    df.rename(columns={'is_winner': col_name}, inplace=True)
    df[col_name] = df[col_name].fillna(0)

# =========================================================
# 3. STATS PROJECTION FOR SEASON 11
# =========================================================
print("--> Projecting stats for Season 11...")

s11_roster = df[df['year'] == PREDICTION_YEAR][['playerID', 'fullName', 'tmID', 'year', 'experience_years', 'GP', 'GS']].copy()
s10_stats = df[df['year'] == LAST_TRAIN_YEAR][['playerID', 'points_pg', 'efficiency_pg', 'rebounds_pg', 'assists_pg', 'steals_pg', 'blocks_pg', 'starter_ratio']].copy()
s10_stats.columns = ['playerID'] + [f"{c}_prev" for c in s10_stats.columns if c != 'playerID']

s11_proj = s11_roster.merge(s10_stats, on='playerID', how='left')

s9_stats = df[df['year'] == (LAST_TRAIN_YEAR - 1)][['playerID', 'efficiency_pg']].copy()
s9_stats.rename(columns={'efficiency_pg': 'efficiency_pg_2yrs_ago'}, inplace=True)
s11_proj = s11_proj.merge(s9_stats, on='playerID', how='left')

# Inject Win Pct BEFORE calculating stats (to adjust rookies based on opportunity)
s11_proj['win_pct'] = s11_proj['tmID'].map(pred_win_map).fillna(0.5)
s11_proj['conf_rank'] = s11_proj['tmID'].map(pred_rank_map).fillna(3)

cols_to_project = ['points_pg', 'efficiency_pg', 'rebounds_pg', 'assists_pg', 'steals_pg', 'blocks_pg']
league_avg_s10 = df[df['year'] == LAST_TRAIN_YEAR][cols_to_project].mean()

for col in cols_to_project:
    prev_col = f"{col}_prev"
    s11_proj[col] = s11_proj[prev_col]
    
    # --- ROOKIES: OPPORTUNITY-BASED PROJECTION ---
    is_rookie_missing = (s11_proj['experience_years'] == 0) & (s11_proj[col].isna())
    
    if not s11_proj.loc[is_rookie_missing].empty:
        # If team wins little (low win_pct), rookie plays more (Factor > 1.0)
        # If team wins a lot (high win_pct), rookie plays less (Factor < 1.0)
        opp_factor = 1.6 - (s11_proj.loc[is_rookie_missing, 'win_pct'] * 1.0)
        
        # Random noise (0.8 to 1.2)
        noise = np.random.uniform(0.8, 1.2, size=is_rookie_missing.sum())
        
        # Assign value: (League Avg * 0.6) * Opportunity * Noise
        s11_proj.loc[is_rookie_missing, col] = (league_avg_s10[col] * 0.6) * opp_factor * noise
    
    # VETERANS WITHOUT DATA
    is_vet_missing = (s11_proj['experience_years'] > 0) & (s11_proj[col].isna())
    if not s11_proj.loc[is_vet_missing].empty:
        s11_proj.loc[is_vet_missing, col] = league_avg_s10[col] * 0.9

s11_proj['starter_ratio'] = s11_proj['starter_ratio_prev'].fillna(0.2)
team_max_eff = s11_proj.groupby('tmID')['efficiency_pg'].transform('max')
s11_proj['is_best_player'] = (s11_proj['efficiency_pg'] == team_max_eff).astype(int)
s11_proj['improvement_trend'] = s11_proj['efficiency_pg'] - s11_proj['efficiency_pg_2yrs_ago'].fillna(s11_proj['efficiency_pg'] * 0.8)

# =========================================================
# 4. PREPARE TRAINING DATA
# =========================================================
print("--> Preparing training data...")
train_df = df[df['year'] <= LAST_TRAIN_YEAR].copy()

train_team_stats = train_df.groupby(['year', 'tmID', 'conference'])['win_pct'].mean().reset_index()
train_team_stats['conf_rank_calc'] = train_team_stats.groupby(['year', 'conference'])['win_pct'].rank(ascending=False, method='min')
train_df = train_df.merge(train_team_stats[['year', 'tmID', 'conference', 'conf_rank_calc']], on=['year', 'tmID', 'conference'], how='left')
train_df['conf_rank'] = train_df['conf_rank_calc'].fillna(4)
train_df.drop(columns=['conf_rank_calc'], inplace=True)

if 'efficiency_pg' not in train_df.columns:
     train_df['efficiency'] = (train_df['points'].fillna(0) + train_df['rebounds'].fillna(0) + 
                               train_df['assists'].fillna(0) + train_df['steals'].fillna(0) - train_df['turnovers'].fillna(0))
     train_df['efficiency_pg'] = train_df['efficiency'] / train_df['GP'].replace(0, 1).fillna(1)

train_max_eff = train_df.groupby(['year', 'tmID'])['efficiency_pg'].transform('max')
train_df['is_best_player'] = (train_df['efficiency_pg'] == train_max_eff).astype(int).fillna(0)
train_df['improvement_trend'] = train_df.groupby('playerID')['efficiency_pg'].diff().fillna(0)

# =========================================================
# 5. PREDICTION FUNCTION WITH TIE-BREAKER
# =========================================================

def get_probs(scores):
    e_x = np.exp(scores - np.max(scores))
    return e_x / e_x.sum()

race_results = {}

def run_race(award_name, features, filters=None):
    target = f"target_{award_name.replace(' ', '_')}"
    train_set = train_df.copy()
    test_set = s11_proj.copy()
    
    if filters:
        train_set = train_set[filters(train_set)]
        test_set = test_set[filters(test_set)]
    
    X_train = train_set[features].fillna(0)
    y_train = train_set[target]
    
    if len(X_train) == 0 or y_train.sum() == 0: return None

    model = XGBRanker(objective='rank:pairwise', n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    groups = [g for g in train_set.groupby('year').size().to_list() if g > 0]
    if not groups: return None

    model.fit(X_train, y_train, group=groups, verbose=False)
    
    if len(test_set) == 0: return None
        
    X_test = test_set[features].fillna(0)
    scores = model.predict(X_test)
    test_set['raw_score'] = scores
    
    # --- TIE-BREAKER LOGIC ---
    # If model scores are too similar (causing 20% splits),
    # we add a fraction of projected stats to the score to force separation.
    if award_name == 'Rookie of the Year':
        # Use Efficiency and Points as tie-breaker
        tie_breaker = (test_set['efficiency_pg'] * 0.1) + (test_set['points_pg'] * 0.1)
        test_set['raw_score'] += tie_breaker
    
    elif award_name == 'Most Improved Player':
        # Use Improvement Trend as tie-breaker
        test_set['raw_score'] += test_set['improvement_trend'] * 0.2

    # Normalize and Select Top 5
    results = test_set.sort_values('raw_score', ascending=False).head(5).copy()
    results['probability'] = get_probs(results['raw_score'].values)
    
    race_results[award_name] = results[['fullName', 'tmID', 'probability']]
    print(f"Computed {award_name} -> Leader: {results.iloc[0]['fullName']} ({results.iloc[0]['tmID']})")

# Run Models
run_race('Most Valuable Player', ['efficiency_pg', 'points_pg', 'win_pct', 'conf_rank', 'is_best_player'])
run_race('Rookie of the Year', ['points_pg', 'rebounds_pg', 'efficiency_pg'], lambda x: x['experience_years'] == 0)
run_race('Defensive Player of the Year', ['steals_pg', 'blocks_pg', 'rebounds_pg', 'win_pct', 'efficiency_pg'])
run_race('Sixth Woman of the Year', ['points_pg', 'efficiency_pg', 'win_pct', 'starter_ratio'], lambda x: x['starter_ratio'] < 0.6)
run_race('Most Improved Player', ['improvement_trend', 'points_pg', 'efficiency_pg'], lambda x: x['experience_years'] > 1)

# =========================================================
# 6. EXPORT AND VISUALIZE
# =========================================================
if race_results:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    palettes = ['viridis', 'mako', 'crest', 'rocket', 'flare']

    for i, (award, df_res) in enumerate(race_results.items()):
        ax = axes[i]
        sns.barplot(
            data=df_res, x='probability', y='fullName', hue='fullName',
            ax=ax, palette=palettes[i % len(palettes)],
            edgecolor='black', alpha=0.9, legend=False
        )
        
        ax.set_title(award, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.02, p.get_y() + p.get_height()/2, 
                    f'{width:.1%}', va='center', fontsize=9, fontweight='bold')

    if len(race_results) < 6:
        for j in range(len(race_results), 6): fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f'WNBA Season {PREDICTION_YEAR} Award Predictions (Projected)', fontsize=18)

    plot_path = f"{plot_dir}/season_{PREDICTION_YEAR}_awards_subplots.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n--> Visualizations saved to: {plot_path}")

    # CSV Export
    results_for_csv = []
    for award, df_res in race_results.items():
        results_for_csv.append(df_res.assign(Award=award))
        
    all_preds = pd.concat(results_for_csv)
    csv_path = f"{output_dir}/season_{PREDICTION_YEAR}_awards_detailed.csv"
    all_preds.to_csv(csv_path, index=False)
    print(f"--> Detailed predictions saved to: {csv_path}")
else:
    print("No predictions generated.")