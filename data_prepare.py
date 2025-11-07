import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2.1: DATA CLEANING
# ============================================================================

print("="*80)
print("PHASE 2: DATA PREPARATION")
print("="*80)

print("\n" + "="*80)
print("STEP 2.1: DATA CLEANING")
print("="*80)

# Load all datasets (assuming data folder from Phase 1)
DATA_PATH = "data/"
CLEAN_PATH = f"{DATA_PATH}cleaned/"

# Create cleaned data directory if it doesn't exist
import os
os.makedirs(CLEAN_PATH, exist_ok=True)

awards_players = pd.read_csv(f"{DATA_PATH}awards_players.csv")
coaches = pd.read_csv(f"{DATA_PATH}coaches.csv")
players = pd.read_csv(f"{DATA_PATH}players.csv")
players_teams = pd.read_csv(f"{DATA_PATH}players_teams.csv")
series_post = pd.read_csv(f"{DATA_PATH}series_post.csv")
teams = pd.read_csv(f"{DATA_PATH}teams.csv")
teams_post = pd.read_csv(f"{DATA_PATH}teams_post.csv")

# ----------------------------------------------------------------------------
# ✅ Standardize ID column names
# ----------------------------------------------------------------------------
# Rename bioID → playerID for consistency
if "bioID" in players.columns:
    players.rename(columns={"bioID": "playerID"}, inplace=True)

# ============================================================================
# 1️⃣ Handle missing values
# ============================================================================
print("\nHandling missing values...")

# Players: impute missing heights/weights with median
players['height'].fillna(players['height'].median(), inplace=True)
players['weight'].fillna(players['weight'].median(), inplace=True)

# Coaches: replace missing postseason stats with 0
for col in ['post_wins', 'post_losses']:
    if col in coaches.columns:
        coaches[col].fillna(0, inplace=True)

# Awards: drop rows with missing playerIDs
awards_players.dropna(subset=['playerID'], inplace=True)

# ============================================================================
# 2️⃣ Remove duplicates
# ============================================================================
print("Removing duplicates...")

for df_name, df in [
    ('players', players),
    ('coaches', coaches),
    ('awards_players', awards_players),
    ('teams', teams),
]:
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  {df_name:15s}: removed {before - len(df)} duplicate rows")

# ============================================================================
# 3️⃣ Correct data inconsistencies
# ============================================================================
print("Correcting inconsistencies...")

# Ensure team and player IDs are uppercase
for df in [players, coaches, teams, players_teams]:
    if 'tmID' in df.columns:
        df['tmID'] = df['tmID'].astype(str).str.strip().str.upper()
    if 'playerID' in df.columns:
        df['playerID'] = df['playerID'].astype(str).str.strip().str.upper()

# Standardize league codes
for df in [awards_players, coaches, teams]:
    if 'lgID' in df.columns:
        df['lgID'] = df['lgID'].astype(str).str.upper()

# ============================================================================
# STEP 2.2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 2.2: FEATURE ENGINEERING")
print("="*80)

# ----------------------------------------------------------------------------
# 🏀 TEAM RANKING FEATURES
# ----------------------------------------------------------------------------
print("\nGenerating team-level performance features...")

# Overall win percentage
coaches['win_pct'] = coaches['won'] / (coaches['won'] + coaches['lost'])

# Postseason win percentage
coaches['post_win_pct'] = np.where(
    (coaches['post_wins'] + coaches['post_losses']) > 0,
    coaches['post_wins'] / (coaches['post_wins'] + coaches['post_losses']),
    np.nan
)

# Basic team stats per year
team_summary = coaches.groupby(['tmID', 'year']).agg(
    games_coached=('year', 'count'),
    avg_win_pct=('win_pct', 'mean'),
    total_wins=('won', 'sum'),
    total_losses=('lost', 'sum')
).reset_index()

# ----------------------------------------------------------------------------
# 🧠 COACH CHANGE FEATURES
# ----------------------------------------------------------------------------
print("Generating coach change and tenure features...")

# Tenure per coach
coach_tenure = coaches.groupby('coachID')['year'].agg(['min', 'max', 'count']).reset_index()
coach_tenure.rename(columns={'min': 'first_year', 'max': 'last_year', 'count': 'seasons'}, inplace=True)
coach_tenure['tenure_years'] = coach_tenure['last_year'] - coach_tenure['first_year'] + 1

# Merge tenure info back
coaches = coaches.merge(coach_tenure[['coachID', 'tenure_years']], on='coachID', how='left')

# Simple performance trend (Δ win%)
coaches['performance_trend'] = coaches.groupby('coachID')['win_pct'].diff()

# ----------------------------------------------------------------------------
# 🏆 PLAYER AWARD FEATURES
# ----------------------------------------------------------------------------
print("Generating player award-related features...")

# Award count per player per season
award_counts = awards_players.groupby(['playerID', 'year']).size().reset_index(name='award_count')

# Total awards per player (career)
total_awards = awards_players.groupby('playerID').size().reset_index(name='total_awards')

# Merge into player dataset
players = players.merge(total_awards, on='playerID', how='left').fillna({'total_awards': 0})

# ============================================================================
# STEP 2.3: DATA INTEGRATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2.3: DATA INTEGRATION")
print("="*80)

print("\nMerging player and team data...")
player_team_season = players_teams.merge(players, on='playerID', how='left')

print("Adding coach data...")
team_with_coach = player_team_season.merge(
    coaches[['tmID', 'year', 'coachID', 'win_pct']],
    on=['tmID', 'year'], how='left'
)

print("Integrating team metadata...")
full_dataset = team_with_coach.merge(
    teams[['tmID', 'lgID', 'franchID']], on='tmID', how='left'
)

print("Integrating awards data...")
full_dataset = full_dataset.merge(award_counts, on=['playerID', 'year'], how='left')
full_dataset['award_count'].fillna(0, inplace=True)

# Rolling win% (momentum indicator)
print("Creating rolling performance indicators...")
full_dataset = full_dataset.sort_values(by=['tmID', 'year'])
full_dataset['rolling_win_pct'] = full_dataset.groupby('tmID')['win_pct'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# ============================================================================
# SAVE CLEANED AND ENRICHED DATASETS
# ============================================================================
print("\nSaving prepared datasets...")

full_dataset.to_csv(f"{CLEAN_PATH}full_dataset_prepared.csv", index=False)
coaches.to_csv(f"{CLEAN_PATH}coaches_cleaned.csv", index=False)
players.to_csv(f"{CLEAN_PATH}players_cleaned.csv", index=False)

print("\n✓ Phase 2 complete: cleaned and integrated dataset saved successfully.")
print("="*80)


