import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "data_path": "data/",
    "cleaned_path": "data/cleaned/",
    "rolling_window":3
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_datasets(path):
    print("Loading datasets...")
    try:
        datasets = {
            'awards_players' : pd.read_csv(f"{path}awards_players.csv"),
            'coaches' : pd.read_csv(f"{path}coaches.csv"),
            'players' : pd.read_csv(f"{path}players.csv"),
            'players_teams' : pd.read_csv(f"{path}players_teams.csv"),
            'teams' : pd.read_csv(f"{path}teams.csv")
        }
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure all CSV files are in the '{path}' directory")
        return None

def clean_data(datasets):
    print("\n" + "="*80)
    print("STEP 2.1: DATA CLEANING")
    print("="*80)
    # ----------------------------------------------------------------------------
    # ✅ Standardize ID column names
    # ----------------------------------------------------------------------------
    # Rename bioID → playerID for consistency
    players = datasets['players']
    if "bioID" in players.columns:
        players.rename(columns={"bioID": "playerID"}, inplace=True)

    # ============================================================================
    # 1️⃣ Handle missing values
    # ============================================================================
    print("\n1. Handling missing values...")

    # Players: impute missing heights/weights with median
    for col in ['height', 'weight']:
        median_val = players[col].median()
        fill_count = players[col].isnull().sum()
        players[col].fillna(median_val, inplace=True)
        print(f" - Imputed {fill_count} missing '{col}' values with median ({median_val:.1f})")

    # Coaches: replace missing postseason stats with 0
    coaches = datasets['coaches']
    coaches[['post_wins', 'post_losses']] = coaches[['post_wins', 'post_losses']].fillna(0)
    print(" - Filled missing coach postseason stats with 0")


    # Awards: drop rows with missing playerIDs
    awards = datasets['awards_players']
    awards.dropna(subset=['playerID'], inplace=True)

    # ============================================================================
    # 2️⃣ Remove duplicates
    # ============================================================================
    print("\n2. Removing duplicates...")

    for name, df in datasets.items():
        if name in ['players', 'coaches', 'awards_players', 'teams']:
            before = len(df)
            df.drop_duplicates(inplace=True)
            print(f"  {name:15s}: removed {before - len(df)} duplicate rows")

    # ============================================================================
    # 3️⃣ Correct data inconsistencies
    # ============================================================================
    print("Correcting inconsistencies...")

    # Ensure team and player IDs are uppercase
    for df in datasets.values():
        for col in ['tmID', 'playerID', 'lgID']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
    return datasets

# ============================================================================
# STEP 2.2: FEATURE ENGINEERING
# ============================================================================
def engineer_features(datasets):
    print("\n" + "="*80)
    print("STEP 2.2: FEATURE ENGINEERING")
    print("="*80)

    coaches = datasets['coaches']
    players = datasets['players']
    awards = datasets['awards_players']
    players_teams = datasets['players_teams']

    # ----------------------------------------------------------------------------
    # 🧠 COACH CHANGE FEATURES
    # ----------------------------------------------------------------------------
    print("Generating coach change and tenure features...")

    # Overall win percentage
    coaches['win_pct'] = coaches['won'] / (coaches['won'] + coaches['lost'])

    # Postseason win percentage
    coaches['post_win_pct'] = (coaches['post_wins'] / (coaches['post_wins'] + coaches['post_losses'])).fillna(0)

    # Tenure per coach
    coach_tenure = coaches.groupby('coachID')['year'].agg(['min', 'max'])
    coach_tenure['tenure_years'] = coach_tenure['max'] - coach_tenure['min'] + 1

    # Merge tenure info back
    coaches = coaches.merge(coach_tenure[['tenure_years']], on='coachID', how='left')

    # Coach change flag for a team-year
    coaches = coaches.sort_values(['tmID', 'year'])
    coaches['is_new_coach'] = (coaches['coachID'] != coaches.groupby('tmID')['coachID'].shift(1)).astype(int)

    # ----------------------------------------------------------------------------
    # 🏆 PLAYER AWARD FEATURES
    # ----------------------------------------------------------------------------
    print("Generating player award-related features...")
    total_awards = awards.groupby('playerID').size().reset_index(name='total_career_awards')
    
    # Merge into player dataset
    players = players.merge(total_awards, on='playerID', how='left').fillna({'total_career_awards': 0})

    # Player experience in  years
    player_experience = players_teams.groupby('playerID')['year'].min().reset_index()
    player_experience.rename(columns={'year':'first_season'}, inplace=True)
    players_teams = players_teams.merge(player_experience, on='playerID', how='left')
    players_teams['experience_years'] = players_teams['year'] - players_teams['first_season']

    datasets['coaches'] = coaches
    datasets['players'] = players
    datasets['players_teams'] = players_teams
    return datasets

# ============================================================================
# STEP 2.3: DATA INTEGRATION
# ============================================================================

def integrate_data(datasets, config):
    print("\n" + "="*80)
    print("STEP 2.3: DATA INTEGRATION")
    print("="*80)

    print("\nMerging player and team data...")
    full_dataset = datasets['players_teams'].merge(datasets['players'], on='playerID', how='left')

    print("Adding coach data...")
    full_dataset = full_dataset.merge(datasets['coaches'], on=['tmID', 'year', 'coachID'], how='left')

    print("Integrating team metadata...")
    full_dataset = full_dataset.merge(datasets['teams'][['tmID', 'lgID', 'franchID', 'name']], on='tmID', how='left')

    print("Integrating awards data...")
    award_counts = datasets['awards_players'].groupby(['playerID', 'year']).size().reset_index(name = 'award_count')
    full_dataset = full_dataset.merge(award_counts, on=['playerID', 'year'], how='left').fillna({'award_count' : 0})

    # Rolling win% (momentum indicator)
    print("Creating rolling performance indicators...")
    full_dataset = full_dataset.sort_values(by=['tmID', 'year'])
    full_dataset['rolling_win_pct'] = full_dataset.groupby('tmID')['win_pct'].transform(lambda x: x.rolling(config['rolling_window'], min_periods=1).mean())

    return full_dataset

# ============================================================================
# SAVE CLEANED AND ENRICHED DATASETS
# ============================================================================

def main():
    # Ensure the cleaned directory exists
    os.makedirs(CONFIG['cleaned_path'], exist_ok=True)

    datasets = load_datasets(CONFIG['data_path'])
    if datasets is None:
        return
    
    cleaned_datasets = clean_data(datasets)
    featured_datasets = engineer_features(cleaned_datasets)
    full_dataset = integrate_data(featured_datasets, CONFIG)


    print("\nSaving prepared datasets...")

    full_dataset.to_csv(f"{CONFIG['cleaned_path']}full_dataset_prepared.csv", index=False)
    featured_datasets['coaches'].to_csv(f"{CONFIG['cleaned_path']}coaches_cleaned.csv", index=False)
    featured_datasets['players'].to_csv(f"{CONFIG['cleaned_path']}players_cleaned.csv", index=False)

    print("\n✓ Phase 2 complete: cleaned and integrated dataset saved successfully.")
    print("="*80)

if __name__ == '__main__':
    main()