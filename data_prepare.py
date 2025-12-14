import pandas as pd
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "data_path": "data/",
    "season_11_path": "data/Season_11/",
    "cleaned_path": "data/cleaned/",
    "rolling_window": 3
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_datasets(path, s11_path):
    print("Loading datasets...")
    try:
        # 1. Carregar dados históricos (Season 1-10)
        datasets = {
            'awards_players' : pd.read_csv(f"{path}awards_players.csv"),
            'coaches' : pd.read_csv(f"{path}coaches.csv"),
            'players' : pd.read_csv(f"{path}players.csv"),
            'players_teams' : pd.read_csv(f"{path}players_teams.csv"),
            'teams' : pd.read_csv(f"{path}teams.csv")
        }

        # 2. Carregar e fundir dados da Season 11 (Teste) se existirem
        if os.path.exists(s11_path):
            print(f"Detected Season 11 data in {s11_path}. Merging...")
            
            # Carregar S11
            s11_coaches = pd.read_csv(f"{s11_path}coaches.csv")
            s11_teams = pd.read_csv(f"{s11_path}teams.csv")
            s11_players_teams = pd.read_csv(f"{s11_path}players_teams.csv")
            
            # Concatenar (Pandas vai criar NaN nas colunas de stats que faltam na S11)
            datasets['coaches'] = pd.concat([datasets['coaches'], s11_coaches], ignore_index=True)
            datasets['teams'] = pd.concat([datasets['teams'], s11_teams], ignore_index=True)
            datasets['players_teams'] = pd.concat([datasets['players_teams'], s11_players_teams], ignore_index=True)
            
            print(" ✓ Season 11 data merged successfully.")
            
        return datasets
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure CSV files exist.")
        return None

def clean_data(datasets):
    print("\n" + "="*80)
    print("STEP 2.1: DATA CLEANING")
    print("="*80)
    
    players = datasets['players']
    if "bioID" in players.columns:
        players.rename(columns={"bioID": "playerID"}, inplace=True)

    # Handling missing values
    print("\n1. Handling missing values...")
    
    # Preencher NaN nas stats da Season 11 com 0 (pois ainda não aconteceram)
    # Isto é crucial para não crashar os cálculos seguintes
    for df_name in ['coaches', 'teams']:
        for col in ['won', 'lost', 'post_wins', 'post_losses', 'o_pts', 'd_pts']:
            if col in datasets[df_name].columns:
                datasets[df_name][col] = datasets[df_name][col].fillna(0)

    # Players: impute missing heights/weights
    for col in ['height', 'weight']:
        if col in players.columns:
            median_val = players[col].median()
            players[col].fillna(median_val, inplace=True)

    awards = datasets['awards_players']
    awards.dropna(subset=['playerID'], inplace=True)

    # Remove duplicates
    print("\n2. Removing duplicates...")
    for name, df in datasets.items():
        if name in ['players', 'coaches', 'awards_players', 'teams']:
            df.drop_duplicates(inplace=True)

    # Correct inconsistencies
    print("Correcting inconsistencies...")
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

    print("Generating coach change and tenure features...")
    # Evitar divisão por zero na Season 11
    coaches['total_games'] = coaches['won'] + coaches['lost']
    coaches['win_pct'] = np.where(coaches['total_games'] > 0, coaches['won'] / coaches['total_games'], 0)
    
    coaches['post_win_pct'] = (coaches['post_wins'] / (coaches['post_wins'] + coaches['post_losses'])).fillna(0)

    coach_tenure = coaches.groupby('coachID')['year'].agg(['min', 'max'])
    coach_tenure['tenure_years'] = coach_tenure['max'] - coach_tenure['min'] + 1
    coaches = coaches.merge(coach_tenure[['tenure_years']], on='coachID', how='left')

    coaches = coaches.sort_values(['tmID', 'year'])
    coaches['is_new_coach'] = (coaches['coachID'] != coaches.groupby('tmID')['coachID'].shift(1)).astype(int)

    print("Generating player award-related features...")
    total_awards = awards.groupby('playerID').size().reset_index(name='total_career_awards')
    players = players.merge(total_awards, on='playerID', how='left').fillna({'total_career_awards': 0})

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

    full_dataset = full_dataset.merge(
    datasets['coaches'][['coachID','tmID','year','win_pct','post_win_pct','tenure_years','is_new_coach']], on=['tmID', 'year'], how='left')

    print("Integrating team metadata...")
    # Garantir que usamos as colunas corretas para o merge
    full_dataset = full_dataset.merge(datasets['teams'][['tmID', 'year', 'lgID', 'confID', 'franchID', 'name']], on=['tmID', 'year'], how='left')

    print("Integrating awards data...")
    award_counts = datasets['awards_players'].groupby(['playerID', 'year']).size().reset_index(name = 'award_count')
    full_dataset = full_dataset.merge(award_counts, on=['playerID', 'year'], how='left').fillna({'award_count' : 0})

    print("Creating rolling performance indicators...")
    full_dataset = full_dataset.sort_values(by=['tmID', 'year'])
    full_dataset['rolling_win_pct'] = full_dataset.groupby('tmID')['win_pct'].transform(lambda x: x.rolling(config['rolling_window'], min_periods=1).mean())

    return full_dataset

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(CONFIG['cleaned_path'], exist_ok=True)

    datasets = load_datasets(CONFIG['data_path'], CONFIG['season_11_path'])
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