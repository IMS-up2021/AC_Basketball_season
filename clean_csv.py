import pandas as pd
import numpy as np

# LOAD DATASETS

players        = pd.read_csv("data/players.csv")
players_teams  = pd.read_csv("data/players_teams.csv")
coaches        = pd.read_csv("data/coaches.csv")
teams          = pd.read_csv("data/teams.csv")
teams_post     = pd.read_csv("data/teams_post_uploaded.csv")
series_post    = pd.read_csv("data/series_post_uploaded.csv")
awards_players = pd.read_csv("data/awards_players_uploaded.csv")

# STANDARDIZE COLUMNS

def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

players, players_teams, coaches, teams, teams_post, series_post, awards_players = [
    normalize_cols(df)
    for df in [players, players_teams, coaches, teams, teams_post, series_post, awards_players]
]

# ensure 'year' is numeric
for df in [players_teams, coaches, teams, teams_post, series_post]:
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

# HANDLE MISSING VALUES

def fill_numeric(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df

def fill_strings(df):
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].fillna("Unknown")
    return df

for df in [players, players_teams, coaches, teams, teams_post]:
    df = fill_numeric(df)
    df = fill_strings(df)

# MERGE TEAM + COACH DATA

# Each (tmID, year) should have one main coach
coach_main = (
    coaches.sort_values(["tmid", "year", "stint"])
    .groupby(["tmid", "year"])
    .first()
    .reset_index()
)

team_season_df = pd.merge(
    teams, coach_main, on=["tmid", "year"], how="left", suffixes=("", "_coach")
)

# MERGE PLAYER + TEAM DATA

player_season_df = pd.merge(
    players_teams,
    players[["playerid", "birthyear", "college", "position"]],
    on="playerid",
    how="left",
)

player_season_df = pd.merge(
    player_season_df,
    team_season_df[["tmid", "year", "won", "lost", "coachid"]],
    on=["tmid", "year"],
    how="left",
)

# CREATE MASTER TABLE (PLAYER + TEAM + COACH)

master_df = player_season_df.merge(
    coach_main[["tmid", "year", "coachid", "firstname", "lastname"]],
    on=["tmid", "year", "coachid"],
    how="left"
)

# SAVE CLEANED FILES

team_season_df.to_csv("/mnt/data/clean_team_season.csv", index=False)
player_season_df.to_csv("/mnt/data/clean_player_season.csv", index=False)
master_df.to_csv("/mnt/data/clean_master.csv", index=False)

