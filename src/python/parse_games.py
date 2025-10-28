import pandas as pd
from pathlib import Path

def parse_games(input_csv: str = "datasets/hex_games.csv") -> None:
    df = pd.read_csv(Path(__file__).resolve().parents[2] / input_csv)
    print(df.head())
    print(f"Loaded {len(df)} games ({len(df.columns)-1} cells each).")
    print(df['winner'].value_counts())

if __name__ == "__main__":
    parse_games()