import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna()
    # Simple encoding
    df = pd.get_dummies(df, drop_first=True)
    return df

if __name__ == "__main__":
    df = load_and_preprocess("Student_Performance.csv")
    df.to_csv("processed.csv", index=False)
