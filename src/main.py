import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/german-credit.data')
    print(df.head())
