import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('../data/german.data-numeric', sep='\t')
    file = data.to_csv('test.data', sep=',')
