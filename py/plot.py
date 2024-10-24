import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

class CsvMetadata:
    def __init__(self, filename: str):
        dim_result = re.search(r'dim_(\d+)', filename)
        nb_result = re.search(r'nb_(\d+)', filename)
        m_result = re.search(r'm_(\d+)', filename)
        ef_result = re.search(r'ef_(\d+)', filename)

        assert dim_result is not None
        assert nb_result is not None
        assert m_result is not None
        assert ef_result is not None

        self.dim = int(dim_result.group(1))
        self.nb = int(nb_result.group(1))
        self.m = int(m_result.group(1))
        self.ef = int(ef_result.group(1))

        print("Param dim:", self.dim)
        print("Param nb: ", self.nb)
        print("Param m:  ", self.m)
        print("Param ef: ", self.ef)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(__file__, usage = "visualize graphs")
    argparser.add_argument('file', type=str, help="path to csv file")
    argparser.add_argument('--warmup', type=int, default=0, help="number of runs to treat as 'warmup' and discarded")
    argparser.add_argument('--plot-top-k', type=int, default=[], nargs="+", help="space deliminated list of top k's to plot")
    args = argparser.parse_args()

    filename = Path(args.file)
    warmup = args.warmup
    assert filename.exists()
    assert warmup >= 0

    metadata = CsvMetadata(filename.name)

    df = pd.read_csv(filename)

    # plot 
    
    top_k = df[df.columns[0]]
    rows = []
    title_k = []
    for target_top_k in args.plot_top_k:
        row_indices = df[df[df.columns[0]].eq(target_top_k)].index.tolist()
        assert len(row_indices) == 0 or len(row_indices) == 1, "multiple rows with same top k ??!?"

        if len(row_indices) == 0:
            print(f"target top_k = {target_top_k} was not found in {filename}")
            continue 

        print(f"plotting top_k = {target_top_k} excluding {warmup} round(s) (warmup)")

        # because row_indices has 1 element
        # start on 2 + warmup and exclude last column
        rows.append(df.iloc[row_indices[0], 1+warmup:-1])
        title_k.append(target_top_k)
    
    num_plots = len(rows)
    plt.figure(figsize=(8, 12))

    for i, (data, k) in enumerate(zip(rows, title_k), start=1):
        plt.subplot(num_plots, 1, i) # (row, col, idx)
        plt.plot(data)
        plt.xlabel('run')
        plt.ylabel('time (us)')
        plt.title(f'Performance For Top K {k} warmup {warmup}')

    plt.tight_layout()
    plt.savefig('output.png')

    # 1st column is top k,
    # start on 2 + warmup and exclude last column
    mean = df.iloc[:, 1+warmup:-1].mean(axis=1)
    p95 =  df.iloc[:, 1+warmup:-1].apply(lambda row: np.percentile(row, 95), axis=1)
    p99 =  df.iloc[:, 1+warmup:-1].apply(lambda row: np.percentile(row, 99), axis=1)
    var =  df.iloc[:, 1+warmup:-1].apply(lambda row: np.var(row), axis=1)

    # recall is last column
    recall = df.iloc[:, -1].apply(lambda row: row * 100)

    df['mean (us)'] = mean
    df['p95 (us)'] = p95
    df['p99 (us)'] = p99
    df['var'] = var
    df['recall (%)'] = recall
    df['top k'] = df['k']

    results = df[['top k', 'recall (%)', 'var', 'mean (us)', 'p95 (us)', 'p99 (us)']]
    print(results)
