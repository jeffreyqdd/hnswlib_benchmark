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
        k_result = re.search(r'K_(\d+)', filename)

        assert dim_result is not None
        assert nb_result is not None
        assert m_result is not None
        assert ef_result is not None
        assert k_result is not None

        self.dim = int(dim_result.group(1))
        self.nb = int(nb_result.group(1))
        self.m = int(m_result.group(1))
        self.ef = int(ef_result.group(1))
        self.k = int(k_result.group(1))

        print("Param dim:", self.dim)
        print("Param nb: ", self.nb)
        print("Param m:  ", self.m)
        print("Param ef: ", self.ef)
        print("value K: ", self.k)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(__file__, usage="visualize graphs")
    argparser.add_argument('file', type=str, help="path to csv file")
    argparser.add_argument('--warmup', type=int, default=0,
                           help="number of runs to treat as 'warmup' and discarded")
    # argparser.add_argument('--plot-top-k', type=int, default=[], nargs="+", help="space deliminated list of top k's to plot")
    args = argparser.parse_args()

    filename = Path(args.file)
    warmup = args.warmup
    assert filename.exists()
    assert warmup >= 0

    metadata = CsvMetadata(filename.name)

    print("reading csv")
    df = pd.read_csv(filename)

    # plot
    number_different_single_queries = len(df.index)

    plt.figure(figsize=(8, 3 * number_different_single_queries))
    for i in range(1, number_different_single_queries + 1):
        print(f"plotting graph {i} of {number_different_single_queries}")
        ax = plt.subplot(number_different_single_queries, 1, i)
        ax.plot(df.iloc[i-1, 1+warmup:])  # +1 to skip first column which is id
        ax.set_title(f"{metadata.k} queries on vector id {
                     i} top k {metadata.k}")
        ax.set_ylim(0, ymax=np.max(df.iloc[i-1, 1+warmup:]) + 200)
        ax.set_xlabel("iteration number")
        ax.set_ylabel("search time (us)")

    plt.tight_layout()
    plt.savefig('tmp.pdf')

    mean = df.iloc[:, 1+warmup:-1].mean(axis=1)
    p95 = df.iloc[:, 1+warmup:-1].apply(lambda r: np.percentile(r, 95), axis=1)
    p99 = df.iloc[:, 1+warmup:-1].apply(lambda r: np.percentile(r, 99), axis=1)
    var = df.iloc[:, 1+warmup:-1].apply(lambda r: np.var(r), axis=1)

    df['mean (us)'] = mean
    df['p95 (us)'] = p95
    df['p99 (us)'] = p99
    df['var'] = var

    results = df[['id', 'var', 'mean (us)', 'p95 (us)', 'p99 (us)']]
    print(results)
