import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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

        self.dim = int(dim_result.group(1))
        self.nb = int(nb_result.group(1))
        self.m = int(m_result.group(1))
        self.ef = int(ef_result.group(1))
        self.k = int(k_result.group(1)) if k_result else 100

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
    argparser.add_argument('--run-idx', type=int, default=[],
                           nargs="+", help="space delim list of run ids to plot")
    args = argparser.parse_args()

    filename = Path(args.file)
    warmup = args.warmup
    assert filename.exists()
    assert warmup >= 0

    metadata = CsvMetadata(filename.name)

    print("reading csv")
    df = pd.read_csv(filename)

    # plot
    title = []
    rows = []
    for run_id in args.run_idx:
        row_indices = df[df[df.columns[0]].eq(run_id)].index.tolist()
        assert len(row_indices) == 0 or len(
            row_indices) == 1, "multiple rows with same id"

        if len(row_indices) == 0:
            print(f"target run_id = {run_id} was not found in {filename}")
            continue

        print(f"plotting run_id = {run_id} excluding {
              warmup} round(s) (warmup)")

        # because row_indices has 1 element
        # start on 2 + warmup and exclude last column
        t = f"Nb={metadata.nb} Topk={metadata.k} Run id {run_id}"
        rows.append(df.iloc[row_indices[0], 1+warmup:-1])
        title.append(t)

    number_different_single_queries = len(rows)

    plt.figure(figsize=(8, 3 * number_different_single_queries))
    for i in range(1, number_different_single_queries + 1):
        print(f"plotting graph {i} of {number_different_single_queries}")
        ax = plt.subplot(number_different_single_queries, 1, i)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # +1 to skip first column which is id
        ax.plot(rows[i - 1])
        ax.set_title(title[i - 1])
        ax.set_ylim(0, ymax=np.max(df.iloc[i-1, 1+warmup:-1]) + 200)
        ax.set_xlabel("iteration number")
        ax.set_ylabel("search time (us)")

    plt.tight_layout()
    plt.savefig('tmp.pdf')

    mean = df.iloc[:, 1+warmup:-1].mean(axis=1)
    p95 = df.iloc[:, 1+warmup:-1].apply(lambda r: np.percentile(r, 95), axis=1)
    p99 = df.iloc[:, 1+warmup:-1].apply(lambda r: np.percentile(r, 99), axis=1)
    var = df.iloc[:, 1+warmup:-1].apply(lambda r: np.var(r), axis=1)

    df['recall'] = df.iloc[:, -1:]
    df['mean (us)'] = mean
    df['p95 (us)'] = p95
    df['p99 (us)'] = p99
    df['var'] = var

    results = df[['id', 'var', 'mean (us)', 'p95 (us)', 'p99 (us)', 'recall']]
    print(results)
