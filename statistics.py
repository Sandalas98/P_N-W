import numpy as np
from scipy.stats import wilcoxon, ranksums
from tabulate import tabulate

from scipy.stats import rankdata

def perform_wilcoxon_test(scores, headers, alfa=.05):
    print(scores)

    scores = scores.T

    ranks = []
    for ms in scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)

    clfs_number = len(scores[0])
    w_statistic = np.zeros((clfs_number, clfs_number))
    p_value = np.zeros((clfs_number, clfs_number))

    for i in range(clfs_number):
        for j in range(clfs_number):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    names_column = np.array([[header] for header in headers])
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("w-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((clfs_number, clfs_number))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((clfs_number, clfs_number))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = ", alfa, "):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)