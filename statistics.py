import numpy as np
from scipy.stats import wilcoxon
from tabulate import tabulate

def perform_ttest(scores, headers, alfa=.05):
    w_statistic = np.zeros((len(scores), len(scores)))
    p_value = np.zeros((len(scores), len(scores)))

    for i in range(len(scores)):
        for j in range(len(scores)):
            w_statistic[i, j], p_value[i, j] = wilcoxon(scores[i], scores[j], zero_method='zsplit')
    names_column = np.array([[header] for header in headers])
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(scores), len(scores)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((len(scores), len(scores)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = ", alfa, "):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)