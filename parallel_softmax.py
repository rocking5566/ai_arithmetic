import math
import numpy as np


def naive_softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def parallel_softmax_helper(x):
    # return ex - max, max, sum(ex - max)
    if len(x) == 0:
        return np.array([]), 0, 0
    elif len(x) == 1:
        m = math.exp(x[0])
        ex = np.exp(x - m)
        return ex, m, ex[0]
    else:
        pivot = int(len(x) / 2)
        ex1, m1, sum1 = parallel_softmax_helper(x[0:pivot])
        ex2, m2, sum2 = parallel_softmax_helper(x[pivot:])
        m = max(m1, m2)
        update1 = math.exp(m1 - m)
        update2 = math.exp(m2 - m)
        ex = np.concatenate((update1 * ex1, update2 * ex2))
        sum = update1 * sum1 + update2 * sum2

    return ex, m, sum


def parallel_softmax(x):
    ex, _, sum = parallel_softmax_helper(x)
    return ex / sum


if __name__ == '__main__':
    x = np.random.randn(10).astype(np.float32)
    y_naive = naive_softmax(x)
    y_parallel = parallel_softmax(x)

    print("Naive: ", y_naive)
    print("parallel ", y_parallel)
