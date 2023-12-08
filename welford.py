import numpy as np


def ParallelWelford(x):
    if len(x) == 0:
        return (0, 0, 0)
    elif len(x) == 1:
        return (1, x[0], 0)
    else:
        pivot = int(len(x) / 2)
        count_A, mean_A, M_A = ParallelWelford(x[0:pivot])
        count_B, mean_B, M_B = ParallelWelford(x[pivot:])
        count = count_A + count_B
        count_B_over_count = count_B / count
        delta = mean_B - mean_A
        mean = mean_A + delta * count_B_over_count
        M = M_A + M_B + delta**2 * count_A * count_B_over_count
        return (count, mean, M)


def welford_update(count, mean, M, currValue):
    count += 1
    delta = currValue - mean
    mean += delta / count
    delta2 = currValue - mean
    M += delta * delta2
    return (count, mean, M)


def naive_update(sum, sum_square, currValue):
    sum = sum + currValue
    sum_square = sum_square + currValue * currValue
    return (sum, sum_square)


if __name__ == '__main__':
    x_arr = np.random.randn(10).astype(np.float32)
    welford_mean = 0
    welford_M = 0
    welford_count = 0
    for i in range(len(x_arr)):
        welford_count, welford_mean, welford_M = welford_update(
            welford_count, welford_mean, welford_M, x_arr[i])
    print("Welford mean: ", welford_mean)
    print("Welford var: ", welford_M / welford_count)

    pwelford_count, pwelford_mean, pwelford_M = ParallelWelford(x_arr)
    print("parallel Welford mean: ", pwelford_mean)
    print("parallel Welford var: ", pwelford_M / pwelford_count)

    naive_sum = 0
    naive_sum_square = 0
    for i in range(len(x_arr)):
        naive_sum, naive_sum_square = naive_update(
            naive_sum, naive_sum_square, x_arr[i])
    naive_mean = naive_sum / len(x_arr)
    naive_var = naive_sum_square / len(x_arr) - naive_mean*naive_mean
    print("Naive mean: ", naive_mean)
    print("Naive var: ", naive_var)
