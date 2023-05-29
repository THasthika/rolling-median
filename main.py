import heapq
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import bisect
import statistics
import numpy as np

from median_tracker import MedianTracker
from median_tracker_v2 import MedianTrackerV2
from median_tracker_v3 import MedianTrackerV3


# compute the median of numbers
def timeit(fn):

    def _inner_fn(*args, **kargs):
        start_time = time.time()
        r = fn(*args, **kargs)
        diff = time.time() - start_time
        return (diff, r)
    _inner_fn.__name__ = fn.__name__
    return _inner_fn


def compute_median(nums):
    if len(nums) == 0:
        return 0

    nums = sorted(nums)

    mid_index = len(nums) // 2
    if len(nums) % 2 == 1:
        return nums[mid_index]
    else:
        return (nums[mid_index-1] + nums[mid_index]) / 2


def compute_median_no_sort(nums):
    if len(nums) == 0:
        return 0

    mid_index = len(nums) // 2
    if len(nums) % 2 == 1:
        return nums[mid_index]
    else:
        return (nums[mid_index-1] + nums[mid_index]) / 2


@timeit
def median_tracker_solution(N=100):
    mt = MedianTracker()
    for _ in range(N):
        r_num = random.random()
        mt.add_number(r_num)
        mt.calculate_median()
    return mt.calculate_median()


@timeit
def median_tracker_v2_solution(N=100):
    mt = MedianTrackerV2(L=10)
    for _ in range(N):
        r_num = random.random()
        mt.add_number(r_num)
        mt.calculate_median()
    return mt.calculate_median()


@timeit
def median_tracker_v3_solution(N=100):
    mt = MedianTrackerV3()
    for _ in range(N):
        r_num = random.random()
        mt.add_number(r_num)
        mt.calculate_median()
    return mt.calculate_median()


@timeit
def naive_solution(N=100):
    data = []
    for _ in range(N):
        r_num = random.random()
        data.append(r_num)
        compute_median(data)
    return compute_median(data)


@timeit
def sorted_list_solution(N=100):
    data = []
    for _ in range(N):
        r_num = random.random()
        data.append(r_num)
        data.sort()
        compute_median_no_sort(data)
    return compute_median_no_sort(data)


@timeit
def statistics_solution(N=100):
    data = []
    for _ in range(N):
        rn = random.random()
        data.append(rn)
        statistics.median(data)
    return statistics.median(data)


@timeit
def bisect_solution(N=100):
    data = []

    for _ in range(N):
        r_num = random.random()
        bisect.insort(data, r_num)
        compute_median_no_sort(data)
    return compute_median_no_sort(data)


def analyse_runtimes(input_range, funcs, n_iters: int = 10):

    print("Input Range: Min: {} | Max: {}".format(min(input_range), max(input_range)))

    f_outs = [{} for _ in funcs]

    for test_size in tqdm(input_range):
        for (f_i, fn) in enumerate(funcs):

            f_outs[f_i][test_size] = []

            for i in range(n_iters):
                (t, v) = fn(test_size)
                f_outs[f_i][test_size].append(t)

            mean = statistics.mean(f_outs[f_i][test_size])
            std = statistics.stdev(f_outs[f_i][test_size])

            f_outs[f_i][test_size] = (mean, std)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for (i, _) in enumerate(funcs):
        means = []
        stds = []
        for x in input_range:
            mean = f_outs[i][x][0]
            std = f_outs[i][x][1]
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)

        # plt.plot(input_range, f_outs[i], color=colors[i], label='%s' % (funcs[i].__name__))
        plt.fill_between(input_range, means - stds, means + stds,
                         color=colors[i], alpha=0.2,
                         label='%s' % (funcs[i].__name__))
        
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()


analyse_runtimes(
    [int(i) for i in np.logspace(10, 15, num=10, base=2)],
    [sorted_list_solution, bisect_solution, median_tracker_solution],
    n_iters=10
)

# analyse_runtimes(
#     # [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 100000],
#     # [100, 1000, 5000, 10000, 20000, 50000, 100000],
#     [],
#     # [100, 200, 300, 500, 1000, 4000],
#     [median_tracker_solution, bisect_solution],
#     n_iters=10
# )


# def plot_median_value(N=100):
#     mt = MedianTracker()
#     x_values = []
#     y1_values = []
#     y2_values = []
#     r = []
#     for x in range(N):
#         x_values.append(x)
#         r_num = random.random()
#         r.append(r_num)
#         y2_values.append(compute_median(r))
#         mt.add_number(r_num)
#         y1_values.append(mt.calculate_median())
#     plt.plot(x_values, y1_values, label='Median Tracker')
#     plt.plot(x_values, y2_values, label='Naive')
#     plt.legend()
#     plt.show()


# plot_median_value(N=1000)
