import heapq
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import bisect
import statistics
import numpy as np



class FixedSortedList:

    def __init__(self, max_size: int, pop_min: bool = True):
        self.max_size = max_size
        self._data = []
        self._pop_min = pop_min

    def add(self, value) -> int | None:
        bisect.insort(self._data, value)
        if len(self._data) > self.max_size:
            if self._pop_min:
                return self._data.pop(0)
            else:
                return self._data.pop()
        return None

    def get_min(self):
        return self._data[0] if self._data else None

    def get_max(self):
        return self._data[-1] if self._data else None
    
    @property
    def number_count(self) -> int:
        return len(self._data)


class BinaryNode:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

    def __str__(self) -> str:
        return 'BinaryNode (%.2f, %d)' % (self.value, self.height)


class BalancedBinarySearchTree:

    def __init__(self):
        self.root = None
        self.count = 0

    @property
    def number_count(self) -> int:
        return self.count
        
    def add_number(self, number: int):
        self.count += 1
        self.root = self._insert_node(self.root, number)
    
    def _insert_node(self, node: BinaryNode | None, number: int):
        if node is None:
            return BinaryNode(number)
        
        if number < node.value:
            node.left = self._insert_node(node.left, number)
        else:
            node.right = self._insert_node(node.right, number)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance_factor = self._get_balance_factor(node)

        if balance_factor > 1:
            if number < node.left.value:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        
        if balance_factor < -1:
            if number > node.right.value:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node
    
    def remove_number(self, number):
        self.count -= 1
        self.root = self._remove_node(self.root, number)

    def _remove_node(self, root, number):
        if root is None:
            return root
        
        if number < root.value:
            root.left = self._remove_node(root.left, number)
        elif number > root.value:
            root.right = self._remove_node(root.right, number)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left

            # Node has both left and right child
            successor = self._find_min_node(root.right)
            root.value = successor.value
            root.right = self._remove_node(root.right, successor.value)

        root.height = 1 + max(self._get_height(root.left), self._get_height(root.right))
        balance_factor = self._get_balance_factor(root)

        if balance_factor > 1:
            if self._get_balance_factor(root.left) < 0:
                root.left = self._rotate_left(root.left)
            return self._rotate_right(root)
        
        if balance_factor < -1:
            if self._get_balance_factor(root.right) > 0:
                root.right = self._rotate_right(root.right)
            return self._rotate_left(root)

        return root

    def _find_min_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height
    
    def _get_balance_factor(self, node):
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_right(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y
    
    def balance(self):
        self.root = self._balance_tree(self.root)

    def _balance_tree(self, root):
        if root is None:
            return None

        root.left = self._balance_tree(root.left)
        root.right = self._balance_tree(root.right)

        root.height = 1 + max(self._get_height(root.left), self._get_height(root.right))
        balance_factor = self._get_balance_factor(root)

        if balance_factor > 1:
            if self._get_balance_factor(root.left) < 0:
                root.left = self._rotate_left(root.left)
            return self._rotate_right(root)
        
        if balance_factor < -1:
            if self._get_balance_factor(root.right) > 0:
                root.right = self._rotate_right(root.right)
            return self._rotate_left(root)

        return root

    def get_minimum(self):
        if self.root is None:
            return None
        current = self.root
        while current.left is not None:
            current = current.left
        return current.value

    def get_maximum(self):
        if self.root is None:
            return None
        current = self.root
        while current.right is not None:
            current = current.right
        return current.value


class MedianTrackerV2:

    ## set a limit L which would keep track of numbers surrounding the median - L and median + L in a heap or bisect
    ## keep the rest inside of two balanced binary tree

    def __init__(self, L: int):
        self._L = L

        self.mid = None

        self.left_store = FixedSortedList(self._L)
        self.right_store = FixedSortedList(self._L, pop_min=False)

        self.left_tree = BalancedBinarySearchTree()
        self.right_tree = BalancedBinarySearchTree()

    def add_number(self, n):

        if self.mid is None:
            self.mid = n
            return

        if self.mid <= n:
            # put number into the fixed list
            self.left_store.add(n)




class MedianTracker:

    def __init__(self):

        self.r_set = []
        self.l_set = []
        self.mid = None

    def add_number(self, n):

        if self.mid is None:
            self.mid = n
            return
        
        # check if the sides are balanced
        # if a number is greater than mid -> take the number and put it to the right side and take the least of the right side
        # if a number is less or equal to mid -> take the number and put it ot the left side and take the largest of the left side
        # replace the mid with the largest or the smallest values from either side

        if n > self.mid:
            if len(self.r_set) == 0:
                heapq.heappush(self.r_set, n)
            else:
                smallest_r = heapq.heappushpop(self.r_set, n)
                heapq.heappush(self.l_set, -self.mid)
                self.mid = smallest_r
        else:
            if len(self.l_set) == 0:
                heapq.heappush(self.l_set, -n)
            else:
                largest_l = heapq.heappushpop(self.l_set, -n)
                heapq.heappush(self.r_set, self.mid)
                self.mid = -largest_l

        d = len(self.l_set) - len(self.r_set)

        if d > 1:
            while d > 1:
                largest_l = heapq.heappop(self.l_set)
                heapq.heappush(self.r_set, self.mid)
                self.mid = -largest_l
                d -= 1
        elif d < -1:
            while d < -1:
                smallest_r = heapq.heappop(self.r_set)
                heapq.heappush(self.l_set, -self.mid)
                self.mid = smallest_r
                d += 1

    def calculate_median(self):

        if self.mid is None:
            return 0

        if (len(self.r_set) + len(self.l_set) + 1) % 2 == 1:
            return self.mid

        if len(self.r_set) > len(self.l_set):
            x = self.r_set[0]
            return (x + self.mid) / 2
        else:
            x = -self.l_set[0]
            return (x + self.mid) / 2
        


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
        # mt.calculate_median()
    return mt.calculate_median()


@timeit
def naive_infinite_rolling_median(N=100):
    data = []
    for _ in range(N):
        r_num = random.random()
        data.append(r_num)
        compute_median(data)
    return compute_median(data)


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

    colors = ['r', 'g', 'b', 'y', 'o', 'i']

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
    plt.legend()
    plt.show()


# analyse_runtimes(
#     # [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 100000],
#     # [100, 1000, 5000, 10000, 20000, 50000, 100000],
#     [100, 500, 1000, 2000, 3000, 4000, 5000],
#     [median_tracker_solution, naive_infinite_rolling_median],
#     n_iters=10
# )
