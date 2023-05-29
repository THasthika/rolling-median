import bisect


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
        return self._data[0] if len(self._data) > 0 else None

    def get_max(self):
        return self._data[-1] if len(self._data) > 0 else None
    
    def pop_min(self):
        return self._data.pop(0)
    
    def pop_max(self):
        return self._data.pop()
    
    @property
    def number_count(self) -> int:
        return len(self._data)
    
    def __str__(self) -> str:
        return "{}".format(self._data)


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

    def remove_number(self, number):
        self.count -= 1
        self.root = self._remove_node(self.root, number)

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

    def balance(self):
        self.root = self._balance_tree(self.root)

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


class MedianTrackerV2:
    # set a limit L which would keep track of numbers
    # surrounding the median - L and median + L in a heap or bisect
    # keep the rest inside of two balanced binary tree
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
        
        d = self._get_left_size() - self._get_right_size()
        if d == 0:
            if n > self.mid:
                self._add_to_right_side(n)
            else:
                self._add_to_left_side(n)
        elif d > 0:
            if n > self.mid:
                self._add_to_right_side(n)
            else:
                self._pull_from_left_to_right()
                self._add_to_left_side(n)
        else:
            if n > self.mid:
                self._pull_from_right_to_left()
                self._add_to_right_side(n)
            else:
                self._add_to_left_side(n)
    
    def calculate_median(self):

        if self.mid is None:
            return 0

        if (self._get_left_size() + self._get_right_size() + 1) % 2 == 1:
            return self.mid

        if self._get_right_size() > self._get_left_size():
            x = self.right_store.get_min()
            return (x + self.mid) / 2
        else:
            x = self.left_store.get_max()
            return (x + self.mid) / 2
        
    def _get_left_size(self):
        return self.left_store.number_count + self.left_tree.number_count
    
    def _get_right_size(self):
        return self.right_store.number_count + self.right_tree.number_count
            
    def _add_to_right_side(self, n):
        overflow_large_number = self.right_store.add(n)
        if overflow_large_number is not None:
            self.right_tree.add_number(overflow_large_number)
    
    def _add_to_left_side(self, n):
        overflow_small_number = self.left_store.add(n)
        if overflow_small_number is not None:
            self.left_tree.add_number(overflow_small_number)

    def _pull_from_left_to_right(self):
        l_v = self.left_store.pop_max()
        if self.left_tree.number_count > 0:
            tree_max = self.left_tree.get_maximum()
            self.left_store.add(tree_max)
            self.left_tree.remove_number(tree_max)
        self._add_to_right_side(self.mid)
        self.mid = l_v
    
    def _pull_from_right_to_left(self):
        r_v = self.right_store.pop_min()
        if self.right_tree.number_count > 0:
            tree_min = self.right_tree.get_minimum()
            self.right_store.add(tree_min)
            self.right_tree.remove_number(tree_min)
        self._add_to_left_side(self.mid)
        self.mid = r_v

