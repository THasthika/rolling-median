import heapq
import bisect


class MedianTrackerV3:

    def __init__(self):

        self.r_set = []
        self.l_set = []
        self.mid = None

    def add_number(self, n):

        if self.mid is None:
            self.mid = n
            return
        
        # check if the sides are balanced
        # if a number is greater than mid -> take the number and put
        #   it to the right side and take the least of the right side
        # if a number is less or equal to mid -> take the number and put
        #   it ot the left side and take the largest of the left side
        # replace the mid with the largest or the smallest values
        #   from either side
        d = len(self.l_set) - len(self.r_set)
        if d == 0:
            if n > self.mid:
                bisect.insort(self.r_set, n)
            else:
                bisect.insort(self.l_set, n)
        elif d > 0:
            if n > self.mid:
                bisect.insort(self.r_set, n)
            else:
                largest_l = self.l_set.pop()
                bisect.insort(self.r_set, self.mid)
                self.mid = largest_l
                bisect.insort(self.l_set, n)
        else:
            if n > self.mid:
                smallest_r = self.r_set.pop(0)
                bisect.insort(self.l_set, self.mid)
                self.mid = smallest_r
                bisect.insort(self.l_set, n)
            else:
                bisect.insort(self.l_set, n)

    def calculate_median(self):

        if self.mid is None:
            return 0

        if (len(self.r_set) + len(self.l_set) + 1) % 2 == 1:
            return self.mid

        if len(self.r_set) > len(self.l_set):
            x = self.r_set[0]
            return (x + self.mid) / 2
        else:
            x = self.l_set[-1]
            return (x + self.mid) / 2
        