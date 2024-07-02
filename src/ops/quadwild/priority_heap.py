import itertools
import heapq

# Adapted from https://docs.python.org/3/library/heapq.html
class PriorityHeap:
    def __init__(self):
        self.entries = []
        self.item_map = {}
        self.REMOVED = [] # just some unique object
        self.counter = itertools.count()
        self.count = 0

    def __len__(self):
        return self.count

    def __bool__(self):
        return self.count > 0

    def __contains__(self, item):
        return item in self.item_map

    def add(self, item, priority=0):
        "Add a new item or update the priority of an existing item"
        if item in self.item_map: self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.item_map[item] = entry
        heapq.heappush(self.entries, entry)
        self.count += 1

    def remove(self, item):
        "Mark an existing item as REMOVED. Raise KeyError if not found."
        entry = self.item_map.pop(item)
        entry[-1] = self.REMOVED
        self.count -= 1

    def pop(self):
        "Remove and return the lowest priority item. Raise KeyError if empty."
        while self.entries:
            priority, count, item = heapq.heappop(self.entries)
            if item is not self.REMOVED:
                del self.item_map[item]
                self.count -= 1
                return item
        raise KeyError('pop from an empty priority queue')
