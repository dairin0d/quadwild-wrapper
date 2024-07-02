import math


copysign = math.copysign


def is_valid(obj):
    if not obj: return False
    try:
        id_data = obj.id_data
        return True
    except (ReferenceError, AttributeError):
        return False


class PrimitiveLock:
    "Primary use of such lock is to prevent infinite recursion"
    __slots__ = ("count",)
    def __init__(self):
        self.count = 0
    def __bool__(self):
        return bool(self.count)
    def __enter__(self):
        self.count += 1
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.count -= 1
