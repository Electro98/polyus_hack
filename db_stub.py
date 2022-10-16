from collections import deque


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LikeDB(metaclass=Singleton):
    def __init__(self) -> None:
        self._deque = deque(maxlen=60)

    def put(self, item):
        self._deque.append(item)

    def get_all(self):
        return list(self._deque)
