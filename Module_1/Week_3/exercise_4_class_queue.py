class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__data = []

    def is_empty(self):
        return len(self.__data) == 0

    def is_full(self):
        return len(self.__data) == self.__capacity

    def enqueue(self, value):
        if self.is_full():
            print("Out of capacity!")
        else:
            return self.__data.append(value)

    def dequeue(self):
        if self.is_empty():
            print("Do nothing!")
            return None
        else:
            return self.__data.pop(0)

    def front(self):
        if self.is_empty():
            print("Do nothing!")
            return None
        else:
            return self.__data[0]


if __name__ == "__main__":
    queue1 = MyQueue(capacity=5)

    queue1.enqueue(1)
    queue1.enqueue(2)

    print(queue1.is_full())

    print(queue1.front())

    print(queue1.dequeue())

    print(queue1.front())

    print(queue1.dequeue())

    print(queue1.is_empty())
