from functools import partial
from multiprocessing import Pool


def add(a, b, c):
    return a + b + c


task_c = partial(add, 1, 2)


if __name__ == "__main__":
    with Pool(2) as p:
        result = p.map(task_c, range(22))
    print(result)
