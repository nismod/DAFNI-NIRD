from functools import partial
from multiprocessing import Pool


class Radiation:
    def __init__(self, numbers):
        self.numbers = numbers

    @staticmethod
    def calculate_sum(parameter, number):
        return parameter + number

    def generate(self, parameter):
        calculate_sum_length = partial(self.calculate_sum, parameter)

        with Pool(3) as p:
            result = p.map(calculate_sum_length, self.numbers)
        return result
