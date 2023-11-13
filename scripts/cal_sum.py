from cal_sum_import import Radiation

model = Radiation(range(12))

if __name__ == "__main__":
    result = model.generate(33)
    print(result)
