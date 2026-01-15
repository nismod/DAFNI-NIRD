"""Constants used in the network flow model"""

CONV_METER_TO_MILE = 0.000621371
CONV_MILE_TO_KM = 1.60934
CONV_KM_TO_MILE = 0.621371
PENCE_TO_POUND = 0.01

VOT_POUND_PER_HOUR = {
    "car": 21.41,  # perceived cost in GBP per hour, car and taxi (passenger and driver)
    "lgv": 16.87,
    "ogv": 20.35,
    "psv": 13.15,
    "rail": 38.28,
}

FUEL_LITRE_PER_KM = {
    "car": {"a": 0.46693, "b": 0.08257, "c": -0.00088, "d": 0.000006},
    "lgv": {"a": 0.40392, "b": 0.15318, "c": -0.00467, "d": 0.000033},
    "ogv": {"a": 4.18094, "b": 0.21864, "c": -0.00298, "d": 0.000023},
    "psv": {"a": 3.36019, "b": 0.29525, "c": -0.00321, "d": 0.000024},
}

NON_FUEL_PENCE_PER_KM = {
    "car": {"a": 6.885, "b": 188.486},
    "lgv": {"a": 10.001, "b": 65.321},
    "ogv": {"a": 13.709, "b": 535.418},
    "psv": {"a": 42.233, "b": 962.974},
}
