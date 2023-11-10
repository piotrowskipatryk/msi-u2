""" przewidywanie stanu wody w danej temperaturze i ciśnieniu atmosferycznym """  # noqa
import json
import random

# stałe
SOLID = 0
LIQUID = 1
GAS = 2

translations = {
    SOLID: "stały",
    LIQUID: "ciekły",
    GAS: "gazowy",
}


def generate_samples(number):
    x = []
    for z in range(number):
        x.append((random.uniform(-273, -60), random.uniform(0.01, 100000))) # solid (Celsius, hPa)  # noqa
    for z in range(number):
        x.append((random.uniform(0, 450), random.uniform(10000, 100000))) # liquid  # noqa
    for z in range(number):
        x.append((random.uniform(100, 470), random.uniform(0.01, 1000))) # gas  # noqa

    y = []
    y.extend([SOLID] * number)
    y.extend([LIQUID] * number)
    y.extend([GAS] * number)

    with open("samples.json", "w") as f:
        f.write(json.dumps([x, y]))
