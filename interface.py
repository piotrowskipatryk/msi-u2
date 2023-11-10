import json

import numpy as np

from method import Classifier
from problem import translations
from problem import generate_samples
from utils import test_classifier


classifier = Classifier(num_classes=3)


def menu():
    print("---")
    print("Wybierz pozycję z menu:")
    print("1. wygeneruj zestaw próbek uczących")
    print("2. załaduj zestaw próbek uczących i wyznacz parametry funkcji potencjału")  # noqa
    print("3. test poprawności działania klasyfikatora")
    print("4. manualna klasyfikacja")
    print("")
    print("wybór (1/2/3/4): ", end="")
    choice = input()
    main(choice)


def load_samples():
    try:
        with open("samples.json") as f:
            data = json.loads(f.read())
            return np.array(data[0]), np.array(data[1])
    except Exception as ex:
        print("Błąd przy ładowaniu próbek uczących:", ex)
        exit()


def main(choice):
    if choice == '1':
        generate_samples(1000)
        print("gotowe.")
        print("")
        menu()
    elif choice == '2':
        x, y = load_samples()
        classifier.train(x, y)
        print("gotowe.")
        print("")
        menu()
    elif choice == '3':
        x, y = load_samples()
        test_classifier(classifier, x, y)
        print("rezultat testów został zapisany do katalogu test/")
        exit()
    elif choice == '4':
        print("")
        print("Podaj wartość temperatury (°C) do predykcji:")
        temperature = float(input())
        print("Podaj wartość ciśnienia atmosferycznego (hPa) do predykcji:")
        air_pressure = float(input())
        y_pred = classifier.predict((temperature, air_pressure))
        print(
            "Przewidziana postać wody dla temperatury ",
            f"{temperature}°C",
            "i ciśnienia",
            f"{air_pressure}hPa",
            "to",
            translations.get(y_pred)
        )
    else:
        print("Nie dokonano poprawnego wyboru.")
        exit()


if __name__ == "__main__":
    print("Program implementujący klasyfikator wg. wielomianowej funkcji potencjału dla 3 klas")  # noqa
    print("na przykładzie przewidywania stanu wody w zadanej temperaturze i ciśnieniu atmosferycznym.")  # noqa
    menu()
