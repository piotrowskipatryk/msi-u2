import numpy as np


class Classifier:
    """
    Klasa uniwersalnego klasyfikatora opartego o MNK.

    Atrybuty
    --------
    num_classes : int
        Liczba klas - etykiet

    Metody
    ------
    train(X, y):
        Wyznacza wektor parametrów funkcji kwadratowej.
    predict(c):
        Sprawdza dopasowanie do klasy wykorzystując funkcję potencjału.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # inicjalizacja macierzy parametrów
        self.theta = np.zeros((num_classes, 6))

    def train(self, x, y):
        # inicjalizacja macierzy cech dla funkcji nieliniowej
        A = np.vstack(
            [
                np.ones(len(x)),
                x[:, 0],
                x[:, 1],
                x[:, 0]**2,
                x[:, 1] * x[:, 0],
                x[:, 1]**2
            ]
        ).T
        for i in range(self.num_classes):
            y_i = (y == i).astype(int)
            # Rozwiązanie układu równań normalnych
            A_T_A_inv = np.linalg.inv(np.dot(A.T, A))
            A_T_y = np.dot(A.T, y_i)
            self.theta[i] = np.dot(A_T_A_inv, A_T_y)

    def predict(self, c):
        # wykorzystanie funkcji potencjału z wyznaczonymi dla poszczególnych
        # klas parametrami w celu znalezienia przydziału do najbardziej
        # odpowiedniej klasy
        scores = np.array([self._check_potential(c, a) for a in self.theta])
        return np.argmax(scores, axis=0)

    def _check_potential(self, c, a):
        # funkcja potencjału o postaci wielomianu drugiego stopnia
        return a[0] + a[1] * c[0] + a[2] * c[1] + a[3] * c[0]**2 + a[4] * c[0] * c[1] + a[5] * c[1]**2  # noqa
