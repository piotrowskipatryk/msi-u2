import os

import numpy as np
import matplotlib.pyplot as plt


def test_classifier(classifier, x, y):
    """ Funckja testująca klasyfikator.
    Rezultat to reprezentacja graficzna i plik monitorujący. """
    try:
        os.makedirs('tests')
    except FileExistsError:
        pass
    # samples plot
    scatter = plt.scatter(
        x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, marker='o')
    plt.title("wizualizacja próbek uczących")
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc="upper right", bbox_to_anchor=(1.32, 1),
        labels=["stan stały", "stan ciekły", "stan gazowy"])
    plt.xlabel("temperatura (°C)")
    plt.ylabel("ciśnienie (hPa)")
    plt.ticklabel_format(style='plain')
    plt.savefig(
        os.path.join('tests', 'learning_samples.png'),
        bbox_inches='tight'
)

    # predictions plot
    y_pred = [classifier.predict(sample) for sample in x]
    plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap=plt.cm.coolwarm, marker='o')
    plt.title("wizualizacja predykcji dla próbek uczących")
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc="upper right",
        bbox_to_anchor=(1.32, 1),
        labels=["stan stały", "stan ciekły", "stan gazowy"]
    )
    plt.xlabel("temperatura (°C)")
    plt.ylabel("ciśnienie (hPa)")
    plt.savefig(
        os.path.join('tests', 'predicted_samples.png'),
        bbox_inches='tight'
    )

    with open(os.path.join('tests', 'monitor.txt'), 'w') as f:
        monitor = f"liczba próbek uczących: {len(x)}\n"
        monitor += f"skuteczność klasyfikacji: {accuracy(y, y_pred)}%\n"
        monitor += "macierz pomyłek między-klasowych:\n"
        monitor += "(stały | ciekły | gazowy)\n"
        monitor += str(confusion_matrix(y, y_pred))
        f.write(monitor)


def accuracy(y_true, y_pred):
    """ Funkcja do obliczania skuteczności klasyfikacji """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return round(accuracy * 100, 2)


def confusion_matrix(y_true, y_pred):
    """ Funkcja do obliczania macierzy pomyłek """
    conf_matrix = np.zeros((3, 3), dtype=int)
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    return conf_matrix
