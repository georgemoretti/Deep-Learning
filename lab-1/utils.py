import numpy as np

def shuffle_data(X, y):
    """
    Перемешивание данных.

    :param X: Входные данные.
    :param y: Метки.
    :return: Перемешанные данные.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def create_minibatches(X, y, batch_size):
    """
    Разбиение данных на мини-батчи.

    :param X: Входные данные.
    :param y: Метки.
    :param batch_size: Размер батча.
    :return: Генератор мини-батчей.
    """
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


def normalize_data(X):
    """
    Нормализация данных (приведение к диапазону [0, 1]).

    :param X: Входные данные.
    :return: Нормализованные данные.
    """
    return X / 255.0