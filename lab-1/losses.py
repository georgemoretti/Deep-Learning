import numpy as np

class Loss:
    """
    Базовый класс для всех функций потерь.
    """
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):
    """
    Среднеквадратичная ошибка (Mean Squared Error).
    Используется для задач регрессии.
    """
    def forward(self, y_pred, y_true):
        """
        Вычисление MSE.

        :param y_pred: Предсказанные значения.
        :param y_true: Истинные значения.
        :return: Значение ошибки.
        """
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        """
        Вычисление градиента MSE.

        :param y_pred: Предсказанные значения.
        :param y_true: Истинные значения.
        :return: Градиент ошибки.
        """
        return 2 * (y_pred - y_true) / y_pred.shape[0]


class CrossEntropyLoss(Loss):
    """
    Кросс-энтропийная ошибка (Cross Entropy Loss).
    Используется для задач классификации.
    """
    def forward(self, y_pred, y_true):
        """
        Вычисление CrossEntropyLoss.

        :param y_pred: Предсказанные вероятности (logits).
        :param y_true: Истинные метки (one-hot encoding).
        :return: Значение ошибки.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Защита от log(0)
        return -np.mean(y_true * np.log(y_pred))

    def backward(self, y_pred, y_true):
        """
        Вычисление градиента CrossEntropyLoss.

        :param y_pred: Предсказанные вероятности (logits).
        :param y_true: Истинные метки (one-hot encoding).
        :return: Градиент ошибки.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Защита от деления на 0
        return -(y_true / y_pred) / y_pred.shape[0]