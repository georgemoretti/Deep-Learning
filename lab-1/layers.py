import numpy as np

class Layer:
    """
    Базовый класс для всех слоев.
    Все слои должны реализовывать методы forward и backward.
    """
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Dense(Layer):
    """
    Полносвязный слой (Dense layer).
    Y = X * W + B.
    """
    def __init__(self, input_size, output_size):
        """
        Инициализация весов и смещений.

        :param input_size: Размер входных данных.
        :param output_size: Размер выходных данных.
        """
        # Инициализация весов небольшими случайными значениями
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None  # Сохраняем вход для backward pass

    def forward(self, x):
        """
        Прямое распространение (forward pass).

        :param x: Входные данные (матрица размера [batch_size, input_size]).
        :return: Выходные данные (матрица размера [batch_size, output_size]).
        """
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        """
        Обратное распространение (backward pass).

        :param grad_output: Градиент потерь по выходу слоя.
        :param learning_rate: Скорость обучения.
        :return: Градиент потерь по входу слоя.
        """
        # Градиенты по параметрам
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Градиент по входу
        grad_input = np.dot(grad_output, self.weights.T)

        # Обновление параметров
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class ReLU(Layer):
    """
    Активационная функция ReLU (Rectified Linear Unit).
    f(x) = max(0, x).
    """
    def __init__(self):
        self.mask = None  # Маска для обратного распространения

    def forward(self, x):
        """
        Прямое распространение.

        :param x: Входные данные.
        :return: Выходные данные после применения ReLU.
        """
        self.mask = (x > 0)  # True, если x > 0, иначе False
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Обратное распространение.

        :param grad_output: Градиент потерь по выходу слоя.
        :return: Градиент потерь по входу слоя.
        """
        return grad_output * self.mask


class Sigmoid(Layer):
    """
    Активационная функция Sigmoid.
    f(x) = 1 / (1 + exp(-x)).
    """
    def __init__(self):
        self.output = None  # Сохраняем выход для backward pass

    def forward(self, x):
        """
        Прямое распространение.

        :param x: Входные данные.
        :return: Выходные данные после применения Sigmoid.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        """
        Обратное распространение.

        :param grad_output: Градиент потерь по выходу слоя.
        :return: Градиент потерь по входу слоя.
        """
        # Градиент Sigmoid: y * (1 - y)
        return grad_output * self.output * (1 - self.output)