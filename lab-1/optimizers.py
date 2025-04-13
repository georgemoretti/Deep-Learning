import numpy as np

class Optimizer:
    """
    Базовый класс для всех оптимизаторов.
    """
    def update(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Стохастический градиентный спуск (SGD).
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Обновление параметров слоя (весов и смещений).

        :param layer: Слой, параметры которого нужно обновить.
        """
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.bias -= self.learning_rate * layer.grad_bias


class MomentumSGD(Optimizer):
    """
    SGD с импульсом (Momentum SGD).
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_weights = None
        self.velocity_bias = None

    def update(self, layer):
        """
        Обновление параметров слоя с использованием импульса.

        :param layer: Слой, параметры которого нужно обновить.
        """
        if self.velocity_weights is None:
            self.velocity_weights = np.zeros_like(layer.weights)
            self.velocity_bias = np.zeros_like(layer.bias)

        # Обновление скоростей
        self.velocity_weights = self.momentum * self.velocity_weights - self.learning_rate * layer.grad_weights
        self.velocity_bias = self.momentum * self.velocity_bias - self.learning_rate * layer.grad_bias

        # Обновление параметров
        layer.weights += self.velocity_weights
        layer.bias += self.velocity_bias


class GradientClipping(Optimizer):
    """
    Ограничение градиента (Gradient Clipping).
    """
    def __init__(self, optimizer, clip_value=1.0):
        self.optimizer = optimizer
        self.clip_value = clip_value

    def update(self, layer):
        """
        Применение ограничения градиента перед обновлением параметров.

        :param layer: Слой, параметры которого нужно обновить.
        """
        # Ограничение градиентов
        layer.grad_weights = np.clip(layer.grad_weights, -self.clip_value, self.clip_value)
        layer.grad_bias = np.clip(layer.grad_bias, -self.clip_value, self.clip_value)

        # Обновление параметров через базовый оптимизатор
        self.optimizer.update(layer)