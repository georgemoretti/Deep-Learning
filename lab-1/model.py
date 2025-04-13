from losses import Loss
from optimizers import Optimizer

class NeuralNetwork:
    """
    Класс нейронной сети.
    """
    def __init__(self, layers, loss_fn, optimizer):
        """
        Инициализация модели.

        :param layers: Список слоев.
        :param loss_fn: Функция потерь.
        :param optimizer: Оптимизатор.
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, x):
        """
        Прямое распространение.

        :param x: Входные данные.
        :return: Выходные данные.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """
        Обратное распространение.

        :param grad_output: Градиент потерь по выходу.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train(self, X, y, epochs, batch_size):
        """
        Обучение модели.

        :param X: Входные данные.
        :param y: Метки.
        :param epochs: Количество эпох.
        :param batch_size: Размер батча.
        """
        for epoch in range(epochs):
            X_shuffled, y_shuffled = shuffle_data(X, y)
            total_loss = 0

            for X_batch, y_batch in create_minibatches(X_shuffled, y_shuffled, batch_size):
                # Forward pass
                y_pred = self.forward(X_batch)

                # Вычисление потерь
                loss = self.loss_fn.forward(y_pred, y_batch)
                total_loss += loss

                # Backward pass
                grad_output = self.loss_fn.backward(y_pred, y_batch)
                self.backward(grad_output)

                # Обновление параметров
                for layer in self.layers:
                    if hasattr(layer, 'weights'):  # Обновляем только слои с обучаемыми параметрами
                        self.optimizer.update(layer)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X)}")