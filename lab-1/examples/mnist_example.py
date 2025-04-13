import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from layers import Dense, ReLU, Sigmoid
from losses import CrossEntropyLoss
from optimizers import SGD
from model import NeuralNetwork
from utils import normalize_data, shuffle_data, create_minibatches

# Загрузка данных MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.astype(np.float32)
y = y.astype(int)

# Нормализация данных
X = normalize_data(X)

# One-hot encoding меток
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = NeuralNetwork(
    layers=[
        Dense(input_size=784, output_size=128),
        ReLU(),
        Dense(input_size=128, output_size=64),
        ReLU(),
        Dense(input_size=64, output_size=10),
        Sigmoid()
    ],
    loss_fn=CrossEntropyLoss(),
    optimizer=SGD(learning_rate=0.01)
)

# Обучение модели
print("Training on MNIST...")
model.train(X_train, y_train, epochs=10, batch_size=64)

# Тестирование модели
def evaluate_model(model, X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        y_pred = model.forward(X_test[i].reshape(1, -1))
        predicted_class = np.argmax(y_pred)
        true_class = np.argmax(y_test[i])
        if predicted_class == true_class:
            correct += 1
    accuracy = correct / len(X_test)
    print(f"Test Accuracy: {accuracy:.4f}")

evaluate_model(model, X_test, y_test)