import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from layers import Dense, ReLU, Sigmoid
from losses import CrossEntropyLoss
from optimizers import SGD
from model import NeuralNetwork
from utils import shuffle_data, create_minibatches

# Загрузка данных Iris
data = load_iris()
X = data.data.astype(np.float32)
y = data.target.astype(int).reshape(-1, 1)

# One-hot encoding меток
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = NeuralNetwork(
    layers=[
        Dense(input_size=4, output_size=10),
        ReLU(),
        Dense(input_size=10, output_size=3),
        Sigmoid()
    ],
    loss_fn=CrossEntropyLoss(),
    optimizer=SGD(learning_rate=0.01)
)

# Обучение модели
print("Training on Iris...")
model.train(X_train, y_train, epochs=50, batch_size=8)

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