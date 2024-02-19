import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Входные данные для обучения (в виде матрицы)
input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
# Ожидаемые выходные данные (в виде столбца)
expected_output = np.array([[0, 1, 1, 0, 0, 1, 0]]).T

# Инициализация случайных весов (веса не случайные, нужно убрать 1 в seed)
# ПО ЗАДАНИЮ ЗАКОМЕНТИРОВАТЬ ДАННУЮ СТРОЧКУ!
# np.random.seed(1)
# synaptic_weights = 2 * np.random.random((3, 1)) - 1

synaptic_weights = np.ones((3,1))


# Списки для хранения значений MSE и номера итерации
iterations = list(range(100))
mse_values = []

# Список для хранения выходов модели на каждой итерации для точки X = (1, 0, 0)
outputs_for_X = []
# Список для хранения выходов модели на каждой итерации для точек X случайной выборки
outputs_for_random_points = []


np.random.seed()
random_index = np.random.randint(len(input_data))
random_point = input_data[random_index]
print("Выбранная случайно точка:", random_point, "Индекс:", random_index)


for iteration in range(3):
    # Прямое распространение
    output = 1 / (1 + np.exp(-(np.dot(input_data, synaptic_weights))))
    # Обратное распространение ошибки и коррекция весов
    synaptic_weights += np.dot(input_data.T, (expected_output - output) * output * (1 - output))
    # Вычисление и сохранение среднеквадратичной ошибки
    mse = mean_squared_error(expected_output, output)
    mse_values.append(mse)

# Вывод финальных значений весовых коэффициентов
print("Финальные значения весовых коэффициентов:")
print(synaptic_weights)
