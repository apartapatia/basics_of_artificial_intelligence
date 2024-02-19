import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Входные данные для обучения (в виде матрицы)
input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
# Ожидаемые выходные данные (в виде вектор - столбец)
expected_output = np.array([[0, 1, 1, 0, 0, 1, 0]]).T

# Инициализация случайных весов (веса не случайные, нужно убрать 1 в seed)
# ПО ЗАДАНИЮ ЗАКОМЕНТИРОВАТЬ ДАННУЮ СТРОЧКУ!
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

# synaptic_weights = np.zeros((3,1))


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


# Обучение модели
for iteration in iterations:
    # Прямое распространение
    output = 1 / (1 + np.exp(-(np.dot(input_data, synaptic_weights))))
    # Обратное распространение ошибки и коррекция весов
    synaptic_weights += np.dot(input_data.T, (expected_output - output) * output * (1 - output))
    # Вычисление и сохранение среднеквадратичной ошибки
    mse = mean_squared_error(expected_output, output)
    mse_values.append(mse)
    # Сохранение выхода модели для точки X = (1, 0, 0)
    outputs_for_X.append(output[5].item())  # Значение выхода для третьего примера (т.е. для X = (1, 0, 0))
    # Сохранение выхода модели для случайной выборки
    outputs_for_random_points.append(output[random_index].item())


# Создание сетки графиков
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Визуализация изменения среднеквадратичной ошибки по ходу обучения
axs[0, 0].plot(iterations, mse_values)
axs[0, 0].set_title('Изменение среднеквадратичной ошибки')
axs[0, 0].set_xlabel('Номер итерации')
axs[0, 0].set_ylabel('Среднеквадратичная ошибка')
axs[0, 0].grid(True)

# График пошагового приближения к значению в точке X = (1, 0, 0)
axs[0, 1].plot(iterations, outputs_for_X)
axs[0, 1].set_title('Приближение к значению в точке X = (1, 0, 0)')
axs[0, 1].set_xlabel('Номер итерации')
axs[0, 1].set_ylabel('Выход модели для X = (1, 0, 0)')
axs[0, 1].grid(True)

# График пошагового приближения к полученным значениям для случайных точек из input_data
axs[1, 0].plot(iterations, outputs_for_random_points)
axs[1, 0].set_title('Приближение к значениям для случайных точек')
axs[1, 0].set_xlabel('Номер итерации')
axs[1, 0].set_ylabel('Выход модели для случайных точек')
axs[1, 0].grid(True)

# Вывод предсказанного значения для входа [1, 0, 0]
predicted_output = 1 / (1 + np.exp(-(np.dot(np.array([1, 0, 0]), synaptic_weights))))
print("Предсказанное значение для входа [1 0 0]:", predicted_output.item())

# Вывод предсказанного значения для входа rand
predicted_output_rand = 1 / (1 + np.exp(-(np.dot(np.array(random_point), synaptic_weights))))
print("Предсказанное значение для входа {}: {}".format(random_point, predicted_output_rand.item()))

plt.tight_layout()
plt.show()
