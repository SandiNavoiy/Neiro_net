import numpy as np  # Работа с массивами
import matplotlib.pyplot as plt  # штука для визуализации

from utils import load_dataset

#Замена нейронов смещения для скрытого и выходного слоев
images, labels = load_dataset()

# Установка весов для двух слоев нейросети, от -0,5 до 0,5
weights_input_to_hidden = np.random.uniform(
    -0.5, 0.5, (20, 784)
)  # Вывод случайых числел с запятой -0,5 до 0,5. 784 кол-во нейронов = кол ву пикселей
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))  # веса выходы 10 цифр

# нейроны смещения
bias_input_to_hidden = np.zeros((20, 1))  # массив c нулями  размерность x,y
bias_hidden_to_output = np.zeros((10, 1))
