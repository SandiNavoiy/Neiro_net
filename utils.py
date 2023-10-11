import numpy as np


def load_dataset():
    """Функция загрузки инфы из файла"""
    with np.load("mnist.npz") as f:
        # конвертация  цветов из RGB в формат Unit RGB
        x_train = f['x_train'].astype("float32") / 255

        # меняем форму массива изображений из (60000, 28, 28) в (60000, 784), 784 нейрона входной нейросети 28*28 = 784
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        # labels
        y_train = f['y_train']

        # конвертация в двухмерный массив 60000 на 10. 10 это 10 возможных цифр
        y_train = np.eye(10)[y_train]

        return x_train, y_train
