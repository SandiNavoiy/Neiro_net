import numpy as np  # Работа с массивами
import matplotlib.pyplot as plt  # штука для визуализации

from utils import load_dataset

# Замена нейронов смещения для скрытого и выходного слоев
images, labels = load_dataset()

# Установка весов для двух слоев нейросети, от -0,5 до 0,5
weights_input_to_hidden = np.random.uniform(
    -0.5, 0.5, (20, 784)
)  # Вывод случайых числел с запятой -0,5 до 0,5. 784 кол-во нейронов = кол ву пикселей
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))  # веса выходы 10 цифр

# нейроны смещения
bias_input_to_hidden = np.zeros((20, 1))  # массив c нулями  размерность x,y
bias_hidden_to_output = np.zeros((10, 1))

# Обучение нейросети, коррекция весов
epochs = 10  # количество эпох обучения
e_loss = 0
e_correct = 0
learning_rate = 0.01  # точность

for epoch in range(epochs):
    print(f"Эпоха № {epoch}")
    # zip итератор, который объединяет элементы изн ескольких источников данных
    for image, label in zip(images,labels):
        # -1 означает, что строка числа (первые измерения) будет
        # вычислена автоматически так, чтобы элементы общего числа в массиве оставались неизменными,
        # а это 1 указывает на то, что каждый элемент в данном случае будет представлен в виде одного столбца
        image = np.reshape(image, (-1, 1)) # reshape изменение самого массива без изменения его содержания  -  в дмумерный
        label = np.reshape(label, (-1, 1))

        # Forward propagation (первый этап обучения)
        #Данные подаются на ввод
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))  # нормализация, сигмойд функция

        # Forward propagation (выходной слой)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        # потери + накопление ошибки
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Backpropagation (выходной слой) корректировка весов
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Backpropagation (hidden layer)
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    # Вывод ошибок
    print(f"Потери: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Точность: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

# случайное изображение из файла
test_image = plt.imread("custom.jpg", format="jpeg")

# Grayscale + Unit RGB + inverse colors
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
test_image = 1 - (gray(test_image).astype("float32") / 255)

# Reshape
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

# Обучение
image = np.reshape(test_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid
# Forward propagation (to output layer)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"Нейросеть предполагает, что пользовательский номер равен: {output.argmax()}")
plt.show()