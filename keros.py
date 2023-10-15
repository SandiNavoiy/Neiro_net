import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

from utils import load_dataset

# Загрузка данных
images, labels = load_dataset()

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(20, input_shape=(784,), activation='sigmoid'))
model.add(Dense(64, activation='relu'))  # Дополнительный скрытый слой с 64 нейронами и функцией активации ReLU
model.add(Dense(32, activation='relu'))  # Еще один скрытый слой с 32 нейронами и функцией активации ReLU
model.add(Dense(10, activation='sigmoid'))

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
batch_size = 32  # Размер батча (можно настраивать)
# Обучение нейросети
epochs = 3
for epoch in range(epochs):
    print(f"Эпоха № {epoch}")
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # Преобразование данных в массивы батчей
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        # Обучение на батче
        history = model.fit(batch_images, batch_labels, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]

    # Вывод ошибок
    print(f"Потери: {round(loss * 100, 3)}%")
    print(f"Точность: {round(accuracy * 100, 3)}")

# Загрузка пользовательского изображения
test_image = plt.imread("custom.jpg", format="jpeg")
gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
test_image = 1 - (gray(test_image).astype("float32") / 255)
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

# Предсказание сети для пользовательского изображения
user_image = test_image.reshape(1, -1)
prediction = model.predict(user_image)

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
predicted_digit = np.argmax(prediction)
plt.title(f"Нейросеть предполагает, что пользовательский номер равен: {predicted_digit}")
plt.show()