import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from utils import load_dataset

# Загрузка данных
images, labels = load_dataset()

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(128, input_shape=(784,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение нейросети
batch_size = 64
epochs = 10

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# # Визуализация результатов
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Эпоха')
# plt.ylabel('Потери')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Эпохи')
# plt.ylabel('Точность')
# plt.legend()
#
# plt.show()

# Загрузка пользовательского изображения
custom_image = plt.imread("custom.jpg", format="jpeg")

# Преобразование изображения в формат, аналогичный обучающим данным
gray = lambda rgb: np.dot(custom_image[..., :3], [0.299, 0.587, 0.114])
custom_image = 1 - (gray(custom_image).astype("float32") / 255)
custom_image = np.reshape(custom_image, (1, 784))  # Предполагается, что изображение имеет размер 28x28 пикселей

# Прогноз сети для пользовательского изображения
prediction = model.predict(custom_image)

plt.imshow(custom_image.reshape(28, 28), cmap="Greys")
predicted_digit = np.argmax(prediction)
plt.title(f"Нейросеть предполагает, что пользовательский номер равен: {predicted_digit}")
plt.show()
