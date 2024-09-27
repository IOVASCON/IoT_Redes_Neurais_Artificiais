# -*- coding: utf-8 -*-
"""Reconhecimento de Frutos Verdes e Maduros com Deep Learning"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuração do logger
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Diretórios de dados (ajustar os caminhos conforme necessário)
base_dir = 'L:/VSCode/PYTHON/DIO/DeepLearning_Fruits_Recognition/data'  # Caminho absoluto para a pasta "data"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_green_dir = os.path.join(train_dir, 'green_fruits')  # Diretório com as imagens de frutos verdes para treino
train_ripe_dir = os.path.join(train_dir, 'ripe_fruits')    # Diretório com as imagens de frutos maduros para treino
validation_green_dir = os.path.join(validation_dir, 'green_fruits')  # Frutos verdes para validação
validation_ripe_dir = os.path.join(validation_dir, 'ripe_fruits')    # Frutos maduros para validação

# Contagem de imagens de treino e validação
num_green_tr = len(os.listdir(train_green_dir))
num_ripe_tr = len(os.listdir(train_ripe_dir))

num_green_val = len(os.listdir(validation_green_dir))
num_ripe_val = len(os.listdir(validation_ripe_dir))

total_train = num_green_tr + num_ripe_tr
total_val = num_green_val + num_ripe_val

print('Total training green fruit images:', num_green_tr)
print('Total training ripe fruit images:', num_ripe_tr)
print('Total validation green fruit images:', num_green_val)
print('Total validation ripe fruit images:', num_ripe_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Parâmetros do modelo
BATCH_SIZE = 100  # Número de exemplos por lote
IMG_SHAPE  = 150  # Tamanho das imagens de entrada

# Pré-processamento das imagens (geração de lotes de treino e validação)
train_image_generator = ImageDataGenerator(rescale=1./255)  # Normalização das imagens de treino
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Normalização das imagens de validação

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=validation_dir,
    shuffle=False,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='binary'
)

# Visualizando algumas imagens de treino
sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    # Salvar a visualização das frutas
    plt.savefig('./results/process_images/sample_training_images.png')  # Salva a imagem na pasta
    plt.close()  # Fecha a figura para não exibir a janela interativa


plotImages(sample_training_images[:5])

# Definindo a arquitetura da rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Para classificação binária
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Treinando o modelo
EPOCHS = 20  # Você pode ajustar o número de épocas conforme necessário
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

# Visualizando os resultados de treino
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Ajustar o número de épocas reais de treino e validação
# Encontre o menor número de épocas entre treino e validação para evitar o erro
actual_epochs = min(len(acc), len(val_acc))  # Pega o menor tamanho entre treino e validação
epochs_range = range(actual_epochs)  # Ajusta o range para o número correto de épocas

# Plotando o gráfico de acurácia e perda
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc[:actual_epochs], label='Training Accuracy')  # Usando apenas até a última época
plt.plot(epochs_range, val_acc[:actual_epochs], label='Validation Accuracy')  # Usando apenas até a última época
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss[:actual_epochs], label='Training Loss')  # Usando apenas até a última época
plt.plot(epochs_range, val_loss[:actual_epochs], label='Validation Loss')  # Usando apenas até a última época
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Salvar os gráficos de treino e validação
plt.savefig('./results/process_images/training_validation_metrics.png')  # Salva os gráficos de acurácia/perda
plt.close()  # Fecha a figura para não exibir a janela interativa

