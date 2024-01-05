import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Path ke direktori dataset
train_dir = 'F:V3\DATASET_TRAIN'
test_dir = 'F:V3\DATASET_TES'
output_dir = 'F:V3\HASIL'

# Fungsi untuk membaca dataset
# Fungsi untuk membaca dataset dan mengabaikan folder .DS_Store dan file .DS_Store
def read_dataset(directory, img_height, img_width, with_filenames=False):
    images = []
    labels = []
    filenames = []  # Tambahkan list untuk menyimpan nama file gambar
    classes = os.listdir(directory)
    for idx, class_name in enumerate(classes):
        if class_name == '.DS_Store':
            continue  # Ignore the .DS_Store folder
        class_dir = os.path.join(directory, class_name)
        
        # Mengabaikan folder .DS_Store
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in tqdm(os.listdir(class_dir), desc=f'Reading {class_name}'):
            if img_name == '.DS_Store':
                continue  # Ignore .DS_Store files
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)  # Read the image using PIL
            img = img.resize((img_width, img_height))  # Resize the image to match the model
            img_array = np.array(img)
            
            # Check if the label is valid (within the range [0, 2])
            if 0 <= idx < 4:
                images.append(img_array)
                labels.append(idx)
                filenames.append(img_name)  # Simpan nama file gambar
            else:
                print(f"Skipping image with invalid label: {img_name}")
    
    if with_filenames:
        return np.array(images), np.array(labels), filenames
    else:
        return np.array(images), np.array(labels)


#HYPERPARAMETER
imgsize = 100
epoch = 100
batchsize = 16

# Ukuran gambar yang diharapkan oleh model CNN
img_width, img_height = imgsize, imgsize

# Baca data pelatihan dan pengujian
train_images, train_labels = read_dataset(train_dir, img_height, img_width)
test_images, test_labels, test_filenames = read_dataset(test_dir, img_height, img_width, with_filenames=True)


# Normalisasi nilai piksel gambar ke rentang [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') 

# Langkah 1: Definisikan arsitektur model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
    layers.Dropout(0.1),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
    layers.Dropout(0.1),
    layers.Dense(3, activation='softmax')  # Jumlah kelas = 3 (babi, sapi, campuran)
])

# Langkah 2: Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Langkah 3: Latih model menggunakan data pelatihan
history = model.fit(train_images, train_labels, epochs=epoch, batch_size=batchsize, verbose=1, validation_split=0.2)


# Ambil nilai training loss, training accuracy, validation loss, dan validation accuracy dari history
training_loss = history.history['loss']
print(f'Training loss: {training_loss[-1]:.4f}')

training_accuracy = history.history['accuracy']
print(f'Training accuracy: {training_accuracy[-1] * 100:.2f}%')

# Langkah 4: Evaluasi model menggunakan data pengujian
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Setelah pelatihan selesai, mencetak akurasi validasi
validation_loss = history.history['val_loss']
print(f'Validation loss: {validation_loss[-1]:.4f}')

validation_accuracy = history.history['val_accuracy'][-1]
print(f'Validation accuracy: {validation_accuracy * 100:.2f}%')



# Prediksi label kelas untuk data pengujian
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Konversi indeks label kelas menjadi nama kelas (jenis daging)
class_names = os.listdir(train_dir)
predicted_class_names = [class_names[label_idx] for label_idx in predicted_labels]



#EVALUASI 4 KATEGORI
# Hitung confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Hitung presisi
precision = precision_score(test_labels, predicted_labels, average='macro')

# Hitung recall
recall = recall_score(test_labels, predicted_labels, average='macro')

# Hitung F1 score
f1 = f1_score(test_labels, predicted_labels, average='macro')

# Hitung akurasi
accuracy = accuracy_score(test_labels, predicted_labels)

# Hitung jumlah data yang benar-benar cocok (true predictions)
true_predictions = np.sum(np.diag(conf_matrix))


# Simpan hasil perhitungan metrik evaluasi dalam file teks
output_folder_name = f'{img_width}x{img_height}_{len(history.epoch)}_Epoch_{batchsize}'
output_folder_path = os.path.join(output_dir, output_folder_name)

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Simpan hasil perhitungan metrik evaluasi dalam file teks
with open(os.path.join(output_folder_path, f'metrics_{img_width}x{img_height}_{len(history.epoch)}_Epoch_.txt'), 'w') as output_file:
    output_file.write(f'Confusion Matrix:\n{conf_matrix}\n')
    output_file.write(f'Akurasi : {accuracy * 100:.2f}%\n')
    output_file.write(f'Presisi : {precision * 100 :.2f}%\n')
    output_file.write(f'Recall: {recall * 100:.2f}%\n')
    output_file.write(f'F1 Score: {f1 * 100:.2f}%\n')
    output_file.write(f'True Predictions: {true_predictions}\n')




# Cetak nama gambar, jenis daging, dan akurasi untuk setiap gambar dalam data pengujian
with open(os.path.join(output_folder_path, f'test_results_{img_width}x{img_height}_{len(history.epoch)}_Epoch.txt'), 'w') as output_file:
    output_file.write("Image Name, True Class, Predicted Class\n")
    for i in range(len(test_images)):
        image_name = test_filenames[i]
        true_class = class_names[test_labels[i]]
        predicted_class = predicted_class_names[i]
        output_file.write(f"{image_name}, {true_class}, {predicted_class}\n")


# Simpan hasil training ke dalam file teks
with open(os.path.join(output_folder_path, f'hasil_training_{img_width}x{img_height}_{len(history.epoch)}_Epoch.txt'), 'w') as output_file:
    output_file.write(f'History: {history.history}\n')
    output_file.write(f'Image Width: {img_width}\n')
    output_file.write(f'Image Height: {img_height}\n')
    output_file.write(f'Number of Epochs: {len(history.epoch)}\n')
    output_file.write(f'Training loss: {training_loss[-1]:.4f}\n')
    output_file.write(f'Training accuracy: {training_accuracy[-1] * 100:.2f}%\n')
    output_file.write(f'Test accuracy: {test_accuracy * 100:.2f}%\n')
    output_file.write(f'Validation loss: {validation_loss[-1]:.4f}\n')
    output_file.write(f'Validation accuracy: {validation_accuracy * 100:.2f}%\n')

# Simpan model ke dalam file .h5
model.save(os.path.join(output_folder_path, f'model_daging{img_width}x{img_height}_{len(history.epoch)}_Epoch.keras'))


# Visualisasi akurasi dan loss selama pelatihan
plt.figure(figsize=(12, 4))

# Akurasi pelatihan dan validasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss pelatihan dan validasi
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
# Simpan gambar grafik
plt.savefig(os.path.join(output_folder_path, f'grafik_training_{img_width}x{img_height}_{len(history.epoch)}_Epoch.png'))
#plt.show()




# # Cetak array dari setiap lapisan dalam model
# for layer in model.layers:
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         print(f"Conv2D Layer: {layer.name}")
#         print(layer.get_weights()[0])  # Cetak kernel filter
#         print(layer.get_weights()[1])  # Cetak bias

#     elif isinstance(layer, tf.keras.layers.Dense):
#         print(f"Dense Layer: {layer.name}")
#         print(layer.get_weights()[0])  # Cetak weight
#         print(layer.get_weights()[1])  # Cetak bias
        