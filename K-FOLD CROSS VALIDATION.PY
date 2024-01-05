import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


# Path ke direktori dataset
train_dir = '/Users/anommahartha/DATA/PY/DAGING/V3/DATASET_TRAIN'
test_dir = '/Users/anommahartha/DATA/PY/DAGING/V3/DATASET_TEST'
output_dir = '/Users/anommahartha/DATA/PY/DAGING/V3_KFOLD/'

# Fungsi untuk membaca dataset
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

# HYPERPARAMETER
imgsize = 75
epoch = 100
batchsize = 64
num_folds = 5

# Ukuran gambar yang diharapkan oleh model CNN
img_width, img_height = imgsize, imgsize

# Baca data pelatihan dan pengujian
train_images, train_labels = read_dataset(train_dir, img_height, img_width)
test_images, test_labels, test_filenames = read_dataset(test_dir, img_height, img_width, with_filenames=True)

# Normalisasi nilai piksel gambar ke rentang [0, 1] hanya untuk data pelatihan
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32')  # Tidak melakukan normalisasi pada data pengujian


skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Inisialisasi list untuk menyimpan hasil evaluasi dari setiap fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Inisialisasi list untuk menyimpan metrik evaluasi dari setiap fold
fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

# Folder utama untuk semua output
main_output_folder_path = os.path.join(output_dir, f'{img_width}x{img_height}_{epoch}Epoch_{batchsize}_{num_folds}')

# Pastikan folder utama ada atau buat jika belum ada
if not os.path.exists(main_output_folder_path):
    os.makedirs(main_output_folder_path)



# Loop untuk setiap fold
for fold, (train_index, val_index) in enumerate(skf.split(train_images, train_labels), 1):
    print(f'\nTraining Fold {fold}...')
    X_train, X_val = train_images[train_index], train_images[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    # Langkah 1: Definisikan arsitektur model CNN
    model = models.Sequential([
        # layers.Conv2D(32, (10, 10), activation='relu', input_shape=(img_height, img_width, 3)),  # Filter 5x5
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(64, (10, 10), activation='relu'),  # Filter 5x5
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(128, (10, 10), activation='relu'),  # Filter 5x5
        # layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(img_height, img_width, 3)),  # Filter 5x5
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),  # Filter 5x5
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu'),  # Filter 5x5
        layers.MaxPooling2D((2, 2)),

        # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(128, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),

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

    # Langkah 3: Latih model menggunakan data pelatihan dan validasi
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batchsize, verbose=1, validation_data=(X_val, y_val))

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

    # EVALUASI 4 KATEGORI
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
    output_folder_name = f'{img_width}x{img_height}_{len(history.epoch)}Epoch_{fold}'
    output_folder_path = os.path.join(main_output_folder_path, output_folder_name)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    # Simpan hasil training ke dalam file teks
    with open(os.path.join(output_folder_path, f'hasil_training_{img_width}x{img_height}_{len(history.epoch)}_Epoch_Fold{fold}.txt'), 'w') as output_file:
        output_file.write(f'History: {history.history}\n')
        output_file.write(f'Image Width: {img_width}\n')
        output_file.write(f'Image Height: {img_height}\n')
        output_file.write(f'Number of Epochs: {len(history.epoch)}\n')
        output_file.write(f'Batch Size: {batchsize}\n')
        output_file.write(f'Training loss: {history.history["loss"][-1]:.4f}\n')
        output_file.write(f'Training accuracy: {history.history["accuracy"][-1] * 100:.2f}%\n')
        output_file.write(f'Validation loss: {validation_loss[-1]:.4f}\n')
        output_file.write(f'Validation accuracy: {validation_accuracy * 100:.2f}%\n')

    # Simpan hasil perhitungan metrik evaluasi dalam file teks
    with open(os.path.join(output_folder_path, f'matriks_{img_width}x{img_height}_{len(history.epoch)}_Epoch_Fold{fold}.txt'), 'w') as output_file:
        output_file.write(f'Confusion Matrix:\n{conf_matrix}\n')
        output_file.write(f'Akurasi : {accuracy * 100:.2f}%\n')
        output_file.write(f'Presisi : {precision * 100 :.2f}%\n')
        output_file.write(f'Recall: {recall * 100:.2f}%\n')
        output_file.write(f'F1 Score: {f1 * 100:.2f}%\n')
        output_file.write(f'True Predictions: {true_predictions}\n')

        for i, class_name in enumerate(class_names):
            output_file.write(f'{class_name}:\n')
            output_file.write(f'  True Positives: {conf_matrix[i, i]}\n')
            output_file.write(f'  False Positives: {np.sum(conf_matrix[i, :]) - conf_matrix[i, i]}\n')
            output_file.write(f'  False Negatives: {np.sum(conf_matrix[:, i]) - conf_matrix[i, i]}\n')
            output_file.write(f'  True Negatives: {np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]}\n')

    # Cetak nama gambar, jenis daging, dan akurasi untuk setiap gambar dalam data pengujian
    with open(os.path.join(output_folder_path, f'hasil_test_{img_width}x{img_height}_{len(history.epoch)}_Epoch_Fold{fold}.txt'), 'w') as output_file:
        output_file.write("Image Name, True Class, Predicted Class\n")
        for i in range(len(test_images)):
            image_name = test_filenames[i]
            true_class = class_names[test_labels[i]]
            predicted_class = predicted_class_names[i]
            output_file.write(f"{image_name}, {true_class}, {predicted_class}\n")

    # Simpan model ke dalam file .h5
    model.save(os.path.join(output_folder_path, f'model_daging{img_width}x{img_height}_{len(history.epoch)}_Epoch_Fold{fold}.keras'))

    # Menyimpan hasil evaluasi dari fold tersebut ke dalam list
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Menambahkan metrik evaluasi dari fold tersebut ke dalam list fold-wise
    fold_accuracies.append(accuracy * 100)  # Mengubah nilai menjadi persentase
    fold_precisions.append(precision * 100)  # Mengubah nilai menjadi persentase
    fold_recalls.append(recall * 100)  # Mengubah nilai menjadi persentase
    fold_f1_scores.append(f1 * 100)  # Mengubah nilai menjadi persentase

    # Visualisasi akurasi dan loss selama pelatihan
    plt.figure(figsize=(12, 4))

    # Akurasi pelatihan dan validasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()

    # Loss pelatihan dan validasi
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # Simpan gambar grafik
    plt.savefig(os.path.join(output_folder_path, f'grafik_training_{img_width}x{img_height}_{len(history.epoch)}_Epoch_Fold{fold}.png'))


# Hitung rata-rata metrik evaluasi dari setiap fold
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions) 
average_recall = np.mean(recalls) 
average_f1 = np.mean(f1_scores)

# Buat grafik rekapitulasi K-Fold Cross Validation
plt.figure(figsize=(14, 8))

# Akurasi
plt.subplot(2, 2, 1)
bars = plt.bar(range(1, num_folds + 1), fold_accuracies, color='blue', alpha=0.7, label='Fold Accuracy')
plt.axhline(y=average_accuracy, color='red', linestyle='--', label=f'Average Accuracy: {average_accuracy:.2f}%')
plt.xlabel('Fold')
plt.ylabel('Accuracy (%)')
plt.title('K-Fold Cross Validation - Accuracy')
plt.legend()
# Menambahkan nilai persentase pada setiap batang grafik
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', color='black')

# Presisi
plt.subplot(2, 2, 2)
bars = plt.bar(range(1, num_folds + 1), fold_precisions, color='green', alpha=0.7, label='Fold Precision')
plt.axhline(y=average_precision, color='red', linestyle='--', label=f'Average Precision: {average_precision:.2f}%')
plt.xlabel('Fold')
plt.ylabel('Precision (%)')
plt.title('K-Fold Cross Validation - Precision')
plt.legend()
# Menambahkan nilai persentase pada setiap batang grafik
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', color='black')

# Recall
plt.subplot(2, 2, 3)
bars = plt.bar(range(1, num_folds + 1), fold_recalls, color='orange', alpha=0.7, label='Fold Recall')
plt.axhline(y=average_recall, color='red', linestyle='--', label=f'Average Recall: {average_recall:.2f}%')
plt.xlabel('Fold')
plt.ylabel('Recall (%)')
plt.title('K-Fold Cross Validation - Recall')
plt.legend()
# Menambahkan nilai persentase pada setiap batang grafik
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', color='black')

# F1 Score
plt.subplot(2, 2, 4)
bars = plt.bar(range(1, num_folds + 1), fold_f1_scores, color='purple', alpha=0.7, label='Fold F1 Score')
plt.axhline(y=average_f1, color='red', linestyle='--', label=f'Average F1 Score: {average_f1:.2f}%')
plt.xlabel('Fold')
plt.ylabel('F1 Score (%)')
plt.title('K-Fold Cross Validation - F1 Score')
plt.legend()
# Menambahkan nilai persentase pada setiap batang grafik
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', color='black')

plt.tight_layout()
plt.savefig(os.path.join(main_output_folder_path, f'k_fold_summary_{img_width}x{img_height}_{len(history.epoch)}_Epoch.png'))

# Simpan hasil rata-rata metrik evaluasi dalam file teks
with open(os.path.join(main_output_folder_path, f'average_metrics_{img_width}x{img_height}_{len(history.epoch)}_Epoch.txt'), 'w') as output_file:
    output_file.write(f'Average Accuracy : {average_accuracy * 100:.2f}%\n')
    output_file.write(f'Average Precision: {average_precision * 100 :.2f}%\n')
    output_file.write(f'Average Recall   : {average_recall * 100:.2f}%\n')
    output_file.write(f'Average F1 Score : {average_f1 * 100:.2f}%\n')
