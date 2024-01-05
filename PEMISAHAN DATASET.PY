import os
import random
import shutil

# Path ke folder dengan dataset (folder berisi banyak subfolder dengan gambar)
dataset_folder = '/Users/anommahartha/DATA/PY/DAGING/AUGMENTASI'

# Path ke folder tujuan untuk data train dan data test
data_train_folder = '/Users/anommahartha/DATA/PY/DAGING/V3/DATASET_TRAIN'
data_test_folder = '/Users/anommahartha/DATA/PY/DAGING/V3/DATASET_TEST'

# Proporsi data yang akan diambil sebagai data test (misalnya, 30%)

# Loop melalui setiap subfolder dalam dataset_folder
for root, dirs, files in os.walk(dataset_folder):
    # Skip folder dataset_folder itu sendiri
    if root == dataset_folder:
        continue
    
    # Hitung jumlah gambar yang akan diambil sebagai data test
    num_test_images = int(len(files) * test_ratio)
    
    # Ambil indeks acak untuk gambar yang akan diambil sebagai data test
    test_indices = random.sample(range(len(files)), num_test_images)
    
    # Buat folder yang sesuai di data_train_folder dan data_test_folder
    relative_path = os.path.relpath(root, dataset_folder)
    train_dest_folder = os.path.join(data_train_folder, relative_path)
    test_dest_folder = os.path.join(data_test_folder, relative_path)
    os.makedirs(train_dest_folder, exist_ok=True)
    os.makedirs(test_dest_folder, exist_ok=True)
    
    # Salin gambar ke folder yang sesuai (data train atau data test)
    for i, file in enumerate(files):
        source_path = os.path.join(root, file)
        if i in test_indices:
            destination_path = os.path.join(test_dest_folder, file)
        else:
            destination_path = os.path.join(train_dest_folder, file)
        shutil.copy(source_path, destination_path)

print("Pemisahan dataset selesai.")
