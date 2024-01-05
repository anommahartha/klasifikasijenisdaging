import cv2
import os
import numpy as np
from tqdm import tqdm

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def shift_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def flip_image(image):
    return cv2.flip(image, 1)

def zoom_image(image, value):
    centerX, centerY = int(image.shape[1] / 2), int(image.shape[0] / 2)
    radiusX, radiusY = int(image.shape[1] / 2 * value), int(image.shape[0] / 2 * value)

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cropped = image[minY:maxY, minX:maxX]
    return cv2.resize(cropped, (image.shape[1], image.shape[0]))

def augment_image(image_path, output_folder, index):
    image = cv2.imread(image_path)

    augmented_images = [
        rotate_image(image, 45),
        shift_image(image, 10, 10),
        flip_image(image),
        zoom_image(image, 0.8)
    ]
    augmentation_types = ['rotate', 'shift', 'flip', 'zoom']

    for aug_image, aug_type in zip(augmented_images, augmentation_types):
        base_name = os.path.basename(image_path)
        new_name = f"{os.path.splitext(base_name)[0]}_{aug_type}_{index}.jpg"
        cv2.imwrite(os.path.join(output_folder, new_name), aug_image)

def process_folders(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        # Filter .DS_Store files (MacOS)
        files = [f for f in files if not f.startswith('.DS_Store')]
        for i, filename in tqdm(enumerate(files), total=len(files)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                # Membuat folder output yang sesuai jika belum ada
                relative_path = os.path.relpath(root, input_folder)
                target_folder = os.path.join(output_folder, relative_path)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                augment_image(file_path, target_folder, i)

# Jalur folder input dan output
input_folder_path = '/Users/anommahartha/DATA/PY/DAGING/DATASET2'
output_folder_path = '/Users/anommahartha/DATA/PY/DAGING/AUGMENTASI'

# Jalankan fungsi dengan folder input dan output
process_folders(input_folder_path, output_folder_path)
