"""
Preprocessing utilities for the Natural Disaster Damage Detection project.
- Splits the dataset into train/val/test with stratification
- Builds Keras ImageDataGenerators with augmentation
- Computes class weights for imbalanced data
"""

import os
import shutil
from typing import Tuple, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CATEGORIES = ["drought", "earthquake", "hurricane",
              "landslide", "no damage", "wild fire"]


def split_data(
    dataset_dir: str,
    output_root: str = "data",
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[str, str, str]:
    """
    Split images in `dataset_dir` into train / val / test folders using stratified sampling.
    Assumes `dataset_dir/<category>/*` structure.
    Returns: (train_dir, val_dir, test_dir)
    """
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")
    test_dir = os.path.join(output_root, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in CATEGORIES:
        source_dir = os.path.join(dataset_dir, category)
        files = sorted(os.listdir(source_dir))

        # train / temp split
        train_files, temp_files = train_test_split(
            files, train_size=train_size, random_state=42
        )

        # relative validation size from remaining data
        rel_val_size = val_size / (1.0 - train_size)
        val_files, test_files = train_test_split(
            temp_files, train_size=rel_val_size, random_state=42
        )

        def _copy(file_list, target_root):
            cat_dir = os.path.join(target_root, category)
            os.makedirs(cat_dir, exist_ok=True)
            for f in file_list:
                shutil.copy(os.path.join(source_dir, f),
                            os.path.join(cat_dir, f))

        _copy(train_files, train_dir)
        _copy(val_files, val_dir)
        _copy(test_files, test_dir)

    return train_dir, val_dir, test_dir


def create_generators(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    img_size: int = 200,
    batch_size: int = 32,
):
    """Create train / val / test ImageDataGenerators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=35,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        vertical_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def compute_class_weights(train_generator) -> Dict[int, float]:
    """Compute balanced class weights from a Keras generator."""
    class_indices = train_generator.class_indices
    labels = train_generator.classes

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    weight_dict = dict(zip(np.unique(labels), class_weights))

    print("Class weights:")
    for idx, w in weight_dict.items():
        class_name = [k for k, v in class_indices.items() if v == idx][0]
        print(f"  {idx} ({class_name}): {w:.3f}")

    return weight_dict
