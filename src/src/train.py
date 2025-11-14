"""
Training and evaluation script for the Natural Disaster Damage Detection CNN.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from src.preprocessing import split_data, create_generators, compute_class_weights
from src.model import build_cnn


def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    precision = history.history["precision"]
    recall = history.history["recall"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_acc, "r^-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training Loss")
    plt.plot(epochs, val_loss, "r^-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, precision, "bo-", label="Precision")
    plt.plot(epochs, recall, "r^-", label="Recall")
    plt.title("Precision and Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_generator):
    y_prob = model.predict(test_generator)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_class_metrics(model, test_generator):
    y_prob = model.predict(test_generator)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_labels))
    )

    x = np.arange(len(class_labels))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, class_labels, rotation=45)
    plt.ylabel("Score")
    plt.title("Per-Class Precision, Recall, and F1-score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # On Kaggle you will set this to:
    # dataset_dir = "/kaggle/input/natural-disasters-sat-images/dataset"
    dataset_dir = "PATH/TO/YOUR/DATASET"

    output_root = "data"

    # 1) Split data
    train_dir, val_dir, test_dir = split_data(dataset_dir, output_root)

    # 2) Generators
    train_gen, val_gen, test_gen = create_generators(train_dir, val_dir, test_dir)

    # 3) Class weights
    class_weights = compute_class_weights(train_gen)

    # 4) Build model
    model = build_cnn()

    # 5) Callbacks
    base_lr = 2e-4
    lr_scheduler = LearningRateScheduler(
        lambda epoch: float(base_lr * tf.math.exp(-epoch / 10)), verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    # 6) Train
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weights,
    )

    # 7) Evaluate
    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")

    # 8) Plots
    plot_history(history)
    plot_confusion_matrix(model, test_gen)
    plot_class_metrics(model, test_gen)


if __name__ == "__main__":
    main()
