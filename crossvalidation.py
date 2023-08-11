import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    return (images, masks)

def crossvalidation(path, batch_size, learning_rate, num_epochs, k_folds):
    (all_x, all_y) = load_data(path)

    kf = KFold(n_splits=k_folds, shuffle=True)

    metrics_per_fold = []

    for fold, (train_index, val_index) in enumerate(kf.split(all_x, all_y)):
        train_x, train_y = np.take(all_x, train_index, axis=0), np.take(all_y, train_index, axis=0)
        valid_x, valid_y = np.take(all_x, val_index, axis=0), np.take(all_y, val_index, axis=0)

        train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
        valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

        model = build_model()
        opt = tf.keras.optimizers.Adam(learning_rate)
        metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

        callbacks = [
            ModelCheckpoint(f"files/model_fold{fold+1}.h5", verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
            CSVLogger(f"files/data_fold{fold+1}_lr{learning_rate}.csv"),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        ]

        train_steps = len(train_x) // batch_size
        valid_steps = len(valid_x) // batch_size

        if len(train_x) % batch_size != 0:
            train_steps += 1
        if len(valid_x) % batch_size != 0:
            valid_steps += 1

        model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=num_epochs,
                    steps_per_epoch=train_steps,
                    validation_steps=valid_steps,
                    callbacks=callbacks)

        data = pd.read_csv(f"files/data_fold{fold+1}_lr{learning_rate}.csv")
        columns_to_round = data.columns[data.columns != 'lr']
        data[columns_to_round] = data[columns_to_round].round(decimals=3)
        print("All done Here ********************")
        # Confusion Matrix and Classification Report
        # y_val_true = np.concatenate([y for x, y in valid_dataset], axis=0)
        # y_val_pred = model.predict(valid_dataset)
        # y_val_pred = (y_val_pred > 0.5).astype(np.uint8)  # Convert probabilities to binary predictions

        # cm = confusion_matrix(y_val_true.flatten(), y_val_pred.flatten())
        # class_report = classification_report(y_val_true.flatten(), y_val_pred.flatten())

        # print(f"Confusion Matrix (Fold {fold+1}, LR={learning_rate}):\n{cm}")
        # print(f"Classification Report (Fold {fold+1}, LR={learning_rate}):\n{class_report}")

        # Learning Curves
        plt.figure(figsize=(8, 6))
        plt.plot(data['loss'], label='Training Loss')
        plt.plot(data['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Learning Curves (Fold {fold+1}, LR={learning_rate})')
        plt.legend()
        plt.savefig(f'files/learning_curves_fold{fold+1}_lr{learning_rate}.png')
        plt.close()

        metrics_per_fold.append(pd.read_csv(f"files/data_fold{fold+1}_lr{learning_rate}.csv"))

    # Calculate mean metrics over all folds
    metrics_overall = pd.concat(metrics_per_fold).groupby(level=0).mean()
    metrics_overall.to_csv("files/data_overall.csv", index=False)

if __name__ == "__main__":
    ## Dataset
    path = "CVC-612/"
    batch_size = 8
    learning_rate = 1e-4 
    num_epochs = 10
    k_folds = 5

    crossvalidation(path, batch_size, learning_rate, num_epochs, k_folds)
