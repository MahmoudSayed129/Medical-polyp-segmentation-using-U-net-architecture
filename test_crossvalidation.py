import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from train import iou
import pandas as pd

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    ## Dataset
    path = "CVC-612/"
    batch_size = 8
    k_folds = 5
    learning_rate = 1e-4
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    with CustomObjectScope({'iou': iou}):
        models = []
        for fold in range(1, k_folds + 1):
            model = tf.keras.models.load_model(f"files/model_fold{fold}.h5")
            models.append(model)

    # Initialize arrays to store results from each fold
    fold_predictions = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)

        # Predictions for each fold
        fold_preds = []
        for model in models:
            y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
            fold_preds.append(y_pred)

        # Aggregate results from each fold
        y_pred_agg = np.mean(fold_preds, axis=0)

        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred_agg) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results_crossvalidation/{i}_fold{model}.png", image)

    print("All images generated.")

    # Calculate overall metrics from cross-validation
    metrics_per_fold = []
    for fold in range(1, k_folds + 1):
        data = pd.read_csv(f"files/data_fold{fold}_lr{learning_rate}.csv")
        metrics_per_fold.append(data)

    metrics_overall = pd.concat(metrics_per_fold).groupby(level=0).mean()
    metrics_overall.to_csv("files/data_overall_test.csv", index=False)

    print("Overall metrics calculated.")
