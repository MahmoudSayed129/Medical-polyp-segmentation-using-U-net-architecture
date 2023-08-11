
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from train import iou

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
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("files/model.h5")

    model.evaluate(test_dataset, steps=test_steps)
    logits = model.predict(test_dataset, steps=test_steps)
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_new = None
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        if(i == 24):
            y_new = (model.predict(np.expand_dims(x, axis=0))[0] > 0.15).astype(np.uint8)
            binary_mask_cleaned = cv2.morphologyEx(y_new, cv2.MORPH_OPEN, 
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)))
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        if (i == 24) :
            all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0,white_line,
            mask_parse(y_new) * 255.0,white_line,
            mask_parse(binary_mask_cleaned) * 255.0,white_line,
            mask_parse(model.predict(np.expand_dims(x, axis=0))[0]) * 255.0
            ]
            print(iou(y,y_pred))
            print(iou(y,binary_mask_cleaned))
        else :
            all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0,white_line,
            mask_parse(model.predict(np.expand_dims(x, axis=0))[0]) * 255.0
            ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results_grayscale/{i}.png", image)