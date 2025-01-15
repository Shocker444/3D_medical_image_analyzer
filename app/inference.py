import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import numpy as np
import streamlit as st
import tqdm
import keras
import tensorflow as tf
import segmentation_models as sm
from preprocess import normalize_vol

x_axis = True
y_axis = True
z_axis = False
image_size = (128, 128)

preprocess_input = sm.get_preprocessing('resnet18')
loaded_model = keras.models.load_model('../unet_model_2.keras')

def percentage_func(part, whole):
    return int(100 * (float(part) / float(whole)))


def predictVol(volume):
    dimz, dimx, dimy = volume.shape
    outimgx = np.zeros((dimz, dimx, dimy))
    outimgy = np.zeros((dimz, dimx, dimy))
    outimgz = np.zeros((dimz, dimx, dimy))

    volume = normalize_vol(volume)
    volume = volume * 255

    count = 0
    progress_counter = 0
    total_slices = dimx + dimy
    progress_bar = st.progress(0)
    status_text = st.empty()

    if x_axis:
        print("[+] slicing x axis")
        count += 1
        for i in tqdm.tqdm(range(dimx), desc="Slicing x axis"):
            slice_2d = volume[:, i, :][..., tf.newaxis]
            slice_2d = tf.image.resize(slice_2d, image_size)
            slice_2d = preprocess_input(slice_2d)
            pred = loaded_model(slice_2d[tf.newaxis])
            pred = tf.where(pred > 0.5, 1, 0)
            pred = tf.image.resize(pred, (dimz, dimy))
            outimgx[:, i, :] = pred[0, :, :, 0]
            progress_counter += 1
            percentage_com = percentage_func(progress_counter, total_slices)
            progress_bar.progress(percentage_com)
            status_text.text(f"Progress: {percentage_com}%")


    if y_axis:
        print("[+] slicing y axis")
        count += 1
        for i in tqdm.tqdm(range(dimy), desc="Slicing y axis"):
            slice_2d = volume[:, :, i][..., tf.newaxis]
            slice_2d = tf.image.resize(slice_2d, image_size)
            slice_2d = preprocess_input(slice_2d)
            pred = loaded_model(slice_2d[tf.newaxis])
            pred = tf.where(pred > 0.5, 1, 0)
            pred = tf.image.resize(pred, (dimz, dimx))
            outimgy[:, :, i] = pred[0, :, :, 0]
            progress_counter += 1
            percentage_com = percentage_func(progress_counter, total_slices)
            progress_bar.progress(percentage_com)
            status_text.text(f"Progress: {percentage_com}%")


    if z_axis:
        print("[+] slicing z axis")
        count += 1
        for i in tqdm.tqdm(range(dimz), desc="Slicing z axis"):
            volume = normalize_vol(volume)
            slice_2d = volume[i, :, :][..., tf.newaxis]
            slice_2d = tf.image.resize(slice_2d, image_size)
            slice_2d = preprocess_input(slice_2d)
            pred = loaded_model(slice_2d[tf.newaxis])
            pred = tf.where(pred > 0.5, 1, 0)
            pred = tf.image.resize(pred, (dimx, dimy))
            outimgz[i, :, :] = pred[0, :, :, 0]
            progress_counter += 1
            percentage_com = percentage_func(progress_counter, total_slices)
            progress_bar.progress(percentage_com)
            status_text.text(f"Progress: {percentage_com}%")

    outimg = (outimgx + outimgy + outimgz) / count

    return outimg