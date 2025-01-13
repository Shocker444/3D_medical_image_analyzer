import plotly.graph_objects as go
from skimage import measure
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def plot_mesh(volume, threshold=0.5):
    verts, faces, _, _ = measure.marching_cubes(volume, level=threshold)
    verts_t, faces_t = np.transpose(verts), np.transpose(faces)
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts_t[0],
            y=verts_t[1],
            z=verts_t[2],
            i=faces_t[0],
            j=faces_t[1],
            k=faces_t[2])
    ])

    return fig

def plot_slices(img):
    # Axis selection
    axis = st.radio("Select viewing axis", options=[0, 1, 2], format_func=lambda x: f"Axis {x}", horizontal=True)

    # Slider for slice index
    max_index = img.shape[axis] - 1
    slice_idx = st.slider(f"Select slice along axis {axis}", 0, max_index, max_index // 2)

    # Extract the slice
    if axis == 0:
        slice_data = img[slice_idx, :, :]
    elif axis == 1:
        slice_data = img[:, slice_idx, :]
    else:
        slice_data = img[:, :, slice_idx]

    # Display the slice
    st.write(f"Showing slice {slice_idx} along axis {axis}")
    plt.figure(figsize=(4, 4))
    plt.imshow(slice_data.T, cmap="gray", origin="lower")
    plt.axis('off')

    return plt