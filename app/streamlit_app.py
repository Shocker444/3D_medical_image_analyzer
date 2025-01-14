import os

import streamlit as st
from skimage import io

from plot_utils import  plot_slices
from inference import predictVol

st.set_page_config(layout='wide')


st.title('3D medical image analyzer')

st.write('This application is a 3D medical image analyzer that uses a deep learning model to predict the presence of a tumor or other diseases in a 3D medical image.')

uploaded_file = st.file_uploader("Upload a 3D medical image", type=["nii", "nii.gz"])

if uploaded_file is not None:

    temp_dir = 'temp_vol'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    img = io.imread(file_path)
    st.write('[INFO] Generating liver segmentation mask...')
    predicted_mask = predictVol(img)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = plot_slices(img, 'Vol_radio', 'Vol_slider')
        st.pyplot(fig)

    with col2:
        fig2 = plot_slices(predicted_mask, 'Mask_radio', 'Mask_slider')
        st.pyplot(fig2)

    

    


