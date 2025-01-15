import os

import streamlit as st
from skimage import io
import numpy as np
import nibabel as nib

from plot_utils import  plot_slices
from inference import predictVol

st.set_page_config(layout='wide')


st.title('3D medical image analyzer')

if "generated_masks" not in st.session_state:
    st.session_state.generated_masks = None

if "original_volume" not in st.session_state:
    st.session_state.original_volume = None

if "volume_key" not in st.session_state:
    st.session_state.volume_key = None  

st.write('This application is a 3D medical image analyzer that uses a deep learning model to predict the presence of a tumor or other diseases in a 3D medical image.')

uploaded_file = st.file_uploader("Upload a 3D medical image", type=["nii", "nii.gz"])

     
if uploaded_file is not None:

    file_key = uploaded_file.name
    if file_key != st.session_state.volume_key:
        st.session_state.volume_key = file_key
        st.session_state.original_volume = None
        st.session_state.generated_masks = None

    temp_dir = 'temp_vol'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    img = io.imread(file_path)

    st.session_state.original_volume = img

    if st.button('Generate liver segmentation mask', type='primary'):
        predicted_mask = predictVol(img)
        st.session_state.generated_masks = predicted_mask
        st.success('Mask generated successfully!')

    if st.session_state.original_volume is not None:
        img = st.session_state.original_volume

        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = plot_slices(img, 'Vol_radio', 'Vol_slider', 'Original Volume')
            st.pyplot(fig)

        with col2:
            if st.session_state.generated_masks is not None:
                masks = st.session_state.generated_masks
                fig2 = plot_slices(masks, 'Mask_radio', 'Mask_slider', 'Mask')
                st.pyplot(fig2)

                if st.button('Save'):
                    # Define an affine transformation (identity matrix in this case)
                    affine = np.eye(4)

                    # Create a NIfTI image
                    nifti_image = nib.Nifti1Image(masks, affine)

                    # Save the NIfTI image to a file
                    nib.save(nifti_image, f'{temp_dir}/output_volume.nii')

                    print(f"3D volume saved as output_volume.nii to {temp_dir}")
            else:
                st.warning('Generate the mask in order to visualize it')
        

    


