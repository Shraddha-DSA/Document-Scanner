import streamlit as st
import numpy as np
from PIL import Image
from model import scan_document   

st.title("📄 AI Document Scanner")

uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    img = np.array(image)

    org, warp, scanned = scan_document(img)

    if warp is None:
        st.error("Could not detect the document edges. Try another image.")
    else:
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(org, caption="Original", use_container_width=True)
        with col2:
            st.image(warp, caption="Warped Document", use_container_width=True)
        with col3:
            st.image(scanned, caption="Scanned Effect", use_container_width=True)

      
        result_img = Image.fromarray(scanned)
        st.download_button("Download Scanned Document",
                           data=result_img.tobytes(),
                           file_name="scanned.png",
                           mime="image/png")
