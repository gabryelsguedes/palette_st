import streamlit as st
import palette

st.title('Palette generator')
uploaded_image = st.file_uploader(
    'Upload a image', 
    type=['jpg', 'jpeg']
)
col1, col2 = st.columns(2)
if uploaded_image:
    with col1:
        st.image(uploaded_image)
    with col2:
        n_clusters = st.slider("Number of colors", 2, 8, 5)
        if st.button("Generate palette"):
            with st.spinner('Generating palette...'):         
                cores, cores_hex = palette.get(uploaded_image, n_clusters)
            figura = palette.show(cores)
            st.pyplot(figura)
            st.code(f"{cores_hex}")
            
            btn = st.download_button(
                label = "Download palette",
                data = palette.save(figura),
                file_name = "pallete.png",
                mime = "image/png")
                