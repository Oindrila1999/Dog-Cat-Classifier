import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def main():
    htmltemp='''<div style="background_color:black;padding:10px;">
                <h2 style="color:white;text-align:center;">Dog Cat Classifier</h2>
                </div>'''
    st.markdown(htmltemp,unsafe_allow_html=True)
    try:
        uploaded_file=st.file_uploader('choose a image',type=['jpg','jpeg','png'])
        if uploaded_file is not None:
            target_size=(64,64)
            uploaded=Image.open(uploaded_file)
            model=load_model('dog cat classifier.h5')
            st.image(uploaded,caption='Uploaded Image',use_column_width=True)
            if st.button('Predict'):
                img=uploaded.resize(target_size)
                img=(np.array(img)/255)
                img=np.expand_dims(img,axis=0)
                result=model.predict(img)[0][0]>0.5
                if result:
                    st.write('The image is of a Dog')
                else:
                    st.write("The image is of a Cat")
    except:
        st.write("corrupted image,Please try again later")
    
if __name__=='__main__':
    main()