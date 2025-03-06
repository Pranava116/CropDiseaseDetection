import streamlit as st
import tensorflow as tf
import numpy as np 
import time


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 

st.set_page_config(initial_sidebar_state="collapsed")
with st.sidebar:
    st.title('Dashboard')
    


app_mode = st.sidebar.selectbox("Select Page",["Disease Recognition","Home"])

if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    
    ## How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ## Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ## About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)


elif(app_mode=="Disease Recognition"):
    st.markdown("<h1 style='text-align: center;'>Disease Recognition</h1>", unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose an Image:")

    if(st.button("Predict")):  
        st.write("Our Prediction")
        st.image(test_image,width=4,use_column_width=True)
        result_index = model_prediction(test_image)

        class_name = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
                    'Blueberry healthy', 'Cherry (including_sour) Powdery mildew', 
                    'Cherry (including_sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 
                    'Corn (maize) Common rust', 'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 
                    'Grape Black rot', 'Grape Esca_(Black_Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 
                    'Grape healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot',
                    'Peach healthy', 'Pepper,bell Bacterial spot', 'Pepper,bell healthy', 
                    'Potato Early blight', 'Potato Late blight', 'Potatohealthy', 
                    'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 
                    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 
                    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 
                    'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 
                    'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus',
                      'Tomato healthy']


        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))