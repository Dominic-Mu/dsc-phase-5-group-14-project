# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from pydub import AudioSegment
# from pydub.playback import play
# from skimage import exposure, io
# import time

# # Load model
# def main():
#     st.set_page_config(
#         page_title="Distracted Driver APP",
#         page_icon="🚗",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     # # Header photo
#     # header_image = "Drive Safe.jpg"
#     # st.image(header_image, use_column_width=True)

#     # Define image paths for each page
#     home_image = "./images/01.jpg"
#     malaria_prediction = "./images/01.jpg"
#     about_us_image = "./images/01.jpg"

#     # Create a sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Home", "Image Predictor", "About Us"])

#     # Display the selected page
#     if page == "Home":
#         home_page(home_image)
#     elif page == "Prediction":
#         prediction(malaria_prediction)
#     elif page == "About Us":
#         about_us_page(about_us_image)

# def home_page(image_path):
#     st.image(image_path, use_column_width=True)
#     st.title('Distracted Driver App')
#     st.markdown("""
#     ## Business Overview

#     You're probably wondering why this APP? Well, road safety remains a critical concern around the world, with distracted driving claimed as being a leading cause of accidents. Distracted driving accounts for at least **9%** of annual car accidents in USA and is the leading cause of accidents worldwide.  

#     According to an NTSA report on accidents in 2023, **1,072** people were killed on our roads, with the main causes being drunk driving, speeding and distracted driving. In Kenya we already have measures in place to tackle the first two: Alcoblow for drunk-driving, speed guns and speed governors for speeding. There seems to be nothing in place to tackle the third cause and that is where our project comes in.  

#     This project aims to leverage computer vision and machine learning techniques to develop a system capable of detecting distracted drivers in real-time, contributing to enhanced road safety measures.

#     ## Problem Statement

#     Distracted driving poses significant risks, including accidents, injuries, and fatalities. Identifying and mitigating instances of distraction while driving is crucial to reducing road accidents.  

#     The ballooning of car insurance claims led Directline Insurance, Kenya, to engage us in this project, with a vision to lower the rising claims from their customers.
#     """, unsafe_allow_html=True)


# # def play_sound(sound_file="Distracted_driver_alert.aac"):
# #     st.markdown(f'<audio src="{sound_file}" autoplay="autoplay" controls="controls"></audio>', unsafe_allow_html=True)
# # def play_sound(sound_file):
# #     # Check if the predicted class is not c1
# #     if categories[predicted_class] != 'c1':
# #         # Play sound if the predicted class is not c1
# #         st.markdown(f'<audio src="{sound_file}" autoplay="autoplay" controls="controls"></audio>', unsafe_allow_html=True)
    

# def prediction(image_path):
#     st.image(image_path, use_column_width=True)
#     st.title("Malaria Prevalence App")



# def about_us_page(image_path):
#     st.image(image_path, use_column_width=True)
#     st.title('About Us')

#     st.subheader('Meet the Team')
#     st.write("""
#         We are all data science students from Flat Iron Moringa School, working on our capstone project.
#     """)

#     # team_members = {
#     #     "Leonard Gachimu": "https://github.com/leogachimu",
#     #     "Rowlandson Kariuki": "https://github.com/RowlandsonK",
#     #     "Francis Njenga": "https://github.com/GaturaN",
#     #     "Mourine Mwangi": "https://github.com/Mourinem97",
#     #     "Khadija Omar": "https://github.com/Khadija-Omar",
#     #     "Victor Mawira": "https://github.com/Victormawira",
#     #     "Onesphoro Kibunja": "https://github.com/Ones-Muiru"
#     # }

#     # for name, link in team_members.items():
#     #     st.markdown(f"- [{name}]({link})")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your data
@st.cache  # This decorator tells Streamlit to cache the data
def load_data():
    # Load your CSV files or any other data source here
    data = pd.read_csv('./Output/modelling_data.csv')
    numeric_columns = data.select_dtypes(include='number').columns
    # Preprocess your data, perform train-test split, scaling, etc.
    # Example assuming your data has features (X) and labels (y)
    X = data[numeric_columns].drop(columns=['final_blood_smear_test'], axis=1)
    y = data['final_blood_smear_test']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform feature scaling if necessary
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Define your Streamlit app
def main():
    st.title("Logistic Regression Model Deployment")

    # Load your data
    X_train_scaled, X_test_scaled, y_train, y_test = load_data()

    # Creating a logistic regression model
    logistic_model = LogisticRegression(random_state=42)

    # Training the model
    logistic_model.fit(X_train_scaled, y_train)

    # Making predictions on the test set
    y_pred = logistic_model.predict(X_test_scaled)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Printing evaluation metrics
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:", conf_matrix)
    st.write("Classification Report:", class_report)

if __name__ == "__main__":
    main()
