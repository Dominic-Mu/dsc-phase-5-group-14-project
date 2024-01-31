import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Load model
def main():
    st.set_page_config(
        page_title="Malaria Prevalence App",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    # Define image paths for each page
    home_image = "https://media.giphy.com/media/jofiBfcQjNmRgFjiHx/giphy.gif"
    malaria_prediction = "../images/dataml.jpg"
    about_us_image = "https://media.giphy.com/media/9yssegcqq1WDlPKdP4/giphy.gif"

    # Create a sidebar for navigation
    st.sidebar.title("Navigation Menu")
    page = st.sidebar.radio("Go to", ["Home", "Malaria Prediction", "About Us"])

    # Display the selected page
    if page == "Home":
        home_page(home_image)
    elif page == "Malaria Prediction":
        prediction(malaria_prediction)
    elif page == "About Us":
        about_us_page(about_us_image)
    

# PAGE 1

def home_page(image_path):
    st.image(image_path, width=800)
    st.title('Malaria Prevalence App')
    st.markdown("""
    ## Business Overview

    The KDHS covers a broad range of topics, including fertility, family planning, maternal and child health, nutrition, malaria, HIV/AIDS, and other health-related issues.The survey results are crucial for monitoring progress towards health-related Sustainable Development Goals (SDGs) and informing policies and programs aimed at addressing public health challenges.

    ## Problem Statement

    To integrate Machine Learning techniques into the analysis of KDHS-MIS dataset. This research aims to contribute to evidence-based decision-making and enhance the effectiveness of malaria control strategies in Kenya, offering a transformative approach to understanding and combatting malaria in the different regions.
    """, unsafe_allow_html=True)
    
# PAGE 2

# Placeholder function for your predictive model
def predict_malaria_infestation(user_inputs):
    # Replace with your own model logic
    # This function currently returns random data for demonstration purposes
    prediction = np.random.choice([0, 1], size=len(user_inputs))
    prediction_proba = np.random.rand(len(user_inputs))
    return prediction, prediction_proba


def prediction(image_path):
    # st.image(image_path, width=800)
    st.title("Malaria Prevalence App")

    # Streamlit app layout

    st.markdown("""
        This app predicts the possibility of malaria infestation based on user inputs.
        Use the input fields to provide information, and click 'Predict' to get the results.
    """)

    
    # Sidebar with user input fields
    st.header('User Input Features')

    # Create input fields for each selected feature
    user_inputs = {}
    selected_features = ['number_of_household_members', 'number_of_children_resident', 'region_of_residence']
    for feature in selected_features:
        user_inputs[feature] = st.number_input(feature, min_value=0, max_value=100, value=0)
        
    
    st.selectbox('Select a Region', ['red','orange','green','blue','violet'])

    # Predict button
    if st.button('Predict'):
            with st.spinner('Predicting...'):
            # Convert user input to DataFrame
                user_data = pd.DataFrame([user_inputs])

            # Placeholder for your model prediction logic
                prediction, prediction_proba = predict_malaria_infestation(user_data)

            # Display prediction result
                st.subheader('Prediction Result')
                st.write(f'Possibility of Malaria Infestation: {prediction_proba[0]:.2%}')
                st.write(f'Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')

    # Additional content, if needed
    st.markdown("Feel free to add more content or explanations here.")

# PAGE 3
def about_us_page(image_path):
    st.image(image_path, width=800)
    st.title('About Us')

    st.subheader('Meet the Team')
    st.write("""
        We are The Group 14 Project students from DSC PartTime Moringa School course, working on our [Malaria Prevalence] Capstone Project.
    """)

    team_members = {
        "Alpha Guya":"mailto:alpha.guya@student.moringaschool.com", 
        "Ben Ochoro":"mailto:ben.ochoro@student.moringaschool.com", 
        "Caleb Ochieng":"mailto:caleb.ochieng@student.moringaschool.com", 
        "Christine Mukiri":"mailto:christine.mukiri@student.moringaschool.com", 
        "Dominic Muli":"mailto:dominic.muli@student.moringaschool.com", 
        "Frank Mandele":"mailto:frank.mandele@student.moringaschool.com", 
        "Jacquiline Tulinye":"mailto:jacquiline.tulinye@student.moringaschool.com",
        "Lesley Wanjiku":"mailto:lesley.wanjiku@student.moringaschool.com"
    }

    for name, link in team_members.items():
        st.markdown(f"- [{name}]({link})")

if __name__ == "__main__":
    main()
