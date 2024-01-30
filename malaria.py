import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier



# Load model
def main():
    st.set_page_config(
        page_title="Malaria Prediction App",
        page_icon="🦟",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    # Define image paths for each page
    home_image = "https://media.giphy.com/media/jofiBfcQjNmRgFjiHx/giphy.gif"
    malaria_prediction = "./images/dataml.jpg"
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
# Load the trained Random Forest model
def load_lasso_model():
    with open('lasso_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to predict malaria infestation
def predict_malaria_infestation(model, user_inputs):
    # Replace with your actual feature names
    selected_features = ['falciparum_present', 'malariae_present', 'rapid_test_result', 'ovale_present',
                         'region_of_residence', 'children_hemoglobin_elig', 'hemoglobin_level_g_dl',
                         'cluster_altitude', 'county', 'sample_weight', 'malaria_endemicity_zone',
                         'children_under_five', 'sex_of_member', 'number_of_children_resident',
                         'hemoglobin_level_adjusted_for_altitude_g_dl', 'childs_age_in_days_country_specific',
                         'number_of_household_members', 'has_bed', 'member_own_bicycle', 'anemia_level',
                         'childs_age_in_months_country_specific_hml16a', 'has_solar_panel', 'flag_age',
                         'net_from_antenatal_immunization_visit', 'owns_agric_land', 'owns_pigs',
                         'type_of_cooking_fuel_energy', 'member_has_bank_account', 'household_relationship_structure',
                         'has_mobile', 'owns_computer', 'has_dvd_player', 'date_measured_month',
                         'wealth_index_urban_rural', 'blood_smear_bar_code', 'has_table', 'has_watch',
                         'type_of_place', 'owns_sheep', 'owns_donkeys', 'member_own_motorcycle',
                         'number_of_persons_slept_under_net', 'owns_cattle', 'used_mosquito_relellent_spray',
                         'female_int_eligibility', 'mosquito_bed_net_designation_number',
                         'net_observed_by_interviewer', 'owns_livestock', 'use_phone_for_finc_transactions',
                         'children_under_mosquito_net', 'has_cd_player', 'corr_age', 'age_of_member',
                         'own_television', 'caretaker_line_number', 'read_consent_statement_hemoglobin',
                         'main_floor_material', 'someone_slept_under_net_last_night', 'wealth_index_factor',
                         'has_microwave', 'owns_goats', 'child_age_in_months', 'line_number_of_person_slept_in_net',
                         'childs_age_in_days', 'main_source_drink_water', 'has_cupboard', 'cooking_fuel_type',
                         'main_wall_material', 'shares_toilet', 'has_mosquito_net', 'months_ago_net_obtained',
                         'wealth_index_comb', 'owns_mules', 'no_sleep_rooms', 'owns_animaldrawn_cart',
                         'type_toilet_facility', 'type_of_cooking_device', 'own_radio', 'owns_poultry',
                         'toilet_location', 'owns_horses', 'owns_cows_bulls', 'malaria_measurement_result',
                         'number_of_mosquito_nets', 'number_of_mosquito_nets_specific', 'has_chair', 'has_sofa',
                         'sex_head_household', 'slept_last_night', 'result_of_measurement_hemoglobin', 'has_clock',
                         'slept_llin_net', 'time_to_water_source', 'own_refrigerator', 'bed_net_type',
                         'household_has_telephone', 'main_roof_material', 'location_of_source_for_water',
                         'childs_age_in_months', 'wealth_index_factor_urban_rural', 'age_household_head',
                         'usual_resident', 'own_electricity', 'sex', 'brand_of_net', 'owns_boat_wmotor',
                         'insecticide_treated_net', 'individual_file_pregnancy_status', 'member_own_car',
                         'result_of_household_interview', 'ever_married', 'household_hemoglobin_measurements',
                         'slept_under_net', 'vivax_present', 'year_of_data_collection'] 

    X = pd.DataFrame([user_inputs], columns=selected_features)
    prediction_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
    return prediction_proba.item()  # Extract scalar value

# Function to get user inputs for selected features
def get_user_inputs(selected_features):
    user_inputs = {}

    # Create input fields for each selected feature
    for feature in selected_features:
        user_inputs[feature] = st.sidebar.number_input(f"Enter {feature}", min_value=0, max_value=100, value=0)

    return user_inputs

# Main app
def prediction(malaria_prediction):
    st.title("Malaria Infestation Prediction App")
    
    # header image
    # st.image(malaria_prediction, width=800)
    
    # Load the trained Random Forest model
    lasso_model = load_lasso_model()

    # Streamlit app layout
    st.markdown("""
        This app predicts the possibility of malaria infestation based on user inputs.
        Use the input fields to provide information, and click 'Predict' to get the results.
    """)

    # Sidebar with user input fields
    st.sidebar.header('User Input Features')

    # Get user inputs for selected features
    selected_features = ['falciparum_present', 'malariae_present', 'rapid_test_result', 'ovale_present',
                         'region_of_residence', 'children_hemoglobin_elig', 'hemoglobin_level_g_dl',
                         'cluster_altitude', 'county', 'sample_weight', 'malaria_endemicity_zone',
                         'children_under_five', 'sex_of_member', 'number_of_children_resident',
                         'hemoglobin_level_adjusted_for_altitude_g_dl', 'childs_age_in_days_country_specific',
                         'number_of_household_members', 'has_bed', 'member_own_bicycle', 'anemia_level',
                         'childs_age_in_months_country_specific_hml16a', 'has_solar_panel', 'flag_age',
                         'net_from_antenatal_immunization_visit', 'owns_agric_land', 'owns_pigs',
                         'type_of_cooking_fuel_energy', 'member_has_bank_account', 'household_relationship_structure',
                         'has_mobile', 'owns_computer', 'has_dvd_player', 'date_measured_month',
                         'wealth_index_urban_rural', 'blood_smear_bar_code', 'has_table', 'has_watch',
                         'type_of_place', 'owns_sheep', 'owns_donkeys', 'member_own_motorcycle',
                         'number_of_persons_slept_under_net', 'owns_cattle', 'used_mosquito_relellent_spray',
                         'female_int_eligibility', 'mosquito_bed_net_designation_number',
                         'net_observed_by_interviewer', 'owns_livestock', 'use_phone_for_finc_transactions',
                         'children_under_mosquito_net', 'has_cd_player', 'corr_age', 'age_of_member',
                         'own_television', 'caretaker_line_number', 'read_consent_statement_hemoglobin',
                         'main_floor_material', 'someone_slept_under_net_last_night', 'wealth_index_factor',
                         'has_microwave', 'owns_goats', 'child_age_in_months', 'line_number_of_person_slept_in_net',
                         'childs_age_in_days', 'main_source_drink_water', 'has_cupboard', 'cooking_fuel_type',
                         'main_wall_material', 'shares_toilet', 'has_mosquito_net', 'months_ago_net_obtained',
                         'wealth_index_comb', 'owns_mules', 'no_sleep_rooms', 'owns_animaldrawn_cart',
                         'type_toilet_facility', 'type_of_cooking_device', 'own_radio', 'owns_poultry',
                         'toilet_location', 'owns_horses', 'owns_cows_bulls', 'malaria_measurement_result',
                         'number_of_mosquito_nets', 'number_of_mosquito_nets_specific', 'has_chair', 'has_sofa',
                         'sex_head_household', 'slept_last_night', 'result_of_measurement_hemoglobin', 'has_clock',
                         'slept_llin_net', 'time_to_water_source', 'own_refrigerator', 'bed_net_type',
                         'household_has_telephone', 'main_roof_material', 'location_of_source_for_water',
                         'childs_age_in_months', 'wealth_index_factor_urban_rural', 'age_household_head',
                         'usual_resident', 'own_electricity', 'sex', 'brand_of_net', 'owns_boat_wmotor',
                         'insecticide_treated_net', 'individual_file_pregnancy_status', 'member_own_car',
                         'result_of_household_interview', 'ever_married', 'household_hemoglobin_measurements',
                         'slept_under_net', 'vivax_present', 'year_of_data_collection'] 


    user_inputs = get_user_inputs(selected_features)
# Check if user_inputs contains all features expected by the model
    if len(user_inputs) != 115:
        raise ValueError("Input data does not contain all features expected by the model.")
    
    # Predict button
    if st.sidebar.button('Predict', key='predict_button'):
        with st.spinner('Predicting...'):
            # Make model prediction
            prediction_proba = predict_malaria_infestation(lasso_model, user_inputs)

            # Display prediction result
            st.subheader('Prediction Result')
            result_text = "Positive" if prediction_proba > 0.5 else "Negative"
            st.write(f'Prediction: **{result_text}**')

            # Display prediction probability
            st.subheader('Prediction Probability')
            st.write(f'The predicted probability of malaria infestation is: **{prediction_proba:.2%}**')

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