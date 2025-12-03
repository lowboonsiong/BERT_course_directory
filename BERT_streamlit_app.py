import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd # <-- Import pandas for CSV handling

# Importing keras from tensorflow, as the user code implies this structure
try:
    from tensorflow import keras 
except ImportError:
    st.error("TensorFlow and Keras are required. Please install with: pip install tensorflow")
    # Exit early if TensorFlow is not available, cannot run the core logic.
    st.stop()


# --- Configuration and Caching ---

# Define the paths as provided by the user
MODEL_PATH = 'data/model/bert_regression_model.keras'
SCALER_PATH = 'data/model/bert_model_scaler.pkl'

# Use st.cache_resource to load the large model and scaler only once
@st.cache_resource
def load_model_and_scaler():
    """Loads the BERT model and the scaler object."""
    loaded_model = None
    loaded_scaler = None
    
    # 1. Load Scaler
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found: {SCALER_PATH}. Please check your file structure.")
    else:
        try:
            with open(SCALER_PATH, 'rb') as file:
                loaded_scaler = pickle.load(file)
            st.success("Data Scaler loaded successfully.")
        except Exception as e:
            st.error(f"Error loading scaler from {SCALER_PATH}: {e}")

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}. Please check your file structure.")
    else:
        try:
            # We assume the model loads correctly using standard Keras load function
            loaded_model = keras.models.load_model(MODEL_PATH)
            st.success("BERT Regression Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model from {MODEL_PATH}: {e}. Ensure you have the correct Keras/TensorFlow version.")
            
    return loaded_model, loaded_scaler

# --- Prediction Function (Original, now also used by the batch function) ---

def predict_new_course_students(model, scaler, course_data):
    """
    Formates the input data, predicts the scaled value, and inverse-transforms it.
    Can handle a single dictionary (original use) or a single row (for batch use).
    """
    # Create the single input text string
    # Assumes 'course_data' is a dictionary or a pandas Series (a row) with required keys/columns
    new_input_text = (
        "TITLE: " + str(course_data['coursetitle']) + ". " +
        "SCHOOL: " + str(course_data['ihl_status']) + ". " +
        "COURSE RATING: " + str(course_data['courseratings_stars']) + ". " +
        "JOB RATING: " + str(course_data['jobcareer_impact_stars']) + ". " +
        "FEE: $" + str(course_data['full_course_fee']) + ". " +
        "FEE AFTER SUBSIDIES (IF ANY): $ " + str(course_data['course_fee_after_subsidies']) + ". " +
        "TRAINING HOUR: " + str(course_data['number_of_hours']) + ". " +
        "TRAINING COMMITMENT: " + str(course_data['training_commitment']) + ". " +
        "DESCRIPTIONS: " + str(course_data['about_this_course']) + ". " + 
        "SKILLS: " + str(course_data['what_you_learn'])
    )

    # Convert the single string into a NumPy array of strings for the model
    X_new = np.array([new_input_text]) 
    
    # Check if the model is loaded before predicting
    if model is None:
        return 0 # Return a default prediction if model is missing

    y_pred_scaled = model.predict(X_new, verbose=0)
    
    # Check if the scaler is loaded before inverse transform
    if scaler is None:
        return int(round(y_pred_scaled[0][0])) # Return raw scaled value rounded if scaler is missing
        
    # Inverse Transform and Finalize ---
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    # Round to the nearest whole number and ensure it's not negative
    predicted_students = max(0, int(round(y_pred[0][0])))
    
    return predicted_students

# --- NEW: Batch Prediction Function ---
@st.cache_data
def predict_batch(_model, _scaler, df):
    """
    Performs batch prediction on a pandas DataFrame.
    Note: Arguments prefixed with '_' are excluded from Streamlit's caching hash.
    """
    if _model is None or _scaler is None:
        st.error("Model or Scaler is not loaded, unable to run batch prediction.")
        return df

    # 1. Create the input text array from all rows in the DataFrame
    df['input_text'] = df.apply(
        lambda row: (
            "TITLE: " + str(row['coursetitle']) + ". " +
            "SCHOOL: " + str(row['ihl_status']) + ". " +
            "COURSE RATING: " + str(row['courseratings_stars']) + ". " +
            "JOB RATING: " + str(row['jobcareer_impact_stars']) + ". " +
            "FEE: $" + str(row['full_course_fee']) + ". " +
            "FEE AFTER SUBSIDIES (IF ANY): $ " + str(row['course_fee_after_subsidies']) + ". " +
            "TRAINING HOUR: " + str(row['number_of_hours']) + ". " +
            "TRAINING COMMITMENT: " + str(row['training_commitment']) + ". " +
            "DESCRIPTIONS: " + str(row['about_this_course']) + ". " + 
            "SKILLS: " + str(row['what_you_learn'])
        ), axis=1
    )
    
    X_batch = np.array(df['input_text'].tolist())

    # 2. Perform Batch Prediction
    with st.spinner(f'Running BERT inference for {len(df)} records...'):
        # Use the underscored arguments for the prediction logic
        y_pred_scaled = _model.predict(X_batch, verbose=0)
    
    # 3. Inverse Transform and Finalize
    y_pred = _scaler.inverse_transform(y_pred_scaled)
    
    # Extract the prediction, round, and ensure non-negativity
    predicted_students = np.maximum(0, np.round(y_pred).astype(int))
    
    # 4. Add the prediction to the DataFrame
    df['predicted_students'] = predicted_students.flatten()
    
    # Remove the intermediate text column
    df = df.drop(columns=['input_text'])
    
    return df

# --- Streamlit Application Layout ---

st.set_page_config(page_title="BERT Course Enrollment Predictor", layout="wide")

st.title("Course Enrollment Prediction using Natural Language Processing & Understanding BERT Model")
st.markdown("Enter the course details below for single prediction or upload a CSV for mass prediction.")

# Load the resources upfront
loaded_model, loaded_scaler = load_model_and_scaler()

# --- Tab Layout for Single vs. Batch Prediction ---
tab1, tab2 = st.tabs(["**Single Prediction**", "**Mass Prediction (CSV Upload)**"])

# --- TAB 1: Single Prediction (Original Logic) ---
with tab1:
    with st.container(border=True):
        st.subheader("Course Detail")
        col1, col2 = st.columns(2)
        
        with col1:
            coursetitle = st.text_input("Course Title", "Engineering Ethics")
            
        with col2:
            ihl_status = st.selectbox("Institutes of Higher Learning (IHL)", ('IHL', 'NON-IHL'), index=0)

        # --- Numerical Inputs ---
        col3, col4 = st.columns(2)

        with col3:
            full_course_fee = st.number_input(
                "Full Course Fee ($)", 
                min_value=0.0, value=1461.0, step=10.0, format="%.2f"
            )
        with col4:
            course_fee_after_subsidies = st.number_input(
                "Fee After Subsidies ($)", 
                min_value=0.0, value=438.3, step=10.0, format="%.2f"
            ) 
            
        col7, col8 = st.columns(2) 
        with col7:
            number_of_hours = st.number_input(
                "Total Training Hours",
                min_value=1, value=36, step=1
            )
        
        with col8:    
            training_commitment = st.selectbox(
                "Training Commitment", 
                ('Full Time', 'Part Time'), index=0
            )

        col5, col6 = st.columns(2)
        with col5:
            courseratings_stars = st.slider(
                "Course Ratings", 
                min_value=0.0, max_value=5.0, value=4.5, step=0.1
            )
        with col6:
            jobcareer_impact_stars = st.slider(
                "Job/Career Impact Rating", 
                min_value=0.0, max_value=5.0, value=4.0, step=0.1
            )
        
        # --- Textual Inputs (Full Width) ---
        about_this_course = st.text_area(
            "About This Course (Descriptions)", 
            "Nowadays, technology has a pervasive and profound impact on everyday life, where engineers play a crucial role in its development. It is therefore extremely important that engineers understand the importance of safety, health, and welfare of the public, when developing this technology. They must be morally committed and equipped to tackle any ethical issues they may encounter. This course aims at training the student to reach such a status through discussions and typical case studies where real examples are thoroughly discussed.",
            height=150
        )

        what_you_learn = st.text_area(
            "What You Will Learn (Skills)", 
            "Ethics and Professionalism. Moral Choices and Ethical Dilemmas. Codes of Ethics. Moral Frameworks. Ethics as Social Experimentation. Safety and Risk. Assessing and Reducing Risk. Workplace Responsibilities and Rights. Truth and Truthfulness. Computer Ethics. Environmental Ethics;",
            height=150
        )

    # --- Prediction Button and Logic for Single Input ---

    if st.button("Predict Student Enrollment", type="primary", use_container_width=True):
        
        if loaded_model is None or loaded_scaler is None:
            st.warning("Prediction cannot proceed because the model or scaler failed to load.")
        else:
            # 1. Collect all inputs into the dictionary format expected by the function
            new_course_features = {
                'coursetitle': coursetitle,
                'ihl_status': ihl_status,
                'courseratings_stars': courseratings_stars,
                'jobcareer_impact_stars': jobcareer_impact_stars,
                'full_course_fee': full_course_fee,
                'course_fee_after_subsidies': course_fee_after_subsidies,
                'number_of_hours': number_of_hours,
                'training_commitment': training_commitment,
                'about_this_course': about_this_course,
                'what_you_learn': what_you_learn
            }
            
            # 2. Perform Prediction
            with st.spinner('Running BERT inference and inverse scaling...'):
                # Note: Modified the original predict function call to only return the count
                predicted_count = predict_new_course_students(
                    loaded_model, loaded_scaler, new_course_features
                )

            # 3. Display Results
            st.success("✅ Prediction Complete")
            
            st.metric(
                label="Predicted Number of Students to Attend",
                value=f"{predicted_count:,}",
                delta="Estimated Enrollment Count"
            )

# --- TAB 2: Mass Prediction (New Logic) ---
with tab2:
    st.header("Upload CSV for Mass Prediction 📊")
    st.markdown("""
        Please ensure your CSV file contains the following columns exactly: 
        `coursetitle`, `ihl_status`, `courseratings_stars`, `jobcareer_impact_stars`, 
        `full_course_fee`, `course_fee_after_subsidies`, `number_of_hours`, 
        `training_commitment`, `about_this_course`, `what_you_learn`
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv"
    )

    if uploaded_file is not None:
        try:
            # Read the CSV file into a pandas DataFrame
            data_df = pd.read_csv(uploaded_file)
            st.info(f"Successfully loaded **{len(data_df)}** rows for prediction.")

            # Check if all required columns are present
            required_cols = [
                'coursetitle', 'ihl_status', 'courseratings_stars', 'jobcareer_impact_stars', 
                'full_course_fee', 'course_fee_after_subsidies', 'number_of_hours', 
                'training_commitment', 'about_this_course', 'what_you_learn'
            ]
            
            missing_cols = [col for col in required_cols if col not in data_df.columns]

            if missing_cols:
                st.error(f"The uploaded CSV is missing the following required columns: **{', '.join(missing_cols)}**")
            elif loaded_model is None or loaded_scaler is None:
                 st.warning("Cannot run mass prediction because the model or scaler failed to load.")
            else:
                # Run the batch prediction
                results_df = predict_batch(loaded_model, loaded_scaler, data_df.copy())
                
                st.success("✅ Mass Prediction Complete")
                
                # --- Display First 5 Rows ---
                st.subheader("First 5 Results")
                st.dataframe(results_df.head())
                
                # --- Create Download Button ---
                # Convert the resulting DataFrame to CSV for download
                csv_output = results_df.to_csv(index=False).encode('utf-8')
                
                st.subheader("Download Full Results")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name='mass_predictions_output.csv',
                    mime='text/csv',
                    type="primary",
                    help="Click to download the CSV file with the new 'predicted_students' column."
                )

        except Exception as e:
            st.error(f"An error occurred during file processing: {e}")