import streamlit as st
import pandas as pd
import joblib
import os

# Teams and cities lists
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Set the directory to the root directory's folder containing the model
model_dir = os.path.join(os.getcwd(), 'Model')
model_path = os.path.join(model_dir, 'ipl_lr_model.pkl')

# Load the model as a pipeline
pipe = joblib.load(model_path)

# Set page title and icon
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏")

# Centered title
st.markdown("<h1 style='text-align: center;'>IPL Win Predictor</h1>", unsafe_allow_html=True)


# User inputs using Streamlit widgets
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox('Batting Team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Bowling Team', sorted(teams))
    with col3:
        selected_city = st.selectbox('Host City', sorted(cities))

    target = st.number_input('Target Score', min_value=0, step=1, value=0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        score = st.number_input('Current Score', min_value=0, step=1, value=0)
    with col5:
        overs = st.number_input('Overs Completed', min_value=0, step=1, value=0)
    with col6:
        wickets = st.number_input('Wickets Lost', min_value=0, max_value=10, step=1, value=0)

    submit_button = st.form_submit_button(label='Predict Probability')

# Predictions
if submit_button:
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.write(f"Probability of {batting_team} winning: {round(win*100, 2)}%")
    st.write(f"Probability of {bowling_team} winning: {round(loss*100, 2)}%")
