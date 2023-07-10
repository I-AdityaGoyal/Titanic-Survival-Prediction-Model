import streamlit as st
import pickle
import pandas as pd


pipeline = pickle.load(open("https://github.com/I-AdityaGoyal/Titanic-Survival-Prediction-Model/blob/main/Streamlit%20Python%20File/pipe_data.pkl","rb"))


def predict_survival(pclass, sex, age, fare, embarked, family_size):
    # Create a DataFrame with the user input
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked],
        'FamilySize': [family_size]
    })
    
    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(data)
    
    return predictions[0]


def main():
    # Set the title and description of the web app
    st.title('Titanic Survival Prediction')
    st.write('Enter the passenger details to predict survival.')
    
    # Get user input using Streamlit input components
    pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.radio('Sex', ['Male', 'Female'])
    age = st.number_input('Age')
    fare = st.selectbox('Fare', ['Low', 'Medium', 'High', 'Very High'])
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
    family_size = st.number_input('Family Size')
    
    # Make predictions on user input
    if st.button('Predict'):
        result = predict_survival(pclass, sex, age, fare, embarked, family_size)
        st.write(f'The predicted survival is: {result}')

if __name__ == '__main__':
    main()
