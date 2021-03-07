
import streamlit as st

import numpy as np
import pandas as pd

# model = joblib.load(model)

st.markdown("<h1 style='text-align: center; color: black;'>Company Value Estimator</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>From experts with over 10 years experience</h2>", unsafe_allow_html=True)


def run():
    # st.title("Linear Regression Model")
    html_temp="""
    """
    st.markdown(html_temp) 
    var_1=st.text_input("Revenue US$m") 
    var_2=st.text_input("EBITDA US$m") 
    var_3=st.text_input("Revenue Growth (last 3 years)") 
    ar_4=st.text_input("Return On Capital Employed") 
    var_5=st.selectbox("Sector", df['Sector'])
    var_5=st.selectbox("Region", df['Region'])
    prediction=""
    if st.button("Predict"): prediction=lr_prediction(var_1,var_2,var_3, var_4,var_5)
    st.success("We estimate that this company has an equity value of :")
    st.markdown("<h2 style='text-align: center; color: black;'> 2,074 US$m</h2>", unsafe_allow_html=True)


# st.footer("<h4 style='text-align: center; color: black;'> Our estimation is based upon a market approach. We analysed valuation of public companies as at December 2019, prior to the outbreak of the coronavirus.</h4>", unsafe_allow_html=True)


# func for df with selectable elements
def get_select_box_data():
    print('get_select_box_data called')
    return pd.DataFrame({
          'Sector': ['Consumer', 'Communication Services', 'Utilities', 'Industrials', 'Materials', 'Information Technology', 'Healthcare, Energy'],
          'Region': ['North America','European Union','Emerging Markets', 'Rest of the world', '', '', '']
        })
df = get_select_box_data()




# hide made by streamlit text
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)


if __name__=='__main__': run()
