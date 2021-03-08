
from Company_Valuation.utils import transfer_roce, transfer_growth_rate, transfer_ebitda_margin, get_revenue_size
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# to run --> streamlit run app.py
# todo: 
    # 1. joblib function
    # 2. input into model
    # 3. heroku
    # 4. footer
    # ... improvement

# Header ---------------------------------------------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: black;'>Company Value Estimator</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>From experts with over 10 years experience</h2>", unsafe_allow_html=True)


# Body
def run():
    # st.title(" XXXX ")
    html_temp="""
    """
    st.markdown(html_temp, unsafe_allow_html = True) 
    # INPUT       
    revenue =        st.number_input("Revenue US$m", 100) 
    ebitda =         st.number_input("EBITDA US$m") 
    net_debt =       st.number_input("Net Debt US$m")
    revenue_growth = st.number_input("Revenue Growth (last 3 years)") 
    return_on_capital_employed = st.number_input("Return On Capital Employed")
    sector =         st.selectbox("Sector", df_sector)
    region =         st.selectbox("Region", df_region)

    # define X
    X = pd.DataFrame(dict(
    sector=[sector],
    country=[region],
    returnOnCapitalEmployed=[transfer_roce(return_on_capital_employed)],
    revenue=[get_revenue_size(revenue)],
    ebitda=[transfer_ebitda_margin(ebitda)],
    growth_rate=[transfer_growth_rate(revenue_growth)],
    ebitda_margin = [transfer_ebitda_margin(ebitda/revenue)]
    ))

    # loading model
    # @st.cache()
    def load_class_model(model='model.joblib'):
        model = joblib.load(model)
        return model

    model = load_class_model()

    prediction=""
    if st.button("Predict"): 
        # add net debt to the prediction
        prediction = model.predict(X) - net_debt
    
        with st.spinner(text='The company is being evaluated ...'):
            time.sleep(3)
            st.success('We estimate that this company has an equity value of : {} US$m'.format(round(prediction[0], 2)))


    # st.success("We estimate that this company has an equity value of :")
    # st.markdown("<h2 style='text-align: center; color: black;'> f"'{prediction}'" </h2>", unsafe_allow_html=True)




# func for df with selectable elements
def get_select_sector():
    print('get_select_sector called')
    return pd.DataFrame({
          'Sector': ['Consumer', 'Communication Services', 'Utilities', 'Industrials', 'Materials', 'Information Technology', 'Healthcare', 'Energy']
        })
df_sector = get_select_sector()

def get_select_box_data():
    print('get_select_box_data called')
    return pd.DataFrame({
          'Region': ['NA','EU','EM', 'ROW']
        })
df_region = get_select_box_data()






# hide made by streamlit text
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

if __name__=='__main__': run()
