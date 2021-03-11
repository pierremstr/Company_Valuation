from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from Company_Valuation.utils import transfer_roce, transfer_growth_rate, transfer_ebitda_margin, get_revenue_size
from Company_Valuation.data import get_data
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import time


# Header ---------------------------------------------------------------------------------------------------------------
st.markdown("<h2 style='text-align: center; color: black;'> Value a company using Machine Learning </h2>", unsafe_allow_html=True)
col1, col2, col3 = st.beta_columns(3)
col2.image('Company_Valuation/clean_data/logo.png', width=500, use_column_width=True)

# Header ---------------------------------------------------------------------------------------------------------------

# Text ----------------------------
# st.markdown("<h4 style='text-align: justify; color: black;'>A common approach to value a company is the market approach. The market approach as a valuation method is used to find the value of a company by comparing it to other similar companies that are publicly traded. This method assesses the value of a business through the application of several ratios of value to financial metrics or non-financial parameters of public companies.</h4>", unsafe_allow_html=True)
# Text ----------------------------

# Body ------------------------------------------------------------------------------------------------------------------

def run():
    # st.title(" XXXX ")
    html_temp="""
    """
    st.markdown(html_temp, unsafe_allow_html = True) 
    # INPUT  
    
    
    revenue =        st.number_input("Revenue US$m", min_value=25.00) 
    ebitda =         st.number_input("EBITDA US$m", min_value=25.00, max_value=1_400.00) 

    # if ebitda is smaller than revenue -> WARNING
    if ebitda > revenue:
        revenue = ebitda
        st.warning('The ebitda is greater than the revenue. Please re-enter your revenue!')

    net_debt =       st.number_input("Net Debt US$m")
    revenue_growth = st.number_input("Revenue Growth (e.g. 0.15 for 15%)") 
    sector =         st.selectbox("Sector", df_sector)
    region =         st.selectbox("Region", df_region)

    # define X
    X = pd.DataFrame(dict(
    sector=[sector],
    country=[get_region(region)],
    ebitda=[ebitda],
    growth_rate=[transfer_growth_rate(revenue_growth)],
    ebitda_margin = [transfer_ebitda_margin(ebitda/revenue)]
    ))

    # loading model
    # @st.cache()
    def load_class_model(model='final_model.joblib'):
        model = joblib.load(model)
        return model

    # assign model to model
    model = load_class_model()

    # predict when button clicked
    prediction_upper = ""
    prediction_lower = ""
    if st.button("Estimate"): 
        # add net debt to the prediction
        prediction = model.predict(X)
        prediction_upper = prediction * (1 + 0.1) - net_debt
        prediction_lower = prediction * (1 - 0.1) - net_debt

        with st.spinner(text='The company is being evaluated ...'):
            time.sleep(2)

            if prediction_upper < 0:
                st.markdown("<h3 style='text-align: center; color: black;'> We estimate that this company has an equity value of: </h3>", unsafe_allow_html=True)
                st.markdown("<h1 style='text-align: center; color: black;'> 0 US$m </h1>".format(int(prediction_lower[0]), int(prediction_upper[0])), unsafe_allow_html=True)
            elif (prediction_lower < 0) & (prediction_upper > 0):
                st.markdown("<h3 style='text-align: center; color: black;'> We estimate that this company has an equity value between: </h3>", unsafe_allow_html=True)
                st.markdown("<h1 style='text-align: center; color: black;'> 0 - {:,} US$m </h1>".format(int(prediction_upper[0])), unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: black;'> We estimate that this company has an equity value between: </h3>", unsafe_allow_html=True)
                st.markdown("<h1 style='text-align: center; color: black;'> {:,} - {:,} US$m </h1>".format(int(prediction_lower[0]), int(prediction_upper[0])), unsafe_allow_html=True)


    # func for df with selectable elements -----------------------
# SECTOR -------
def get_select_sector():
    print('get_select_sector called')
    return pd.DataFrame({
          'Sector': ['Consumer', 'Communication Services', 'Utilities', 'Industrials', 'Materials', 'Information Technology', 'Healthcare', 'Energy']
        })

df_sector = get_select_sector()

# REGION ------
def get_select_box_data():
    print('get_select_box_data called')
    return pd.DataFrame({
          'Region': ['North America','Europe','Emerging Markets', 'Rest of World']
        })

df_region = get_select_box_data()

def get_region(reagion):
    if 'North America':
        return 'NA'
    elif 'Europe':
        return 'EU'
    elif 'Emerging Markets':
        return 'EM'
    else:
        return 'ROW'
# Body ------------------------------------------------------------------------------------------------------------------


# Text -------------------------------------------
st.markdown("<p style='text-align: justify; color: black;'>A common approach to value a company is the market approach. The market approach as a valuation method is used to find the value of a company by comparing it to other similar companies that are publicly traded. This method assesses the value of a business through the application of several ratios of value to financial metrics or non-financial parameters of public companies. The estimator below applies a market approach using Machine Learning models to identify patterns in thousand of public companies.</p>", unsafe_allow_html=True)

run()

components.html(
    """
    <hr style="max-width: 60%">
    """, height=10
)

st.markdown("<p style='text-align: justify; color: black;'>For the market approach to be successful, it is critical to ensure that all companies being used for comparison are similar to the subject company or that premiums and discounts are applied for divergent features.</p>", unsafe_allow_html=True)

st.markdown("<p style='text-align: justify; color: black;'>We have analysed the relationship between the market valuation of approximately two thousand public companies with their financial and business profile. Our optimized model enables us to apply this relationship to the profile of any company that meets our criteria and derive an estimation of its equity value. </p>", unsafe_allow_html=True)

st.markdown("<p style='text-align: justify; color: black;'>To ensure a level of accuracy in our model, we did not consider companies that met any of the following criteria:</p>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies with negative EBITDA</li>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies in the financial sector or in real estate</li>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies whose financial information was not available in the last 4 years</li>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies with revenue smaller than US$10 million</li>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies with an enterprise value lower than US$300 million or higher than US$10 bn </li>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies with a “Enterprise Value / EBITDA” ratio lower than 4x or higher than 23x</li>", unsafe_allow_html=True)

st.markdown("<li color: black;'>Companies whose annual growth rate has been greater than 200%</li>", unsafe_allow_html=True)

# Text -------------------------------------------





# Footer ------------------------------------------------------------------------------------------------------------------
# hide made by streamlit text
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)


# Footer 
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 100px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="inline",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(0)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Our analysis was conducted based on market prices as at December 2019, prior to the outbreak of COVID-19.",
        br(),
        "Made in ",
        image('https://seeklogo.com/images/U/united-kingdom-flag-logo-1088704B5E-seeklogo.com.png',
                width=px(20), height=px(15)),
        " by Pierre, Harry & Ian",
        # link("https://twitter.com/ChristianKlose3", "Pierre, Harry & Ian"),
        # br(),with ❤️
        # link("https://buymeacoffee.com/chrischross", image('https://i.imgur.com/thJhzOO.png')),
    ]
    layout(*myargs)

# Footer ------------------------------------------------------------------------------------------------------------------

# BACKGROUND

    # background-image: url('https://media.istockphoto.com/vectors/business-candle-chart-trading-on-the-stock-markets-the-index-and-vector-id1071705056?b=1&k=6&m=1071705056&s=612x612&w=0&h=OD6Z71MnK6qUIzjk_sSEjOm8HXZVQnc0IP6c8ibgbVU=');
    # background-size: cover;

CSS = """
h1 {
    color: red;
}
body {
    background-image: url('http://nextechinnovation.com/wp-content/uploads/2018/08/7243-01-low-poly-background-16x9-1.jpg');
    background-size: cover;
}
label {
    font-weight: bolder;
    font-size: 1.0rem;
}
.block-container {
    padding: 5rem 1rem 0rem;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)




footer()


