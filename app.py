
# from htbuilder import HtmlElement, ul, li, br, hr, a, p, img, styles, classes, fonts
# from htbuilder.units import percent, px
# from htbuilder.funcs import rgba, rgb
# from htbuilder import HtmlElement, styles, classes, fonts
# from htbuilder import H
# px = H.px

# from htbuilder import HtmlElement, styles, classes, fonts
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

# from htbuilder import H
# div = H.div
# ul = H.ul
# li = H.li
# img = H.img

# br = H.br
# a = H.a
# p = H.p
# styles = H.styles
# classes = H.classes
# fonts = H.fonts
# hr = H.hr

# px = H.px
# percent = H.percent
# rgba = H.rgba
# rgb = H.rgb







from Company_Valuation.utils import transfer_roce, transfer_growth_rate, transfer_ebitda_margin, get_revenue_size
from Company_Valuation.data import get_data
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# to run --> streamlit run app.py
# todo: 
    # 1. joblib function 
    # 2. input into model [x]
    # 3. heroku [x]
    # 4. footer 
    # ... improvement

# Header ---------------------------------------------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: black;'>Company Value Estimator</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey;'> ... slogan ... </h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Who are we? What we do? How we can bring value?</h4>", unsafe_allow_html=True)
# Header ---------------------------------------------------------------------------------------------------------------

# Body ------------------------------------------------------------------------------------------------------------------
def run():
    # st.title(" XXXX ")
    html_temp="""
    """
    st.markdown(html_temp, unsafe_allow_html = True) 
    # INPUT  

    # min_revenue = get_data()['revenue'].min()
    # st.write(min_revenue)

    revenue =        st.number_input("Revenue US$m", 100) 
    ebitda =         st.number_input("EBITDA US$m", min_value=0.0) 
    net_debt =       st.number_input("Net Debt US$m")
    revenue_growth = st.number_input("Revenue Growth (last 3 years)") 
    return_on_capital_employed = st.number_input("Return On Capital Employed")
    sector =         st.selectbox("Sector", df_sector)
    region =         st.selectbox("Region", df_region)

    # define X
    X = pd.DataFrame(dict(
    sector=[sector],
    country=[get_region(region)],
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

    # assign model to model
    model = load_class_model()

    # predict when button clicked
    prediction_upper = ""
    prediction_lower = ""
    if st.button("Predict"): 
        # add net debt to the prediction
        prediction = model.predict(X)
        prediction_upper = prediction * (1 + 0.1) - net_debt
        prediction_lower = prediction * (1 - 0.1) - net_debt

        with st.spinner(text='The company is being evaluated ...'):
            time.sleep(3)
            st.info('We estimate that this company has an equity range between  {} and  {} US$m.'.format(round(prediction_lower[0], 2), round(prediction_upper[0], 2)))


    # st.success("We estimate that this company has an equity value of :")
    # st.markdown("<h2 style='text-align: center; color: black;'> f"'{prediction}'" </h2>", unsafe_allow_html=True)



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
          'Region': ['North American','Europe','Emerging Markets', 'Rest of World']
        })
df_region = get_select_box_data()

def get_region(reagion):
    if 'North American':
        return 'NA'
    elif 'Europe':
        return 'EU'
    elif 'Emerging Markets':
        return 'EM'
    else:
        return 'ROW'


# Body ------------------------------------------------------------------------------------------------------------------


# st.markdown("<div class="footer"> <p> style='text-align: center; color: black;'>Company Value Estimator </p></div>", unsafe_allow_html=True)


# Others ----------------------------------------------------------------------------------------------------------------
# hide made by streamlit text
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)





def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 70px; }
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
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
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
        "Made in ",
        image('https://seeklogo.com/images/U/united-kingdom-flag-logo-1088704B5E-seeklogo.com.png',
                width=px(20), height=px(15)),
        " with ❤️ by Pierre, Harry & Ian",
        # link("https://twitter.com/ChristianKlose3", "Pierre, Harry & Ian"),
        # br(),
        # link("https://buymeacoffee.com/chrischross", image('https://i.imgur.com/thJhzOO.png')),
    ]
    layout(*myargs)








# BACKGROUND

CSS = """
h1 {
    color: red;
}
body {
    background-image: url('https://cdn.hipwallpaper.com/i/26/43/QDrqXi.jpg');
    background-size: cover;
}
.controls {
  display: none;
}

"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

if __name__=='__main__': 
    run()
    footer()
