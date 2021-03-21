import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle

bool_to_int = {'False': 0, 'True': 1}
result = None

col1, col2, col3 = st.beta_columns(3)
col2.image('images/LeWagonLogo.jpeg', use_column_width=True, output_format='JPEG')

st.markdown("<h1 style='text-align: center; color: red;'>\
    Le Wagon Hackathon, 21 March 2021</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.beta_columns(3)
col1.markdown("<h2 style='text-align: center; color: grey'>\
    Michael Hayman</h3>", unsafe_allow_html=True)
col1.image('images/MichaelHayman.jpeg', use_column_width=True, output_format='JPEG')
col2.markdown("<h2 style='text-align: center; color: grey'>\
    Martin Clark</h3>", unsafe_allow_html=True)
col2.image('images/MartinClark.jpeg', use_column_width=True, output_format='JPEG')
col3.markdown("<h2 style='text-align: center; color: grey'>\
    Karina Pacut</h3>", unsafe_allow_html=True)
col3.image('images/KarinaPacut.jpeg', use_column_width=True, output_format='JPEG')

col2.image('images/TrayLogo.jpeg', use_column_width=True, output_format='JPEG')

st.markdown("<h2 style='text-align: center; color: grey'>\
    Tray.io challenge:</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>\
    Algorithmically predict the SaaS product that will generate more interest \
    and traffic, to proactively create integrations for unicorns.</p>", \
    unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: grey'>\
    Input:</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>\
    To predict whether your new API will be successful, please fill in \
    the details below:</p>", unsafe_allow_html=True)

analyst_value = st.slider(label="Analyst value (0-5)", \
    min_value=0, max_value=5)
partner_value = st.slider(label="Partner value (0-5)", \
    min_value=0, max_value=5)
persona_value = st.slider(label="Persona value (0-5)", \
    min_value=0, max_value=5)
growing_market = st.select_slider(label="Growing market", \
    options=['True', 'False'])
seo_value = st.slider(label="SEO value (0-3)", min_value=0, max_value=3)
organic_search_volume = st.number_input(label="Organic search volume")

col1, col2, col3 = st.beta_columns(3)
if col2.button('Predict outcome for this API'):
    y = [[
        analyst_value,
        partner_value,
        persona_value,
        bool_to_int[growing_market],
        seo_value,
        organic_search_volume,
    ]]
    y_scaled = StandardScaler().fit_transform(y)
    loaded_model = pickle.load(open('model/gm_for_predicton_6dims.sav', 'rb'))
    result = loaded_model.predict(y)[0]

if result == 0:
    st.markdown("<p style='text-align: center'>\
        Result:<br>This is a <b>failure</b>.<br>The API falls into Group 0 \
        (see below).</p>", unsafe_allow_html=True)
elif result == 1:
    st.markdown("<p style='text-align: center'>\
        Result:<br>This is a <b>unicorn</b>.<br>The API falls into Group 1 \
        (see below).</p>", unsafe_allow_html=True)
elif result in [0, 2]:
    st.markdown("<p style='text-align: center'>\
        Result:<br>This is a <b>failure</b>.<br>The API falls into Group 2 \
        (see below).</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align: center'>\
        Please enter valid inputs above.</p>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: grey'>\
    Groupings:</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>\
    This graph illustrates the APIs in the dataset, arranged in three groups \
    using a clustering algorithm. While shown here in two dimensions, it is \
    abstracted from six dimensions. APIs falling into group are most likely \
    to suceed on Tray's platform.</p>", unsafe_allow_html=True)

st.image('images/pic.png', use_column_width=True, output_format='PNG')

st.markdown("<p style='text-align: center'>\
    Represented again here, because 3D graphs are fun. ;)</p>", \
    unsafe_allow_html=True)

st.image('images/3d_pic.png', use_column_width=True, output_format='PNG')
