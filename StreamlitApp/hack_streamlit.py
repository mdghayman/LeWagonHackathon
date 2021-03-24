import pandas as pd
import os
import streamlit as st
import pickle
from scikit-learn.preprocessing import StandardScaler
from build_graph import build_graph

bool_to_int = {'False': 0, 'True': 1}
result, new_graph = None, False

col1, col2, col3 = st.beta_columns(3)
col2.image('images/LeWagonLogo.jpeg', use_column_width=True, \
    output_format='JPEG')

st.markdown("<h1 style='text-align: center; color: red;'>\
    Le Wagon Hackathon, 21 March 2021</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.beta_columns(3)
col1.markdown("<h2 style='text-align: center; color: grey'>\
    Michael Hayman</h3>", unsafe_allow_html=True)
col1.image('images/MichaelHayman.jpeg', use_column_width=True, \
    output_format='JPEG')
col2.markdown("<h2 style='text-align: center; color: grey'>\
    Martin Clark</h3>", unsafe_allow_html=True)
col2.image('images/MartinClark.jpeg', use_column_width=True, \
    output_format='JPEG')
col3.markdown("<h2 style='text-align: center; color: grey'>\
    Karina Pacut</h3>", unsafe_allow_html=True)
col3.image('images/KarinaPacut.jpeg', use_column_width=True, \
    output_format='JPEG')

col2.image('images/TrayLogo.jpeg', use_column_width=True, output_format='JPEG')

st.markdown("<h2 style='text-align: center; color: grey'>\
    Tray.io challenge</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>\
    Algorithmically predict the SaaS product that will generate more interest \
    and traffic, to proactively create integrations for unicorns.</p>", \
    unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: grey'>\
    Input</h2>", unsafe_allow_html=True)
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
organic_search_volume = st.number_input(label="Organic search volume")
seo_value = st.slider(label="SEO value (0-3)", min_value=0, max_value=3)

col1, col2, col3 = st.beta_columns(3)
if col2.button('Predict outcome for this API'):
    os.system("rm pic.png")
    new_graph = False
    y = [[
        analyst_value,
        partner_value,
        persona_value,
        bool_to_int[growing_market],
        organic_search_volume,
        seo_value,
    ]]

    df = pd.read_csv('data.csv')
    scaler = StandardScaler()
    features = ['Analyst value (0 - 5)', 'Partner value (0 - 5)',
           'Persona value (0 - 5)', 'Growing market',
           'Organic Search Volume', 'SEO Value (0 - 3)']
    scaler.fit(df[features])
    y_scaled = scaler.transform(y)
    loaded_model = pickle.load(open('model/gm_for_predicton_6dims.sav', 'rb'))
    result = loaded_model.predict(y)[0]
    build_graph(y_scaled)
    new_graph = True

responses = [
    "0>>> Not so sure about this one 😐️",
    "1>>> It’s a 🦄! This api has a good chance of increasing traffic.",
    "2>>> This one is probably not going to do so well🥶",
]

if result in [0, 1, 2]:
    st.markdown(f"<p style='text-align: center'>Result:<br>\
        {responses[result]}</p>", unsafe_allow_html=True)
    if new_graph:
        st.image('pic.png', use_column_width=True, output_format='PNG')
else:
    st.markdown("<p style='text-align: center'>\
        Please enter valid inputs above.</p>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: grey'>\
    Groupings</p>", unsafe_allow_html=True)
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

st.markdown("<h2 style='text-align: center; color: grey'>\
    How to understand these groupings</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>\
    The graphs below show the average perfomance of the three groups, in \
    Organic search volume and Average SEO value respectively.</p>", \
    unsafe_allow_html=True)

col1, col2 = st.beta_columns(2)
col1.image('images/graph01.png', use_column_width=True, output_format='PNG')
col2.image('images/graph02.png', use_column_width=True, output_format='PNG')

st.markdown("<h2 style='text-align: center; color: grey'>\
    Exploratory analysis</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>\
    The diagrams below illustrate correlation among the model's attributes, \
    and the fitting of a decision tree to determine the value of the growing \
    market attribute based on other attributes.</p>", unsafe_allow_html=True)

col1, col2 = st.beta_columns(2)
col1.image('images/correlation.png', use_column_width=True, output_format='PNG')
col2.image('images/tray_tree.png', use_column_width=True, output_format='PNG')
