import os
import streamlit as st
import numpy as np
from PIL import  Image

from multipage import MultiPage
from modules import analyze_metrics, customer_seg, customer_ltv, churn, predict_npd, predict_sales

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('Logo.png')
display = np.array(display)
col1, col2 = st.beta_columns(2)
col1.image(display, width = 400)
col2.title("Dashboard")

app.add_page("Quick Overview", analyze_metrics.app)
app.add_page("Customer Segments", customer_seg.app)
app.add_page("Lifetime Values", customer_ltv.app)
app.add_page("Churn", churn.app)
app.add_page("Next Purchase Day", predict_npd.app)
app.add_page("Sales Predictions", predict_sales.app)

app.run()