# import libraries
from __future__ import division
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb
#initate plotly
# pyoff.init_notebook_mode()

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

def app():
    tx_data = pd.read_csv('data/online_retail_II.csv')

    #REVENUE
    #converting the type of Invoice Date Field from string to datetime.
    tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

    #creating YearMonth field for the ease of reporting and visualization
    tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)

    #calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
    tx_data['Revenue'] = tx_data['Price'] * tx_data['Quantity']
    tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()

    #MONTHLY REVENUE
    #creating a new dataframe with UK customers only
    tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)

    #creating monthly active customers dataframe by counting unique Customer IDs
    tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['Customer ID'].nunique().reset_index()

    plot_data = [
        go.Scatter(
            x=tx_revenue['InvoiceYearMonth'],
            y=tx_revenue['Revenue'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category"},
            title='Montly Revenue'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig, use_container_width=True)

    #MONTHLY GROWTH RATE
    #using pct_change() function to see monthly percentage change
    tx_revenue['MonthlyGrowth'] = tx_revenue['Revenue'].pct_change()

    plot_data = [
        go.Scatter(
            x=tx_revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
            y=tx_revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category"},
            title='Montly Growth Rate'
        )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig, use_container_width=True)
    # pyoff.iplot(fig)

    #creating a new dataframe with UK customers only
    tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)

    #creating monthly active customers dataframe by counting unique Customer IDs
    tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['Customer ID'].nunique().reset_index()

    #plotting the output
    plot_data = [
        go.Bar(
            x=tx_monthly_active['InvoiceYearMonth'],
            y=tx_monthly_active['Customer ID'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category"},
            title='Monthly Active Customers'
        )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig, use_container_width=True)
    # pyoff.iplot(fig)
    return