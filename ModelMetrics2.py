import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from DataGen2 import agg_freq, getAgg, getSteelDesign, calculateMetrics

def evaluate_model(model, X, y):
     # Making predictions
    y_pred = model.predict(X)

    # Calculating metrics
    N = y_pred.shape[0]
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    ape = np.mean(np.abs((y - y_pred) / y)) * 100

    # Creating a DataFrame to display the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['N','Average Percent Error', 'MAE', 'RMSE', 'R2'],
        'Value': [N, ape, mae, rmse, r2]
    })

    # Rounding values to 3 decimal places
    metrics_df['Value'] = metrics_df['Value'].round(3)
    return metrics_df

# Calculate metrics
def plotPreds(df):
    true_values, predicted_values = df[st.session_state['modelType']], df['propPred']
    r2 = r2_score(true_values, predicted_values)  # R-squared
    mae = mean_absolute_error(true_values, predicted_values)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))  # Root Mean Squared Error

    # Determine the range for the perfect fit line
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))

    # Create a scatter plot
    scatter_plot = go.Figure()

    # Add scatter plot for predictions
    for aggOption in st.session_state['selected_agg_options']:
        data = df.loc[df[st.session_state['aggOptionFullName']] == aggOption]
        scatter_plot.add_trace(go.Scatter(x=data[st.session_state['modelType']], y=data['propPred'], mode='markers', name=aggOption))

    # Add line for perfect fit
    scatter_plot.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', name='Perfect Fit', 
                                    line=dict(color='black', dash='dash')))

    # Update layout
    scatter_plot.update_layout(
        title='Model Predictions vs True Values',
        title_x=0.3,
        font_color="black",
        font=dict(color='black'),
        title_font=dict(color='black'),
        xaxis=dict(
            title='True Values',
            title_font=dict(color='black'),  # X-axis title font color
            tickfont=dict(color='black'),    # X-axis tick label font color
            showline=True,  
            linewidth=2,    
            linecolor='black', 
            showgrid=True,
            gridcolor='lightgray',
            tickcolor='black',  
            tickwidth=2,        
            color='black',       
            ticks='outside',  
            ticklen=10       
        ),
        yaxis=dict(
            title='Predicted Values',
            title_font=dict(color='black'),  # X-axis title font color
            tickfont=dict(color='black'),    # X-axis tick label font color
            showline=True,  
            linewidth=2,    
            linecolor='black', 
            showgrid=True,
            gridcolor='lightgray',
            tickcolor='black',  
            tickwidth=2,        
            color='black',       
            ticks='outside',  
            ticklen=10       
        ),
        legend=dict(
                        font=dict(color='black')  # Legend text color
                     ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        annotations=[
            dict(
                x=1.125, y=0.5,  # Position relative to the axes
                xref='paper', yref='paper',
                text=f'N= {len(predicted_values) + 1} <br>R²={r2:.2f}<br>MAE={mae:.2f}<br>RMSE={rmse:.2f}',  # Text with metrics
                showarrow=False,
                align='left',
                font=dict(
                            color='black'  # Text color
                         )
            )
        ]
    )
    return scatter_plot

def plot_residuals_histogram(df):
    yTrue = df[st.session_state['modelType']]
    yPred = df['propPred']
    df['residuals'] = yTrue - yPred
    mae = mean_absolute_error(yTrue, yPred)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(yTrue, yPred))
    maxResidual = max(np.abs(df['residuals']))
    historicalData = st.session_state['cleanDataAllCols']
    
    # Plot histogram
    fig = go.Figure()
    for aggOption in st.session_state['selected_agg_options']:
        data = df.loc[df[st.session_state['aggOptionFullName']] == aggOption]
        fig.add_trace(go.Histogram(x=data['residuals'], name=aggOption))
    fig.update_layout(
        title='Histogram of Residuals',
        title_x=0.3,
        xaxis=dict(title='Residuals'),
        yaxis=dict(title='Frequency'),
        showlegend=True,
        annotations=[
            dict(
                x=1.1, y=0.5,
                xref='paper', yref='paper',  # Position relative to the axes
                text=f'N= {yTrue.shape[0]} <br>MAE={mae:.2f}<br>RMSE={rmse:.2f} <br>|Max Res|={maxResidual: .2f}', 
                align='left', # Text with metrics
            )
        ],
        barmode='stack'
    )
    return fig

def model_metrics_page():
    st.session_state['modelType'] = st.session_state.get('modelType', 'TensileStrength')

    st.title(f"RF Model Metrics {st.session_state['modelType']}")

    st.write('Includes both training and test Data set')

    default_index_modelType = st.session_state['modelTypes'].index(st.session_state.get('modelType', 'TensileStrength'))  # Get the index of the saved state
    selected_steelDesign = st.selectbox("Select Mech Prop for prediction", st.session_state['modelTypes'], index=default_index_modelType)
    st.session_state['modelType'] = selected_steelDesign

    st.session_state['model'] = st.session_state['models'][st.session_state['modelType']]
    model = st.session_state['model']

    #Make the user select and agg type (grade or steeldesign)
    aggOptions = ['Grade', 'Steel Design']
    #store agg option in session
    aggIndex = aggOptions.index(st.session_state.get('aggOption', aggOptions[0]))  # Get the index of the saved state
    selected_aggOption = st.selectbox("Aggregation Option", aggOptions, index=aggIndex)
    st.session_state['aggOption'] = selected_aggOption

    #if steel design is selected
    if st.session_state['aggOption'] == 'Steel Design':
        #full agg name store in session very hard coded but fuck it
        st.session_state['aggOptionFullName'] = 'steel_design_description'

    if st.session_state['aggOption'] == 'Grade':
        #full agg name store in session very hard coded but fuck it
        st.session_state['aggOptionFullName'] = 'grade'
 
    #get Steel design with counts of number of data points
    aggKeysWithCounts = agg_freq(st.session_state['cleanDataAllCols'], st.session_state['aggOptionFullName'])

    if 'selected_aggKeysWithCounts' not in st.session_state:
        selected_aggKeysWithCountsTemp = []
    else:
        selected_aggKeysWithCountsTemp = st.session_state['selected_aggKeysWithCounts']
    #Allow user to select multiple steel desing and data handling 
    selected_aggKeysWithCounts = st.multiselect(f'Select Steel Design/s. You have to select twice for some reason I will fix it (hopefully)', aggKeysWithCounts,
                                                selected_aggKeysWithCountsTemp)

    st.session_state['selected_aggKeysWithCounts'] = selected_aggKeysWithCounts
    

    if 'selected_aggKeysWithCounts' in st.session_state:
        #format steel design with out counts 
        selected_aggKeys = []
        for key in st.session_state['selected_aggKeysWithCounts']:
            selected_aggKeys.append(getSteelDesign(key))

        st.session_state['selected_agg_options'] = selected_aggKeys
        #Calculate metrics for each steel design
        if 'metrics' not in st.session_state:
            st.session_state['metrics'] = {}

        if list(st.session_state['metrics'].keys()) != st.session_state['selected_agg_options']:
            st.session_state['metrics'] = calculateMetrics(df=st.session_state['cleanDataAllCols'], 
                                                        aggCol=st.session_state['aggOptionFullName'], 
                                                        aggNames=selected_aggKeys,
                                                        default_columns=st.session_state['default_columns'])
               

    if st.session_state['selected_agg_options'] == 'All':
        data = st.session_state['cleanDataAllCols']

    else:
        data = st.session_state['cleanDataAllCols'].loc[st.session_state['cleanDataAllCols'][st.session_state['aggOptionFullName']].isin(st.session_state['selected_agg_options'])]
   
    if data.shape[0] == 0:
        figHistorical = None

    else:
        #Measured values 
        propTrue = data[st.session_state['modelType']]
        propTrue_count = propTrue.count()
        propTrue_mean = np.mean(propTrue)
        propTrue_std_dev = np.std(propTrue)
        propTrue_min = np.min(propTrue)
        propTrue_max = np.max(propTrue)

        #Predicited values
        
        inputData = data[st.session_state['default_columns']] 
        propPred = model.predict(inputData)
        data['propPred'] = propPred
        propPred_count = propPred.shape[0]
        propPred_mean = np.mean(propPred)
        propPred_std_dev = np.std(propPred)
        propPred_min = np.min(propPred)
        propPred_max = np.max(propPred)

        #Metrics
        dfMetrics = evaluate_model(model, inputData, propTrue)

         # Centering the table
        st.write(dfMetrics)
        st.plotly_chart(plotPreds(data))
        st.plotly_chart(plot_residuals_histogram(data))

    return 