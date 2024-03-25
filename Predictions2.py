import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import copy





#make page for predictions comparision
def predictions_page():
    if 'dfGenInputData' not in st.session_state:
        return st.write('# Generate Data Set in Data Gen page before viewing predictions')
    if 'selected_agg_options' not in st.session_state:
        st.write('# Generate Data in Data Gen page')

    st.session_state['modelType'] = st.session_state.get('modelType', 'TensileStrength')

    st.title('RF Model Prop Preds')

    
    default_index_modelType = st.session_state['modelTypes'].index(st.session_state.get('modelType', 'TensileStrength'))  # Get the index of the saved state
    selected_model_type= st.selectbox("Select Mech Prop for prediction", st.session_state['modelTypes'], index=default_index_modelType)
    st.session_state['modelType'] = selected_model_type

    st.session_state['model'] = st.session_state['models'][st.session_state['modelType']]
    model = st.session_state['model']
    st.markdown('##')
    st.write(f"{st.session_state['aggOption']}/s")
    for i in st.session_state['selected_agg_options']:
        st.markdown("- " + i + 'will make a table at some point (hopefully)', unsafe_allow_html=True)


    st.markdown('##')
    # st.write(f"# SteelDesign: {st.session_state['selected_agg_options']}")
    # Displaying Historical Data for Steel Disign
    historicalData = st.session_state['cleanDataAllCols'].loc[st.session_state['cleanDataAllCols'][st.session_state['aggOptionFullName']].isin(st.session_state['selected_agg_options'])]
    if historicalData.shape[0] == 0:
        figHistorical = None

    else:
        propData = historicalData[st.session_state['modelType']]
        
        count_hist = propData.count()
        mean_hist = np.mean(propData)
        std_dev_hist = np.std(propData)
        min_hist = np.min(propData)
        max_hist = np.max(propData)
            
    
    dfInput = st.session_state['dfGenInputData']
    predictions = model.predict(dfInput)
    st.session_state['predictions'] = predictions
    fullDataSet = copy.copy(st.session_state['dfGenDataFull'])
    fullDataSet['predictions'] = predictions
    st.session_state['predictionsFullData'] = fullDataSet

    # Calculate the maximum x-axis range
    max_x_range = max(max(historicalData[st.session_state['modelType']]), max(predictions)) 
    max_x_range += max_x_range * 0.01
    min_x_range = min(min(historicalData[st.session_state['modelType']]), min(predictions))
    min_x_range += max_x_range * -0.01
    # Update the layout of both histograms to have the same x-axis range

    figHistorical = go.Figure()
    for aggOption in st.session_state['selected_agg_options']:
        data = historicalData.loc[historicalData[st.session_state['aggOptionFullName']] == aggOption]
        figHistorical.add_trace(go.Histogram(x=data[st.session_state['modelType']], 
                                  # histnorm='probability', 
                                   name=aggOption))
    figHistorical.update_layout(title=f"Historical {st.session_state['modelType']}",
                                barmode='overlay')
    figHistorical.update_xaxes(range=[min_x_range, max_x_range])
    # Display a histogram of the predictions
    figPredictions = go.Figure()
    for aggOption in st.session_state['selected_agg_options']:
        data = fullDataSet.loc[fullDataSet[st.session_state['aggOptionFullName']] == aggOption]
        figPredictions.add_trace(go.Histogram(x=data['predictions'], 
                                  # histnorm='probability', 
                                   name=aggOption))

    figPredictions.update_xaxes(range=[min_x_range, max_x_range])
    figPredictions.update_layout(title=f"Predicted {st.session_state['modelType']}",
                                 barmode='overlay')

    # Calculate and display statistics
    mean_pred = np.mean(predictions)
    std_dev_pred = np.std(predictions)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)

    col1, col2 = st.columns(2)
    # Display the charts in their respective columns
    with col1:
        if figHistorical:
            st.write(f"Numb Historical Test Results: {count_hist:.4f}")
            st.write(f"Mean Historical: {mean_hist:.4f}")
            st.write(f"STDev of Historical: {std_dev_hist:.4f}")
            st.write(f"Minimum Historical: {min_hist:.4f}")
            st.write(f"Maximum Historical: {max_hist:.4f}")
            st.plotly_chart(figHistorical, use_container_width=True)
        else:
            st.write(f"No Historical Data")

    with col2:
        st.write(f'Numb Predictions: {predictions.shape[0]}')
        st.write(f"Mean Predictions: {mean_pred:.4f}")
        st.write(f"STDev Predictions: {std_dev_pred:.4f}")
        st.write(f"Minimum Prediction: {min_pred:.4f}")
        st.write(f"Maximum Prediction: {max_pred:.4f}")
        st.plotly_chart(figPredictions, use_container_width=True)