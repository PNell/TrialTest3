import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil

def steel_design_freq(df):
    df = df.sort_values(by='steel_design_description')
    steelDesigns = df['steel_design_description'].unique()
    counts = []
    for steelDesign in steelDesigns:
        counts.append(df.loc[df['steel_design_description']==steelDesign].count()['steel_design_description'])
    steelDisgnsCounts = []
    for i in range(len(steelDesigns)):
        steelDisgnsCounts.append(f'{steelDesigns[i]}: {counts[i]}')
    return steelDisgnsCounts
    

def getSteelDesign(s):
    return s.split(':')[0]


def agg_freq(df, agg):
    df = df.sort_values(by=agg)
    steelDesigns = df[agg].unique()
    counts = []
    for steelDesign in steelDesigns:
        counts.append(df.loc[df[agg]==steelDesign].count()[agg])
    steelDisgnsCounts = []
    for i in range(len(steelDesigns)):
        steelDisgnsCounts.append(f'{steelDesigns[i]}: {counts[i]}')
    return steelDisgnsCounts

def getAgg(s):
    return s.split(':')[0]

def calculateMetrics(df, aggCol, aggNames, default_columns):
    df = df[default_columns + [aggCol]]
    dataDic = {}
    for steelDesign in  aggNames:
        steelDF = df.loc[df[aggCol] == steelDesign]
        VarData = {}
        for col in default_columns:
            VarData[col] = {'mean': steelDF[col].mean(),
                            'std': steelDF[col].std(),
                            'count': steelDF[col].count(),
                            'distribution': 'Normal',
                            'name': col}

        dataDic[steelDesign] = VarData
    return dataDic

def getMetrics(data, aggNames, searchCol):
    return data[aggNames][searchCol]

def getTotalCount(data):
    count = 0
    for steelDesign in data:
        count += data[steelDesign]['c_amount']['count']
    return count

def getHistoricalCounts(df, colName, rowVals):
    count = 0
    for rowVal in rowVals:
        count += df.loc[df[colName] == rowVal].shape[0]
    return count

def generate_data(distribution, mean, std_dev, count):
    if distribution == 'Normal':
        data = np.random.normal(mean, std_dev, count)
    elif distribution == 'Log-Normal':
        # For log-normal distribution, mean and std_dev are the mean and
        # standard deviation of the underlying normal distribution
        # from which exp(value) is drawn.
        mu = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
        sigma = np.sqrt(np.log(std_dev**2 / mean**2 + 1))
        data = np.random.lognormal(mu, sigma, count)
    
    if mean > 0:
            data = np.clip(data, a_min=0, a_max=None)

    return data

def makeDistPlotGrid(default_columns, dataSteelDesign):
    ncols = 3  # Number of columns in the grid
    nrows = ceil(len(default_columns) / ncols)  # Calculate the number of rows needed
    # Create a subplot figure with a 3-column layout
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=default_columns)
    df = st.session_state['cleanDataAllCols']
    df = df.loc[df[st.session_state['aggOptionFullName']] == st.session_state['selected_agg_key_modify']]
    for idx, colName in enumerate(default_columns):
        
        row = (idx // ncols) + 1
        col = (idx % ncols) + 1

        var = dataSteelDesign[colName]
        data = generate_data(var['distribution'], var['mean'], var['std'], var['count'])

        #This does not look good maybe find a way to do it later

        # historicalData = df[colName]
        # fig.add_trace(go.Histogram(x=historicalData, histnorm='probability', showlegend=True, name='Historical Data'),
        #             row=row, col=col)

        fig.add_trace(go.Histogram(x=data, histnorm='probability', showlegend=True, name='Generated Data'),
                    row=row, col=col)
    # Update layout for a better view
    fig.update_layout(
        title_text='Distributions Grid',
        height=900,
        width=900,
        showlegend=False
    )
    st.plotly_chart(fig)

def generateFullDataSet():
    metrics = st.session_state['metrics']
    data = pd.DataFrame()
    for aggName in metrics:
        variables = metrics[aggName]
        dfTemp = pd.DataFrame()
        for i in variables:
            var = variables[i]
            x = generate_data(var['distribution'], var['mean'], var['std'], var['count'])
            dfTemp[var['name']] = x
        dfTemp[st.session_state['aggOptionFullName']] = aggName
        data = pd.concat([data, dfTemp])
    return data
def generateInputDataSet():
    df = generateFullDataSet()
    df = df.drop(columns=[st.session_state['aggOptionFullName']])
    return df


def data_gen_page():#
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
        st.session_state['selected_aggKeysWithCounts'] = []
    else:
        st.session_state['selected_aggKeysWithCounts'] = st.session_state['selected_aggKeysWithCounts']
    #Allow user to select multiple steel desing and data handling 
    st.session_state['selected_aggKeysWithCounts'] = st.multiselect(f'Select Steel Design/s. You have to select twice for some reason I will fix it (hopefully)', 
                                                                    aggKeysWithCounts,
                                                                    default=st.session_state['selected_aggKeysWithCounts']
                                                                    )

    
    

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
            st.session_state['count'] = getHistoricalCounts(df=st.session_state['cleanDataAllCols'], 
                                             colName=st.session_state['aggOptionFullName'], 
                                             rowVals=st.session_state['selected_agg_options'])
               

    if 'selected_agg_options' in st.session_state and st.session_state['selected_agg_options']: #st.button("Modify Data Set") and

        #select steel design to modify metrics and plot
        selected_agg_key_modify = st.selectbox("Select Subset to modify", st.session_state['selected_agg_options'])
        st.session_state['selected_agg_key_modify'] = selected_agg_key_modify
        data_agg_key = st.session_state['metrics'][selected_agg_key_modify]

        countInDataSet = getHistoricalCounts(df=st.session_state['cleanDataAllCols'], 
                                             colName=st.session_state['aggOptionFullName'], 
                                             rowVals=st.session_state['selected_agg_options']) 
        
        countHistSelectedAggKey = getHistoricalCounts(df=st.session_state['cleanDataAllCols'], 
                                             colName=st.session_state['aggOptionFullName'], 
                                             rowVals=[st.session_state['selected_agg_key_modify']])
        
        countTotalGeneratedDataset = getTotalCount(st.session_state['metrics'])
        st.session_state['countTotalGeneratedDataset'] = countTotalGeneratedDataset
        countGeneratedSelectedAggKey = data_agg_key["c_amount"]["count"]
        
        
        col1, col2 = st.columns(2)
        count = st.number_input(f"Number of generated data points for {selected_agg_key_modify}", min_value=1, value=st.session_state.get('count', data_agg_key["c_amount"]["count"]), step=100)
        st.session_state['count'] = count

        #metric modification
        colCounts1, colCounts2 = st.columns(2)
        with colCounts1:
            st.write(f'# Historical Counts')
            st.write(f'Total Number of Historical Data Points: {countInDataSet}')
            st.write(f'Number of Historical Data points for {selected_agg_key_modify}: {countHistSelectedAggKey}')
            st.write(f'Percent of Historical Data points {selected_agg_key_modify}: {countHistSelectedAggKey / countInDataSet * 100:.1f}')

        with colCounts2:
            st.write(f'# Generated Counts')
            st.write(f'Total Number of Generated Data Points: {countTotalGeneratedDataset}')
            st.write(f'Current Number of Generated Data Points for {selected_agg_key_modify}: {countGeneratedSelectedAggKey}')
            st.write(f'Percent of Generated Data Points for {selected_agg_key_modify}: {countGeneratedSelectedAggKey / countTotalGeneratedDataset * 100:.1f}')
        middle = round(len(st.session_state['default_columns']) / 2)
        with col1:
            for i in range(middle):  # First 5 variables
                with st.expander(f"{st.session_state['default_columns'][i]}"):
                    dist_type = data_agg_key[st.session_state['default_columns'][i]]['distribution']
                    mean = data_agg_key[st.session_state['default_columns'][i]]['mean']
                    std = data_agg_key[st.session_state['default_columns'][i]]['std']
                    distributions = ['Normal', 'Log-Normal']

                    dist_index = distributions.index(dist_type)
                    dist_type = st.selectbox(
                        "Distribution Type",
                        ['Normal', 'Log-Normal'],
                        index=dist_index,
                        key=f'dist_type_{i}'
                    )
                    mean = st.number_input("Mean", value=mean,
                                        key=f'mean_{i}',  format="%0.3f")
                    std = st.number_input("Standard Deviation", value=std,
                                            min_value=0.000, key=f'std_{i}',  format="%0.3f")


                    st.session_state['metrics'][selected_agg_key_modify][st.session_state['default_columns'][i]] = {
                                        'mean': mean,
                                        'std': std,
                                        'count': count,
                                        'distribution': dist_type,
                                        'name': st.session_state['default_columns'][i]
                                        }

        with col2:
            for i in range(middle, len(st.session_state['default_columns'])):  # First 5 variables
                with st.expander(f"{st.session_state['default_columns'][i]}"):
                    dist_type = data_agg_key[st.session_state['default_columns'][i]]['distribution']
                    mean = data_agg_key[st.session_state['default_columns'][i]]['mean']
                    std = data_agg_key[st.session_state['default_columns'][i]]['std']
                    distributions = ['Normal', 'Log-Normal']
                    dist_index = distributions.index(dist_type)

                    dist_type = st.selectbox(
                        "Distribution Type",
                        ['Normal', 'Log-Normal'],
                        index=dist_index,
                        key=f'dist_type_{i}'
                    )
                    mean = st.number_input("Mean", value=mean,
                                        key=f'mean_{i}',  format="%0.3f")
                    std = st.number_input("Standard Deviation", value=std,
                                            min_value=0.000, key=f'std_{i}',  format="%0.3f")


                    st.session_state['metrics'][selected_agg_key_modify][st.session_state['default_columns'][i]] = {
                                        'mean': mean,
                                        'std': std,
                                        'count': count,
                                        'distribution': dist_type,
                                        'name': st.session_state['default_columns'][i]}

            #Show distirbution created from mertrics 
            #want to add historical data can not figure out how to make it look nice

    if 'selected_aggKeysWithCounts' in st.session_state and st.button("Generate Data Set"):
        makeDistPlotGrid(st.session_state['default_columns'], data_agg_key)
        st.session_state['dfGenDataFull'] = generateFullDataSet()
        st.session_state['dfGenInputData'] = generateInputDataSet()


    return