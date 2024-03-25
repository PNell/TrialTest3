import streamlit as st
import numpy as np
import pandas as pd
import shap
from streamlit_shap import st_shap
from DataGen2 import generate_data

def makeShapModel(pipeline):
    # Extract the estimator from the pipeline
    model = pipeline.named_steps['model'] 
    # explainer = joblib.load('shap_explainer.joblib')
    explainer = shap.Explainer(model)
    return explainer

def makeShapValues(explainer, pipeline, X):
    feature_names = pipeline[:-1].get_feature_names_out()
     # Apply transformations to X
    X_transformed = X.copy()
    for name, transformer in pipeline.named_steps.items():
        if name != 'model':  # Exclude the model step
            X_transformed = transformer.transform(X_transformed)

    shap_values = explainer(X_transformed)
    
    shap_values.feature_names = feature_names
    shap_values.data = np.array(X)
    return shap_values
      

def feature_importance_page():
    if 'countTotalGeneratedDataset' not in st.session_state:
        return st.write('# Generate Data Set in Data Gen page before viewing predictions')

    model = st.session_state.get('model', 'No model defined')
    if model == 'No model defined':
        st.write(f"# A mechanical prop model is not selected. Please define a model in Prediction Page")
        return

    #page layout 
    st.title('Feature Importance Using SHAP')
    st.write("[SHAP Documentation/Introduction](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)")
    st.write(f"Feature importance based of generated Data set for:")
    for selected_agg_option in st.session_state['selected_agg_options']:
        selected_agg_option_count = st.session_state['metrics'][selected_agg_option]["c_amount"]["count"]

        st.write(f" {selected_agg_option} Number of Data Points: {selected_agg_option_count}")
    st.write(f"Total Number of Data Points: {st.session_state['countTotalGeneratedDataset']}")
    
    st.session_state['shapValues'] = st.session_state.get('shapValues', None)
    feature_names = list(model[:-1].get_feature_names_out())

    if st.button("Generate Shap Data"):
        if 'dfGenInputData' in st.session_state:
            st.session_state['shap_selected_agg_options'] = st.session_state['selected_agg_options']


            dfInput = st.session_state['dfGenInputData']
            st.session_state['explainer'] = makeShapModel(model)
            st.session_state['shapValues'] = makeShapValues(st.session_state['explainer'], model, dfInput)



        # default_ShapScatterVar = feature_names.index(st.session_state.get('selectedShapScatterVar', feature_names[0]))  # Get the index of the saved state
        #index=default_ShapScatterVar
        # st.session_state['selectedShapScatterVar'] = selected_ShapScatterVar
    if st.session_state['shapValues']:
        st_shap(shap.summary_plot(st.session_state['shapValues']))

        selected_ShapScatterVar = st.selectbox("Plotting Var", feature_names) 
        selected_ShapScatterVarColor = st.selectbox("Color Var", ['None'] + feature_names) 
        if selected_ShapScatterVarColor == 'None':
            st_shap(shap.plots.scatter(st.session_state['shapValues'][:, selected_ShapScatterVar]))
        else:
            st_shap(shap.plots.scatter(st.session_state['shapValues'][:, selected_ShapScatterVar],  color=st.session_state['shapValues'][:, selected_ShapScatterVarColor]))


        st.write('Partial Dependence Plit')

        
