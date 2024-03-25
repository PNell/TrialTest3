#Library Imports
import streamlit as st
import pandas as pd
import joblib

#Streamlit Page Fucntions
from FeatureImportance2 import feature_importance_page
from DataGen2 import data_gen_page
from Predictions2 import predictions_page
from ModelMetrics2 import model_metrics_page
from DataGen2 import data_gen_page
#Data Loading

# st.session_state['dfMechanicals']  = pd.read_csv('Data/mechanicalsWithOutTransitions.csv')
# st.session_state['steelDesignAggData'] = pd.read_pickle('Data/SteelDesignProccessAgg.pickle')
# st.session_state['allDataCleaned'] = pd.read_csv('Data/allDataCleanedWithOutTransitions.csv')
# st.session_state['modelData'] = pd.read_csv('Data/modelDataWOutTransitions.csv')
st.session_state['cleanDataAllCols'] = pd.read_csv('Data/StreamlitData032524.csv')

st.session_state['default_columns']  = [

                                        'coil_thickness',

                                        'c_amount','mn_amount',  'nb_amount', 'n_amount', 'v_amount', 
                                        'ti_amount','al_amount',  'si_amount', #'p_amount', 's_amount',
                                        
                                        
                                        'TF1_EXIT_AVG_BODY','TBC_thickness', 
                                        
                                        'TEMP_FM_ENTRY_AVG', 'TEMP_FM_EXIT_AVG', 'TEMP_COILING_AVG'

                                    ]

st.session_state['modelTypes'] = ['YieldStrength', 'TensileStrength', 'TotalElongation']

st.session_state['models'] = {'YieldStrength': joblib.load('Models/RFYSModelwRMData.joblib'), 
                                'TensileStrength': joblib.load('Models/RFTSModelwRMData.joblib'), 
                                'TotalElongation': joblib.load('Models/RFTEModelwRMData.joblib')} 



# Simple navigation
st.set_page_config(layout="wide")
st.sidebar.title('Navigation')
page = st.sidebar.radio("Select a Page:", ["Data Gen", "Predictions", 
                                            "Feature Importance", "Model Metrics"
                                            ])

if page == "Data Gen":
    data_gen_page()
elif page == "Predictions":
    predictions_page()
elif page == "Feature Importance":
    feature_importance_page()
elif page == "Model Metrics":
    model_metrics_page()



#trialtest-qifwjxuasv9kj2p6vjuobo
