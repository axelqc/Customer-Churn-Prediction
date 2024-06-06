import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
st.title(f"Predicción Abandono de clientes")


sexo = st.radio(
    "Ingrese el sexo del cliente",
    ["Masculino", "Femenino"],
    index=None,
)
citizen = st.radio(
    "El cliente es de la tercera edad?",
    ["Sí", "No"],
    index=None,
)

partner = st.radio(
    "El cliente tiene pareja?",
    ["Sí", "No"],
    index=None,
)

Dependent = st.radio(
    "El cliente depende de alguien más?",
    ["Sí", "No"],
    index=None,
)

PhoneService = st.radio(
    "El cliente tiene servicio telefónico?",
    ["Sí", "No"],
    index=None,
)

MultipleLines = st.radio(
    "El cliente tiene múltples líneas?",
    ["Sí", "No"],
    index=None,
)

OnlineSecurity = st.radio(
    "El cliente tiene protección virtual?",
    ["Sí", "No"],
    index=None,
)

OnlineBackup = st.radio(
    "El cliente tiene respaldo virtual?",
    ["Sí", "No"],
    index=None,
)

DeviceProtection = st.radio(
    "El cliente tiene protección para sus dispositivos?",
    ["Sí", "No"],
    index=None,
)

TechSupport = st.radio(
    "El cliente tiene apoyo técnico?",
    ["Sí", "No"],
    index=None,
)

StreamingTV = st.radio(
    "El cliente tiene servicio de cable?",
    ["Sí", "No"],
    index=None,
)

Streaming = st.radio(
    "El cliente tiene servicio Streaming?",
    ["Sí", "No"],
    index=None,
)

Paperless = st.radio(
    "El cliente tiene servicio de facturación electrónica?",
    ["Sí", "No"],
    index=None,
)

Monthly = st.number_input("Ingrese su cargo mensual ($)",min_value=0.0, max_value=999999.8)

Total =  st.number_input("Ingrese el total ($)",min_value=0.0, max_value=999999.8)

ChurnScore = st.slider("Ingrese la probabilidad de abandono del cliente")

CLTV = st.number_input("Ingrese el CLTV del cliente",min_value=0, max_value=9999999)

InternetService = st.radio(
    "Ingrese el tipo de Internet que contrató el cliente",
    ["DSL", "Fibra optica", "No tiene internet"],
    index=None,
)

Contract = st.radio(
    "Ingrese tipo de contrato del cliente",
    ["Mes a mes", "Un año de contrato", "Dos años de contrato"],
    index=None,
)

PaymenMethod = st.radio(
    "Ingrese la forma de pago del cliente",
    ["Transferencia bancaria", "Tarjeta de crédito", "e-Check", 'Enviada por correo'],
    index=None,
)

TenureMonths = st.radio(
    "Ingrese cuantos meses lleva el cliente con la empresa",
    ["1 - 12", "13 - 24", "25 - 36", '37 - 48', "49 - 60", "61 - 72"],
    index=None,
)

def enconding(var,pos):
    return 1 if var == pos else 0

def encoding_many(options, variable):
    data = {option: [1 if option == variable else 0] for option in list(options.keys())}
    data = pd.DataFrame(data)
    data.rename(columns=options,inplace=True)
    return data


def making_df(sexo,citizen,partner,Dependent,PhoneService,MultipleLines,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,Streaming,Paperless,Monthly,Total,ChurnScore,CLTV,InternetService,Contract,PaymenMethod,TenureMonths):
    dict_df = {
        'Gender':enconding(sexo,"Masculino"),
        'Senior Citizen':enconding(partner,"Sí"),
        'Partner':enconding(citizen,"Sí"),
        'Dependents':enconding(Dependent,"Sí"),
        'Phone Service':enconding(PhoneService,"Sí"),
        'Multiple Lines':enconding(MultipleLines,"Sí"),
        'Online Security':enconding(OnlineSecurity,"Sí"),
        'Online Backup':enconding(OnlineBackup,"Sí"),
        'Device Protection':enconding(DeviceProtection,"Sí"),
        'Tech Support':enconding(TechSupport,"Sí"),
        'Streaming TV':enconding(StreamingTV,"Sí"),
        'Streaming Movies':enconding(Streaming,"Sí"),
        'Paperless Billing':enconding(Paperless,"Sí"),
        'Monthly Charges':Monthly,
        'Total Charges':Total,
        'Churn Score':ChurnScore,
        'CLTV':CLTV
    }
    df = pd.DataFrame(dict_df, index=[0])

    options_internet = {"DSL": "Internet Service_DSL", "Fibra optica": "Internet Service_Fiber optic", "No tiene internet": "Internet Service_No"}
    data_internet = encoding_many(options_internet, InternetService)

    options_contract = {"Mes a mes":"Contract_Month-to-month", "Un año de contrato":"Contract_One year", "Dos años de contrato":"Contract_Two year"}
    data_contract = encoding_many(options_contract, Contract)

    options_payment = {"Transferencia bancaria":"Payment Method_Bank transfer (automatic)", "Tarjeta de crédito":'Payment Method_Credit card (automatic)', "e-Check":'Payment Method_Electronic check', 'Enviada por correo':'Payment Method_Mailed check'}
    data_payment = encoding_many(options_payment, PaymenMethod)

    options_tenure = {"1 - 12":'Tenure Group_1 - 12', "13 - 24":'Tenure Group_13 - 24', "25 - 36":'Tenure Group_25 - 36', '37 - 48':'Tenure Group_37 - 48', "49 - 60":'Tenure Group_49 - 60', "61 - 72":'Tenure Group_61 - 72'}
    data_tenure = encoding_many(options_tenure, TenureMonths)

    df = pd.concat([df, data_internet,data_contract,data_payment,data_tenure], axis=1)
    int64_cols = df.select_dtypes(include=['int64']).columns
    df[int64_cols] = df[int64_cols].astype('int32')

    model(df)

def model(df):
    clf_xgb = pd.read_pickle("model_final.pickle")


    y_pred = clf_xgb.predict(df)
    y = "sí" if y_pred == 1 else "no"
    if y == "no":
        st.balloons()
        message = st.chat_message("assistant")
        message.write("El cliente NO se irá dentro de los próximos 3 meses")
    else:
        st.warning('El cliente se irá dentro de los próximos 3 meses', icon="⚠️")


                
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True
    making_df(sexo,citizen,partner,Dependent,PhoneService,MultipleLines,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,Streaming,Paperless,Monthly,Total,ChurnScore,CLTV,InternetService,Contract,PaymenMethod,TenureMonths)   

st.button("Run!", type="primary",on_click=click_button)