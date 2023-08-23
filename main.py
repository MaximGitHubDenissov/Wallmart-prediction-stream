import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import numpy as np



# Настройка страницы
st.set_page_config(
    page_title="Weekly_Sales Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальные переменные
uploaded_file = None

# Создание переменных session state
if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()

# ML section start

# linear regression model
model_file_path = r'C:\Users\77017\OneDrive\Рабочий стол\AI_projects\wallmart-predict-app\final_model2.sav'
model = joblib.load(open(model_file_path, 'rb'))


@st.cache_data
def predict_sales_df(df_input):
    df_input.drop(['Unnamed: 0'],axis=1,inplace=True)
    df_original = df_input.copy()
    df_input['Date'] = pd.to_datetime(df_input['Date'])
    df_input['Date'] = df_input['Date'].dt.strftime('%Y-%m-%d')
    df_input['Week'] = df_input['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%V'))
    df_input.drop(['Date'],axis=1,inplace=True)
    df_input.drop(['Weekly_Sales'],axis=1,inplace=True)
    y_pred = model.predict(df_input)
    y_pred = np.expm1(y_pred).round(1)
    df_original['Weekly_Sales_predicted'] = y_pred
    
    return df_original
@st.cache_data
def predict_sales_one(df_input):
    df_original = df_input.copy()
    df_input['Date'] = pd.to_datetime(df_input['Date'])
    df_input['Date'] = df_input['Date'].dt.strftime('%Y-%m-%d')
    df_input['Week'] = df_input['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%V'))
    y_pred = model.predict(df_input.drop(['Date'],axis=1))
    y_pred = np.expm1(y_pred).round(1)
    df_original['Weekly_Sales_predicted'] = y_pred
    
    return df_original


@st.cache_data
def convert_df(df):
    # Функция для конвертации датафрейма в csv
    return df.to_csv(index=False).encode('utf-8')

# Sidebar section start
# Сайдбар блок
with st.sidebar:
    st.title('🗂 Ввод данных')
    
    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])
    with tab1:
        # Вкладка с загрузкой файла, выбором порога и кнопкой предсказания (вкладка 1)
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv', 'xlsx'], on_change=reset_session_state)
        if uploaded_file is not None:
            
            prediction_button = st.button('Предсказать', type='primary', use_container_width=True, key='button1')
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_sales_df(st.session_state['df_input'])
                st.session_state['tab_selected'] = 'tab1'

    with tab2:
        # Вкладка с вводом данных вручную, выбором порога и кнопкой предсказания (вкладка 2)
        date = st.text_input('Prediction date', placeholder='yyyy-dd-mm', help='Введите дату в формате yyyy-dd-mm')
        store = st.slider( 'Номер магазина', 1, 45, 1, 1, key='slider1')
        holiday_flag = st.selectbox('Праздники на неделе', ('yes', 'no'))
        temperature = st.slider('Температура в фаренгейтах', -20.0, 120.0, 50.0, 0.1, key='slider2')
        fuel_price = st.slider('Цена на бензин', 0.000, 5.000, 2.500, 0.001, key='slider3')
        cpi = st.slider('Индекс потребительских цен', 0.000, 300.000, 150.000, 0.001, key='slider4')
        unemployment = st.slider('Уровень безработицы', 0.000, 20.000, 10.000, 0.001, key='slider5')
    
        # Если введена дата, то показываем слайдер с порогом и кнопку предсказания
        if date != '':
            prediction_button_tab2 = st.button('Предсказать', type='primary', use_container_width=True, key='button2')
            
            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'
                # Сохраняем введенные данные в session state в виде датафрейма
                st.session_state['df_input'] = pd.DataFrame({
                    'Store': store,
                    'Date': date,
                    'Holiday_Flag': 1 if holiday_flag == 'yes' else 0,
                    'Temperature': temperature,
                    'Fuel_Price': fuel_price,
                    'CPI': cpi,
                    'Unemployment': unemployment
                }, index=[0])
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_sales_one(st.session_state['df_input'])

                

# Sidebar section end

# Main section start
# Основной блок
st.image('https://rbkgames.com/files/editor_uploads/Rins/gta/%D0%9D%D0%BE%D0%B2%D0%B0%D1%8F%D0%9F%D0%B0%D0%BF%D0%BA%D0%B0%2056/wall2.jpg', width=400)
st.title('Прогнозирование недельных продаж по магазинам сети Wallmart')

with st.expander("Описание проекта"):
    st.write("""
    В данном проекте мы будем работать с данными о продажах продуктов в магазинах сети Wallmart.
    Наша задача - построить модель, которая будет прогнозировать недельную продажу.
    """)

# Вывод входных данных (из файла или введенных пользователем)
if len(st.session_state['df_input']) > 0:
    # Если предсказание еще не было сделано, то выводим входные данные в общем виде
    if len(st.session_state['df_predicted']) == 0:
        st.subheader('Данные из файла')
        st.write(st.session_state['df_input'])
    else:
        # Если предсказание уже было сделано, то выводим входные данные в expander
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])
    # Примеры визуализации данных
    # st.line_chart(st.session_state['df_input'][['tenure', 'monthlycharges']])
    # st.bar_chart(st.session_state['df_input'][['contract']])


# Выводим результаты предсказания для отдельной даты (вкладка 2)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
    st.image('https://i.pinimg.com/originals/a8/60/ca/a860ca1982b938e1048ab522f0af2802.jpg', width=200)
    st.subheader(f'Прогноз продаж на {st.session_state["df_input"]["Date"][0]}: составит {st.session_state["df_predicted"]["Weekly_Sales_predicted"][0]}')




# Выводим результаты предсказания для клинтов из файла (вкладка 1)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab1':
    # Результаты предсказания для всех клиентов в файле
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    # Скачиваем результаты предсказания для всех клиентов в файле
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-churn-predicted-all.csv',
        mime='text/csv',
    )

    