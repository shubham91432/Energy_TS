import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

st.set_page_config('Energy consumption by Company:',layout='wide')
st.title('Energy consumption by Company')
st.markdown('___')

# DATA OF COMPANY AEP
df_AEP = pd.read_csv("newTS/AEP_hourly.csv")
df_AEP.set_index('Datetime')
df_AEP.Datetime = pd.to_datetime(df_AEP.Datetime)


# DATA OF COMPANY PJME

df_PJME = pd.read_csv("newTS/PJME_hourly.csv")
df_PJME.set_index('Datetime')
df_PJME.Datetime = pd.to_datetime(df_PJME.Datetime)


# DATA OF COMPANY DAYTON

df_DAYTON = pd.read_csv("newTS/DAYTON_hourly.csv")
df_DAYTON.set_index('Datetime')
df_DAYTON.Datetime = pd.to_datetime(df_DAYTON.Datetime)


# DATA OF COMPANY PJMW

df_PJMW = pd.read_csv("newTS/PJMW_hourly.csv")
df_PJMW.set_index('Datetime')
df_PJMW.Datetime = pd.to_datetime(df_PJMW.Datetime)


# DATA OF COMPANY DOM

df_DOM = pd.read_csv("newTS/DOM_hourly.csv")
df_DOM.set_index('Datetime')
df_DOM.Datetime = pd.to_datetime(df_DOM.Datetime)


selected_comp = st.selectbox("****Select the company****", ["AEP", "PJME", "DAYTON", "PJMW", "DOM"])

if selected_comp is not None:
    if selected_comp == "AEP":
        # Visualization of data
        st.subheader('*This is the normal visualization of data for  selected company.*' )
        
        nor_visualization = px.scatter(df_AEP,x = 'Datetime',y = 'AEP_MW',title='AEP Energy usage in MW')
        nor_visualization.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(nor_visualization)
        train = df_AEP.loc[df_AEP.Datetime < '2015-01-01']
        test = df_AEP.loc[df_AEP.Datetime >= '2015-01-01']
        
       
        from_date = st.text_input('Enter the date strat from 2015 and format is YYYY-MM-DD')
        end_Date = st.text_input('Enter the date end till 2019 and format is YYYY-MM-DD')
        date_show_btn = st.button("Show data")

        if date_show_btn:
            st.markdown('___')
            st.subheader('*The trendline for selected date.*')
            selected_date = df_AEP.loc[(df_AEP.Datetime > str(from_date)) & (df_AEP.Datetime <= str(end_Date))]
            trend_for_date = px.scatter(selected_date,x = 'Datetime',y = 'AEP_MW',trendline='rolling',trendline_options=dict(window=5, win_type= 'gaussian', function_args = dict(std=2)))
            trend_for_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(trend_for_date)


        def create_features(df_AEP):
            df_AEP = df_AEP.copy()
            df_AEP['hour'] = df_AEP.Datetime.dt.hour
            df_AEP['dayofweek'] = df_AEP.Datetime.dt.dayofweek
            df_AEP['quarter'] = df_AEP.Datetime.dt.quarter
            df_AEP['month'] = df_AEP.Datetime.dt.month
            df_AEP['year'] = df_AEP.Datetime.dt.year
            df_AEP['dayofyear'] = df_AEP.Datetime.dt.dayofyear
            df_AEP['dayofmonth'] = df_AEP.Datetime.dt.day
            df_AEP['weekofyear'] = df_AEP.Datetime.dt.isocalendar().week
            return df_AEP

        df_AEP = create_features(df_AEP)
        features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
        target = ['AEP_MW']
        
        
        train = create_features(train)
        test = create_features(test)

        x_train = train[features]
        x_test = test[features]

        y_train = train[target]
        y_test = test[target]
        reg = xgb.XGBRegressor(n_estimators=1000, booster='gbtree', base_score=0.5, early_stopping_rounds=50, objective='reg:linear', max_depth=5, learning_rate=0.01)
        reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

        fi = pd.DataFrame(data=reg.feature_importances_, index = reg.feature_names_in_, columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()
        st.markdown('___')
        st.subheader('*Predecting test data.*')
        test['prediction'] = reg.predict(x_test)
        df_AEP = df_AEP.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        prediction = px.scatter(df_AEP,x='Datetime',y=['AEP_MW','prediction'])
        prediction.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(prediction)


        st.markdown('___')
        st.subheader('*Comparison trendline graph for selected date.*')
        df1 = df_AEP.loc[(df_AEP.Datetime > str(from_date)) & (df_AEP.Datetime <= str(end_Date))]
        comparison_graph = px.scatter(df1,x='Datetime',y=['AEP_MW','prediction'],trendline='rolling',trendline_options=dict(window =5))
        comparison_graph.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(comparison_graph)

        # Predction for a date 

        predicted_date = st.text_input("Enter the date that you want to predict the format is YYYY-MM-DD")
        date_for_prediction = pd.to_datetime(predicted_date)
        date_show_btn1 = st.button("Show Prediction")
        if date_show_btn1:
            ls = []
            for i in range(0,24 , 1):
                t = date_for_prediction + pd.Timedelta(hours = i)
                ls.append(t)

            date_for_prediction = pd.DataFrame(ls , columns= ["Date"])

            def create_features(date_for_prediction):
                date_for_prediction['hour'] = date_for_prediction.Date.dt.hour
                date_for_prediction['dayofweek'] = date_for_prediction.Date.dt.dayofweek
                date_for_prediction['quarter'] = date_for_prediction.Date.dt.quarter
                date_for_prediction['month'] = date_for_prediction.Date.dt.month
                date_for_prediction['year'] = date_for_prediction.Date.dt.year
                date_for_prediction['dayofyear'] = date_for_prediction.Date.dt.dayofyear
                date_for_prediction['dayofmonth'] = date_for_prediction.Date.dt.day
                date_for_prediction['weekofyear'] = date_for_prediction.Date.dt.isocalendar().week
                return date_for_prediction
            
            date_for_prediction = create_features(date_for_prediction)
            date_for_prediction1 = date_for_prediction.drop(['Date','dayofmonth','weekofyear'],axis=1)
            date_for_prediction['FPredicted'] = reg.predict(date_for_prediction1)
            date_for_prediction = date_for_prediction[['Date','FPredicted']]
            st.dataframe(date_for_prediction)

            sums = date_for_prediction['FPredicted'].sum()
            st.subheader(f'The total energy consumption for Date  {predicted_date} is :-   {sums}')
            Prediction_sel_date = px.scatter(date_for_prediction,x='Date',y='FPredicted',trendline='rolling',trendline_options=dict(window =5))
            Prediction_sel_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(Prediction_sel_date)
        

    if selected_comp == "PJME":
        # Visualization of data
        st.subheader('*This is the normal visualization of data for  selected company.*' )
        
        nor_visualization = px.scatter(df_PJME,x = 'Datetime',y = 'PJME_MW',title='PJME Energy usage in MW')
        nor_visualization.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(nor_visualization)
        train = df_PJME.loc[df_PJME.Datetime < '2015-01-01']
        test = df_PJME.loc[df_PJME.Datetime >= '2015-01-01']
        
       
        from_date = st.text_input('Enter the date strat from 2015 and format is YYYY-MM-DD')
        end_Date = st.text_input('Enter the date end till 2019 and format is YYYY-MM-DD')
        date_show_btn = st.button("Show data")

        if date_show_btn:
            st.markdown('___')
            st.subheader('*The trendline for selected date.*')
            selected_date = df_PJME.loc[(df_PJME.Datetime > str(from_date)) & (df_PJME.Datetime <= str(end_Date))]
            trend_for_date = px.scatter(selected_date,x = 'Datetime',y = 'PJME_MW',trendline='rolling',trendline_options=dict(window=5, win_type= 'gaussian', function_args = dict(std=2)))
            trend_for_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(trend_for_date)


        def create_features(df_PJME):
            df_PJME = df_PJME.copy()
            df_PJME['hour'] = df_PJME.Datetime.dt.hour
            df_PJME['dayofweek'] = df_PJME.Datetime.dt.dayofweek
            df_PJME['quarter'] = df_PJME.Datetime.dt.quarter
            df_PJME['month'] = df_PJME.Datetime.dt.month
            df_PJME['year'] = df_PJME.Datetime.dt.year
            df_PJME['dayofyear'] = df_PJME.Datetime.dt.dayofyear
            df_PJME['dayofmonth'] = df_PJME.Datetime.dt.day
            df_PJME['weekofyear'] = df_PJME.Datetime.dt.isocalendar().week
            return df_PJME

        df_PJME = create_features(df_PJME)
        features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
        target = ['PJME_MW']
        
        
        train = create_features(train)
        test = create_features(test)

        x_train = train[features]
        x_test = test[features]

        y_train = train[target]
        y_test = test[target]
        reg = xgb.XGBRegressor(n_estimators=1000, booster='gbtree', base_score=0.5, early_stopping_rounds=50, objective='reg:linear', max_depth=5, learning_rate=0.01)
        reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

        fi = pd.DataFrame(data=reg.feature_importances_, index = reg.feature_names_in_, columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()
        st.markdown('___')
        st.subheader('*Predecting test data.*')
        test['prediction'] = reg.predict(x_test)
        df_PJME = df_PJME.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        prediction = px.scatter(df_PJME,x='Datetime',y=['PJME_MW','prediction'])
        prediction.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(prediction)


        st.markdown('___')
        st.subheader('*Comparison trendline graph for selected date.*')
        df1 = df_PJME.loc[(df_PJME.Datetime > str(from_date)) & (df_PJME.Datetime <= str(end_Date))]
        comparison_graph = px.scatter(df1,x='Datetime',y=['PJME_MW','prediction'],trendline='rolling',trendline_options=dict(window =5))
        comparison_graph.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(comparison_graph)

        # Predction for a date 

        predicted_date = st.text_input("Enter the date that you want to predict the format is YYYY-MM-DD")
        date_for_prediction = pd.to_datetime(predicted_date)
        date_show_btn1 = st.button("Show Prediction")
        if date_show_btn1:
            ls = []
            for i in range(0,24 , 1):
                t = date_for_prediction + pd.Timedelta(hours = i)
                ls.append(t)

            date_for_prediction = pd.DataFrame(ls , columns= ["Date"])

            def create_features(date_for_prediction):
                date_for_prediction['hour'] = date_for_prediction.Date.dt.hour
                date_for_prediction['dayofweek'] = date_for_prediction.Date.dt.dayofweek
                date_for_prediction['quarter'] = date_for_prediction.Date.dt.quarter
                date_for_prediction['month'] = date_for_prediction.Date.dt.month
                date_for_prediction['year'] = date_for_prediction.Date.dt.year
                date_for_prediction['dayofyear'] = date_for_prediction.Date.dt.dayofyear
                date_for_prediction['dayofmonth'] = date_for_prediction.Date.dt.day
                date_for_prediction['weekofyear'] = date_for_prediction.Date.dt.isocalendar().week
                return date_for_prediction
            
            date_for_prediction = create_features(date_for_prediction)
            date_for_prediction1 = date_for_prediction.drop(['Date','dayofmonth','weekofyear'],axis=1)
            date_for_prediction['FPredicted'] = reg.predict(date_for_prediction1)
            date_for_prediction = date_for_prediction[['Date','FPredicted']]
            st.dataframe(date_for_prediction)

            sums = date_for_prediction['FPredicted'].sum()
            st.subheader(f'The total energy consumption for Date  {predicted_date} is :-   {sums}')
            Prediction_sel_date = px.scatter(date_for_prediction,x='Date',y='FPredicted',trendline='rolling',trendline_options=dict(window =5))
            Prediction_sel_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(Prediction_sel_date)



    if selected_comp == "DAYTON":
        # Visualization of data
        st.subheader('*This is the normal visualization of data for  selected company.*' )
        
        nor_visualization = px.scatter(df_DAYTON,x = 'Datetime',y = 'DAYTON_MW',title='DAYTON Energy usage in MW')
        nor_visualization.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(nor_visualization)
        train = df_DAYTON.loc[df_DAYTON.Datetime < '2015-01-01']
        test = df_DAYTON.loc[df_DAYTON.Datetime >= '2015-01-01']
        
       
        from_date = st.text_input('Enter the date strat from 2015 and format is YYYY-MM-DD')
        end_Date = st.text_input('Enter the date end till 2019 and format is YYYY-MM-DD')
        date_show_btn = st.button("Show data")

        if date_show_btn:
            st.markdown('___')
            st.subheader('*The trendline for selected date.*')
            selected_date = df_DAYTON.loc[(df_DAYTON.Datetime > str(from_date)) & (df_DAYTON.Datetime <= str(end_Date))]
            trend_for_date = px.scatter(selected_date,x = 'Datetime',y = 'DAYTON_MW',trendline='rolling',trendline_options=dict(window=5, win_type= 'gaussian', function_args = dict(std=2)))
            trend_for_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(trend_for_date)


        def create_features(df_DAYTON):
            df_DAYTON = df_DAYTON.copy()
            df_DAYTON['hour'] = df_DAYTON.Datetime.dt.hour
            df_DAYTON['dayofweek'] = df_DAYTON.Datetime.dt.dayofweek
            df_DAYTON['quarter'] = df_DAYTON.Datetime.dt.quarter
            df_DAYTON['month'] = df_DAYTON.Datetime.dt.month
            df_DAYTON['year'] = df_DAYTON.Datetime.dt.year
            df_DAYTON['dayofyear'] = df_DAYTON.Datetime.dt.dayofyear
            df_DAYTON['dayofmonth'] = df_DAYTON.Datetime.dt.day
            df_DAYTON['weekofyear'] = df_DAYTON.Datetime.dt.isocalendar().week
            return df_DAYTON

        df_DAYTON = create_features(df_DAYTON)
        features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
        target = ['DAYTON_MW']
        
        
        train = create_features(train)
        test = create_features(test)

        x_train = train[features]
        x_test = test[features]

        y_train = train[target]
        y_test = test[target]
        reg = xgb.XGBRegressor(n_estimators=1000, booster='gbtree', base_score=0.5, early_stopping_rounds=50, objective='reg:linear', max_depth=5, learning_rate=0.01)
        reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

        fi = pd.DataFrame(data=reg.feature_importances_, index = reg.feature_names_in_, columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()
        st.markdown('___')
        st.subheader('*Predecting test data.*')
        test['prediction'] = reg.predict(x_test)
        df_DAYTON = df_DAYTON.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        prediction = px.scatter(df_DAYTON,x='Datetime',y=['DAYTON_MW','prediction'])
        prediction.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(prediction)


        st.markdown('___')
        st.subheader('*Comparison trendline graph for selected date.*')
        df1 = df_DAYTON.loc[(df_DAYTON.Datetime > str(from_date)) & (df_DAYTON.Datetime <= str(end_Date))]
        comparison_graph = px.scatter(df1,x='Datetime',y=['DAYTON_MW','prediction'],trendline='rolling',trendline_options=dict(window =5))
        comparison_graph.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(comparison_graph)

        # Predction for a date 

        predicted_date = st.text_input("Enter the date that you want to predict the format is YYYY-MM-DD")
        date_for_prediction = pd.to_datetime(predicted_date)
        date_show_btn1 = st.button("Show Prediction")
        if date_show_btn1:
            ls = []
            for i in range(0,24 , 1):
                t = date_for_prediction + pd.Timedelta(hours = i)
                ls.append(t)

            date_for_prediction = pd.DataFrame(ls , columns= ["Date"])

            def create_features(date_for_prediction):
                date_for_prediction['hour'] = date_for_prediction.Date.dt.hour
                date_for_prediction['dayofweek'] = date_for_prediction.Date.dt.dayofweek
                date_for_prediction['quarter'] = date_for_prediction.Date.dt.quarter
                date_for_prediction['month'] = date_for_prediction.Date.dt.month
                date_for_prediction['year'] = date_for_prediction.Date.dt.year
                date_for_prediction['dayofyear'] = date_for_prediction.Date.dt.dayofyear
                date_for_prediction['dayofmonth'] = date_for_prediction.Date.dt.day
                date_for_prediction['weekofyear'] = date_for_prediction.Date.dt.isocalendar().week
                return date_for_prediction
            
            date_for_prediction = create_features(date_for_prediction)
            date_for_prediction1 = date_for_prediction.drop(['Date','dayofmonth','weekofyear'],axis=1)
            date_for_prediction['FPredicted'] = reg.predict(date_for_prediction1)
            date_for_prediction = date_for_prediction[['Date','FPredicted']]
            st.dataframe(date_for_prediction)

            sums = date_for_prediction['FPredicted'].sum()
            st.subheader(f'The total energy consumption for Date  {predicted_date} is :-   {sums}')
            Prediction_sel_date = px.scatter(date_for_prediction,x='Date',y='FPredicted',trendline='rolling',trendline_options=dict(window =5))
            Prediction_sel_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(Prediction_sel_date)



    if selected_comp == "PJMW":
        # Visualization of data
        st.subheader('*This is the normal visualization of data for  selected company.*' )
        
        nor_visualization = px.scatter(df_PJMW,x = 'Datetime',y = 'PJMW_MW',title='PJMW Energy usage in MW')
        nor_visualization.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(nor_visualization)
        train = df_PJMW.loc[df_PJMW.Datetime < '2015-01-01']
        test = df_PJMW.loc[df_PJMW.Datetime >= '2015-01-01']
        
       
        from_date = st.text_input('Enter the date strat from 2015 and format is YYYY-MM-DD')
        end_Date = st.text_input('Enter the date end till 2019 and format is YYYY-MM-DD')
        date_show_btn = st.button("Show data")

        if date_show_btn:
            st.markdown('___')
            st.subheader('*The trendline for selected date.*')
            selected_date = df_PJMW.loc[(df_PJMW.Datetime > str(from_date)) & (df_PJMW.Datetime <= str(end_Date))]
            trend_for_date = px.scatter(selected_date,x = 'Datetime',y = 'PJMW_MW',trendline='rolling',trendline_options=dict(window=5, win_type= 'gaussian', function_args = dict(std=2)))
            trend_for_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(trend_for_date)


        def create_features(df_PJMW):
            df_PJMW = df_PJMW.copy()
            df_PJMW['hour'] = df_PJMW.Datetime.dt.hour
            df_PJMW['dayofweek'] = df_PJMW.Datetime.dt.dayofweek
            df_PJMW['quarter'] = df_PJMW.Datetime.dt.quarter
            df_PJMW['month'] = df_PJMW.Datetime.dt.month
            df_PJMW['year'] = df_PJMW.Datetime.dt.year
            df_PJMW['dayofyear'] = df_PJMW.Datetime.dt.dayofyear
            df_PJMW['dayofmonth'] = df_PJMW.Datetime.dt.day
            df_PJMW['weekofyear'] = df_PJMW.Datetime.dt.isocalendar().week
            return df_PJMW

        df_PJMW = create_features(df_PJMW)
        features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
        target = ['PJMW_MW']
        
        
        train = create_features(train)
        test = create_features(test)

        x_train = train[features]
        x_test = test[features]

        y_train = train[target]
        y_test = test[target]
        reg = xgb.XGBRegressor(n_estimators=1000, booster='gbtree', base_score=0.5, early_stopping_rounds=50, objective='reg:linear', max_depth=5, learning_rate=0.01)
        reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

        fi = pd.DataFrame(data=reg.feature_importances_, index = reg.feature_names_in_, columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()
        st.markdown('___')
        st.subheader('*Predecting test data.*')
        test['prediction'] = reg.predict(x_test)
        df_PJMW = df_PJMW.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        prediction = px.scatter(df_PJMW,x='Datetime',y=['PJMW_MW','prediction'])
        prediction.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(prediction)


        st.markdown('___')
        st.subheader('*Comparison trendline graph for selected date.*')
        df1 = df_PJMW.loc[(df_PJMW.Datetime > str(from_date)) & (df_PJMW.Datetime <= str(end_Date))]
        comparison_graph = px.scatter(df1,x='Datetime',y=['PJMW_MW','prediction'],trendline='rolling',trendline_options=dict(window =5))
        comparison_graph.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(comparison_graph)

        # Predction for a date 

        predicted_date = st.text_input("Enter the date that you want to predict the format is YYYY-MM-DD")
        date_for_prediction = pd.to_datetime(predicted_date)
        date_show_btn1 = st.button("Show Prediction")
        if date_show_btn1:
            ls = []
            for i in range(0,24 , 1):
                t = date_for_prediction + pd.Timedelta(hours = i)
                ls.append(t)

            date_for_prediction = pd.DataFrame(ls , columns= ["Date"])

            def create_features(date_for_prediction):
                date_for_prediction['hour'] = date_for_prediction.Date.dt.hour
                date_for_prediction['dayofweek'] = date_for_prediction.Date.dt.dayofweek
                date_for_prediction['quarter'] = date_for_prediction.Date.dt.quarter
                date_for_prediction['month'] = date_for_prediction.Date.dt.month
                date_for_prediction['year'] = date_for_prediction.Date.dt.year
                date_for_prediction['dayofyear'] = date_for_prediction.Date.dt.dayofyear
                date_for_prediction['dayofmonth'] = date_for_prediction.Date.dt.day
                date_for_prediction['weekofyear'] = date_for_prediction.Date.dt.isocalendar().week
                return date_for_prediction
            
            date_for_prediction = create_features(date_for_prediction)
            date_for_prediction1 = date_for_prediction.drop(['Date','dayofmonth','weekofyear'],axis=1)
            date_for_prediction['FPredicted'] = reg.predict(date_for_prediction1)
            date_for_prediction = date_for_prediction[['Date','FPredicted']]
            st.dataframe(date_for_prediction)

            sums = date_for_prediction['FPredicted'].sum()
            st.subheader(f'The total energy consumption for Date  {predicted_date} is :-   {sums}')
            Prediction_sel_date = px.scatter(date_for_prediction,x='Date',y='FPredicted',trendline='rolling',trendline_options=dict(window =5))
            Prediction_sel_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(Prediction_sel_date)



    if selected_comp == "DOM":
        # Visualization of data
        st.subheader('*This is the normal visualization of data for  selected company.*' )
        
        nor_visualization = px.scatter(df_DOM,x = 'Datetime',y = 'DOM_MW',title='DOM Energy usage in MW')
        nor_visualization.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(nor_visualization)
        train = df_DOM.loc[df_DOM.Datetime < '2015-01-01']
        test = df_DOM.loc[df_DOM.Datetime >= '2015-01-01']
        
       
        from_date = st.text_input('Enter the date strat from 2015 and format is YYYY-MM-DD')
        end_Date = st.text_input('Enter the date end till 2019 and format is YYYY-MM-DD')
        date_show_btn = st.button("Show data")

        if date_show_btn:
            st.markdown('___')
            st.subheader('*The trendline for selected date.*')
            selected_date = df_DOM.loc[(df_DOM.Datetime > str(from_date)) & (df_DOM.Datetime <= str(end_Date))]
            trend_for_date = px.scatter(selected_date,x = 'Datetime',y = 'DOM_MW',trendline='rolling',trendline_options=dict(window=5, win_type= 'gaussian', function_args = dict(std=2)))
            trend_for_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(trend_for_date)


        def create_features(df_DOM):
            df_DOM = df_DOM.copy()
            df_DOM['hour'] = df_DOM.Datetime.dt.hour
            df_DOM['dayofweek'] = df_DOM.Datetime.dt.dayofweek
            df_DOM['quarter'] = df_DOM.Datetime.dt.quarter
            df_DOM['month'] = df_DOM.Datetime.dt.month
            df_DOM['year'] = df_DOM.Datetime.dt.year
            df_DOM['dayofyear'] = df_DOM.Datetime.dt.dayofyear
            df_DOM['dayofmonth'] = df_DOM.Datetime.dt.day
            df_DOM['weekofyear'] = df_DOM.Datetime.dt.isocalendar().week
            return df_DOM

        df_DOM = create_features(df_DOM)
        features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
        target = ['DOM_MW']
        
        
        train = create_features(train)
        test = create_features(test)

        x_train = train[features]
        x_test = test[features]

        y_train = train[target]
        y_test = test[target]
        reg = xgb.XGBRegressor(n_estimators=1000, booster='gbtree', base_score=0.5, early_stopping_rounds=50, objective='reg:linear', max_depth=5, learning_rate=0.01)
        reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

        fi = pd.DataFrame(data=reg.feature_importances_, index = reg.feature_names_in_, columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()
        st.markdown('___')
        st.subheader('*Predecting test data.*')
        test['prediction'] = reg.predict(x_test)
        df_DOM = df_DOM.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        prediction = px.scatter(df_DOM,x='Datetime',y=['DOM_MW','prediction'])
        prediction.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(prediction)


        st.markdown('___')
        st.subheader('*Comparison trendline graph for selected date.*')
        df1 = df_DOM.loc[(df_DOM.Datetime > str(from_date)) & (df_DOM.Datetime <= str(end_Date))]
        comparison_graph = px.scatter(df1,x='Datetime',y=['DOM_MW','prediction'],trendline='rolling',trendline_options=dict(window =5))
        comparison_graph.update_layout(autosize = False ,width = 1200)
        st.plotly_chart(comparison_graph)

        # Predction for a date 

        predicted_date = st.text_input("Enter the date that you want to predict the format is YYYY-MM-DD")
        date_for_prediction = pd.to_datetime(predicted_date)
        date_show_btn1 = st.button("Show Prediction")
        if date_show_btn1:
            ls = []
            for i in range(0,24 , 1):
                t = date_for_prediction + pd.Timedelta(hours = i)
                ls.append(t)

            date_for_prediction = pd.DataFrame(ls , columns= ["Date"])

            def create_features(date_for_prediction):
                date_for_prediction['hour'] = date_for_prediction.Date.dt.hour
                date_for_prediction['dayofweek'] = date_for_prediction.Date.dt.dayofweek
                date_for_prediction['quarter'] = date_for_prediction.Date.dt.quarter
                date_for_prediction['month'] = date_for_prediction.Date.dt.month
                date_for_prediction['year'] = date_for_prediction.Date.dt.year
                date_for_prediction['dayofyear'] = date_for_prediction.Date.dt.dayofyear
                date_for_prediction['dayofmonth'] = date_for_prediction.Date.dt.day
                date_for_prediction['weekofyear'] = date_for_prediction.Date.dt.isocalendar().week
                return date_for_prediction
            
            date_for_prediction = create_features(date_for_prediction)
            date_for_prediction1 = date_for_prediction.drop(['Date','dayofmonth','weekofyear'],axis=1)
            date_for_prediction['FPredicted'] = reg.predict(date_for_prediction1)
            date_for_prediction = date_for_prediction[['Date','FPredicted']]
            st.dataframe(date_for_prediction)

            sums = date_for_prediction['FPredicted'].sum()
            st.subheader(f'The total energy consumption for Date  {predicted_date} is :-   {sums}')
            Prediction_sel_date = px.scatter(date_for_prediction,x='Date',y='FPredicted',trendline='rolling',trendline_options=dict(window =5))
            Prediction_sel_date.update_layout(autosize = False ,width = 1200)
            st.plotly_chart(Prediction_sel_date)