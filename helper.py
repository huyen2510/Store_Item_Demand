import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

from sklearn.preprocessing import MinMaxScaler

class Rossmann( object ):
    def data_cleaning( self, df1 ):
        
        ## 1.1 Rename olumns
        cols_old = ['date', 'store', 'item', 'sales']

        ## 1.3. Data Types
        df1['date'] = pd.to_datetime( df1['date'] )
    
        return df1

    def feature_engineering( self, df2 ):
        
        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # day of week
        df2['day_of_week'] = df2['date'].dt.dayofweek

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # 3.0. STEP 03 - VARIABLE FILTERING
        ## 3.1. Line Filtering
        # df2 = df2[(df2['sales']) != 0]
        
        return df2
    
    def data_preparation( self, df5 ):
        
        ## 5.2. Rescaling
        # year
        mms = MinMaxScaler() 
        df5['year'] = mms.fit_transform( df5[['year']].values ) 
        
        ### 5.3.3. Nature Transformation
        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day
        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        cols_selected = ['store',
                'item',
                'year',
                'day_of_week_sin',
                'day_of_week_cos',
                'month_sin',
                'month_cos',
                'week_of_year_sin',
                'week_of_year_cos']
        return df5[ cols_selected ]
    
    def get_prediction( self, model, original_data, test_data ):
        
        # prediction
        pred = model.predict( test_data._get_numeric_data() )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data

def test():
    model = pickle.load( open( 'C:/Users/Admin/Documents/Đồ án tốt nghiệp/Store_Item_Demand/model/model.pkl', 'rb' ) )

    data = ['1', '2023-01-01', '2023-04-10', '1']
    print("Data:", data)
    store_id = int(data[0])
    from_date = str(data[1])
    to_date = str(data[2])
    item_id = int(data[3])
    sales_dates = pd.date_range(from_date, to_date).date.tolist()
    store_list, item_list = [], []
    for i in range(len(sales_dates)):
        store_list.append(store_id)
        item_list.append(item_id)

    df_test = pd.DataFrame(list(zip(sales_dates, store_list, item_list)), columns =['date', 'store','item'])
    # print(df_test)
    # print(sales_dates[0])
    datetime_tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    pipeline = Rossmann()
                # data cleaning
    df1 = pipeline.data_cleaning( df_test )        
                # feature engineering
    df2 = pipeline.feature_engineering( df1 )
            # data preparation
    df3 = pipeline.data_preparation( df2 )
                # prediction
    df_response = pipeline.get_prediction( model, df_test, df3 )
            
    sales = list(df_response['prediction'])
    
    a = df_response.groupby(df_response['date'].dt.strftime('%y-%m'))['prediction'].sum()
    print(a.values)
    # month = a.index.to_list()
    sm = a.values.tolist()
    print(sm)
    # months_in_order = ['January',
    #                     'February',
    #                     'March',
    #                     'April',
    #                     'May',
    #                     'June',
    #                     'July',
    #                     'August',
    #                     'September',
    #                     'October',
    #                     'November',
    #                     'December']
    # df_test2 = pd.DataFrame(list(zip(month, sm)), columns =['month', 'sales'])
    # print(df_test2)
    # df_test2.month = pd.Categorical(
    #     df_test2.month,
    # categories=months_in_order,
    # ordered=True
    #         )
    # print(df_test2.sort_values('month'))
    # print(round(sum(sales)),sales)

if __name__ == '__main__':
    test()
    