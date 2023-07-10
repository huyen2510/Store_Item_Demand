import pickle
import pandas as pd
from datetime import datetime
import traceback
from flask import Flask, request, app, jsonify, render_template
from helper import Rossmann

# loading model
model = pickle.load( open( 'C:/Users/Admin/Documents/Đồ án tốt nghiệp/Store_Item_Demand/model/model.pkl', 'rb' ) )

app = Flask( __name__ )
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST', 'GET'])
def predict_api():
    print("Method:",request.method)
    try:
        if request.method == 'POST':
            data = [x for x in request.form.values()]
            # print("Data:", data)
            store_id = int(data[0])
            from_date = pd.Timestamp(str(data[1])).date().strftime("%d-%m-%Y")
            to_date = pd.Timestamp(str(data[2])).date().strftime("%d-%m-%Y")
            item_id = int(data[3])
            
            sales_dates = pd.date_range(from_date, to_date).date.tolist()
            labels = [d.strftime('%Y-%m-%d') for d in pd.date_range(from_date, to_date)]
            store_list, item_list = [], []
            for i in range(len(sales_dates)):
                store_list.append(store_id)
                item_list.append(item_id)

            df_test = pd.DataFrame(list(zip(sales_dates, store_list, item_list)), columns =['date', 'store','item'])
            print(df_test)

            pipeline = Rossmann()
                # data cleaning
            df1 = pipeline.data_cleaning( df_test )        
                # feature engineering
            df2 = pipeline.feature_engineering( df1 )
                # data preparation
            df3 = pipeline.data_preparation( df2 )
                # prediction
            df_response = pipeline.get_prediction( model, df_test, df3 )

            sales = list(round(df_response['prediction']))
            # order by month
            sr_month = df_response.groupby(df_response['date'].dt.strftime('%y-%m'))['prediction'].sum()
            month = sr_month.index.to_list()
            salesperm = sr_month.values.tolist()
            
            return render_template('index.html', total = str(round(sum(sales))), labels= labels , values= sales, 
                                    month_labels= month, sales_per_month = salesperm , storeid= store_id, from_date=from_date, to_date=to_date, item_id= item_id)
        else:
            return render_template('index.html')
    except Exception as e:
        print("Error")
        print(type(e))
        print(e)
        print(traceback.print_exc())
        
if __name__ == '__main__':
    app.run( '0.0.0.0' )