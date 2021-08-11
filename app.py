import requests, json, time, datetime
from flask import Flask, request, jsonify

# database import
from database.database import *

# LSTM model
#from lstm_model import *
from para_apagar_classes import *

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

regressor = keras.models.load_model('multivariate_1000_batch128.h5')
model = RNN_LSTM_Model('DAYTON_hourly.csv', 100, 'Datetime', 'DAYTON_MW', 1000, 200)
X_test = model.get_X_test()
df = model.dataset_preparation()

@app.route("/api/train")
def train():

    # get the last timestamp event to prevent repeated data
    last_timestamp = get_last_timestamp()
    
    # get data from dataset
    #model = RNN_LSTM_Model('DAYTON_hourly.csv', 100, 'Datetime', 'DAYTON_MW', 200, 10)
    model = RNN_LSTM_Model('DAYTON_hourly.csv', 100, 'Datetime', 'DAYTON_MW', 1000, 200)

    dataset = model.dataset_preparation()
    
    for i in range(len(dataset)):
        
        # check for repeated data
        if str(last_timestamp) < str(dataset.index[i]):
            print(str(dataset.index[i]) + ' ' + str(dataset.DAYTON_MW[i]))
            add_energy_consumption(dataset.index[i], dataset.DAYTON_MW[i])
    
    # train the model
    model.train_model()

    return (jsonify({"status": "success", "message": "model's train completed"}), 200)

@app.route("/api/forecast/<int:number_predictions>")
def forecast(number_predictions):

    # call the forecast's class
    #model = RNN_LSTM_Model('DAYTON_hourly.csv', 100, 'Datetime', 'DAYTON_MW', 200, 24)
    #forecast_data = model.forecast(number_predictions)
    
    model = RNN_LSTM_Model('DAYTON_hourly.csv', 100, 'Datetime', 'DAYTON_MW', 1000, 200)
    forecast_data = model.predict_values(regressor, number_predictions)

    # get last timestamp from CSV file
    dataset = model.dataset_preparation()
    last_timestamp = str(dataset.index[-1])
    
    # convert timestamp into datetime
    last_timestamp = datetime.datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')

    # response
    response = {
        "data" : [],
        "success" : True
    }

    # get last timestamp
    last_timestamp_database = get_last_timestamp_forecast()

    for i in range(len(forecast_data)):
        
        # increment one hour
        last_timestamp = last_timestamp + datetime.timedelta(hours = 1)

        if last_timestamp_database != 0:
            if last_timestamp_database < last_timestamp:
                add_energy_consumption_forecast(last_timestamp, round(float(forecast_data[i]), 1))
        
        else:
            add_energy_consumption_forecast(last_timestamp, round(float(forecast_data[i]), 1))
        
        forecasted_data = {
            "timestamp" : last_timestamp,
            "value" : round(float(forecast_data[i]), 1)
        }

        response["data"].append(forecasted_data)
    
    return (jsonify(response), 200)

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    app.run(debug = True, port='5555', host='0.0.0.0')