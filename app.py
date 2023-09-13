from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = joblib.load('model/RF_my_carmodel.pkl')

manufacturer_mapping = {
    "ford":1,
    "chevrolet":2,
    "toyota":3,
    "honda":4,
    "jeep":5, 
    "nissan":6,
    "ram":7,
    "gmc":8,
    "bmw":9,
    "dodge":10,
    "mercedes-benz":11,
    "hyundai":12,
    "subaru":13,
    "volkswagen":14, 
    "kia":15,
    "lexus":16,
    "audi":17,
    "cadillac":18, 
    "chrysler":19,
    "acura":20, 
    "buick":21,
    "mazda":22,
    "infiniti":23,
    "lincoln":24, 
    "mitsubishi":25, 
    "volvo":26, 
    "mini":27,
    "pontiac":28,
    "jaguar":29,
    "rover":29,
    "porsche":30,
    "mercury":31, 
    "saturn":32,
    "alfa-romeo":33,
    "tesla":34,
    "fiat":35, 
    "harley-davidson":36,
    "ferrari":37,
    "datsun":38, 
    "aston-martin":39,
    "landrover":40,
}

condition_mapping = {
    "excellent": 1,
    "good": 2, 
    "like new":3,
    "fair":4, 
    "new":5,
    "salvage":6,
}

cylinders_mapping = {
    "6cylinders": 6, 
    "4cylinders": 4, 
    "8cylinders":8, 
    "10cylinders":10,
    "5cylinders":5, 
    "other":9, 
    "3cylinders":3,
    "12cylinders":12,
}
fuel_mapping = {
    "gas": 0,
    "diesel": 1,
    "hybrid":2, 
    "electric":3, 
    "other":4,
}
transmission_mapping = {
    "automatic": 0,
    "manual": 1,
    "other":2,
}
drive_mapping = {
    "4wd": 0, 
    "fwd": 1, 
    "rwd":2,
}

type_mapping = {
    "sedan": 0,
    "SUV": 1, 
    "pickup":2,
    "truck":3, 
    "other":4,
    "coupe":5,
    "hatchback":6,
    "wagon":7, 
    "van":8, 
    "convertible":9, 
    "mini-van":10,
    "offroad":11, 
    "bus":12,
}

paintcolor_mapping = {
    "white": 0,
    "black": 1, 
    "silver":2, 
    "blue":3, 
    "red":4,
    "grey":5,
    "green":6, 
    "custom":7,
    "brown":8,
    "yellow":9,
    "orange":10,
    "purple":11,
}
@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    year = int(request.form.get('year'))
    manufacturer = request.form.get('manufacturer')
    condition = request.form.get('condition')
    cylinders = request.form.get('cylinders')
    fuel = request.form.get('fuel')
    transmission = request.form.get('transmission')
    drive = request.form.get('drive')
    type = request.form.get('type')
    paintcolor = request.form.get('paintcolor')
    odometer = float(request.form.get('odometer'))

    manufacturer_encoded = manufacturer_mapping.get(manufacturer, -1)
    condition_encoded = condition_mapping.get(condition, -1)
    cylinders_encoded = cylinders_mapping.get(cylinders, -1)
    fuel_encoded = fuel_mapping.get(fuel, -1)
    transmission_encoded = transmission_mapping.get(transmission, -1)
    drive_encoded = drive_mapping.get(drive, -1)
    type_encoded = type_mapping.get(type, -1)
    paintcolor_encoded = paintcolor_mapping.get(paintcolor, -1)

    label_encoded_inputs = [year, manufacturer_encoded, condition_encoded, cylinders_encoded, fuel_encoded, transmission_encoded, drive_encoded, type_encoded, paintcolor_encoded, odometer]
    

   
    prediction = model.predict([label_encoded_inputs])

    return render_template('index.html', pred_value=prediction)


if __name__ == '__main__':
    app.run(debug=True)
