from flask import Flask, request, render_template
import pickle

# WSGI app
application = Flask(__name__)
app = application

# Load model artifacts
ridge_model = pickle.load(open('model_ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        user_row = {
            'Temperature': Temperature,
            'RH': RH,
            'Ws': Ws,
            'Rain': Rain,
            'FFMC': FFMC,
            'DMC': DMC,
            'ISI': ISI,
            'Classes': Classes,
            'Region': Region,
        }

        # Rebuild the exact training-time feature vector expected by StandardScaler.
        # `feature_columns` defines the exact order and set of columns used during training.
        # If the training data contained columns we don't have inputs for (e.g., 'Unnamed: 0'),
        # we fill them with 0 so inference won't crash.
        X_row = [[float(user_row.get(col, 0.0)) for col in feature_columns]]

        new_data_scaled = standard_scaler.transform(X_row)
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])

    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

