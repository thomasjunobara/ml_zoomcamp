# ### Load the model
# - restart the kernel before

import pickle
from pathlib import Path
from flask import Flask
from flask import request
from flask import jsonify

model_file = Path(__file__).parent / 'model_C1.0.bin'

with open(model_file, 'rb') as f_in: # if we don't change the wb to rb, it will overwrite the file
    (dv, model) = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST']) # - we use 'POST', since we want to send information about the customer
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)