import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as file:
    clf = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Extract form data
            features = [
                int(request.form.get(name)) for name in [
                    'age', 'gender', 'Polyuria', 'Polydipsia', 'Weight', 'Weakness', 'Polyphagia',
                    'Thrush', 'Blurring', 'Itching', 'Irritability', 'Healing', 'Paresis',
                    'Stiffness', 'Alopecia', 'Obesity'
                ]
            ]
            prediction = clf.predict([features])[0]
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
